# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions for K-FAC integration with verl's parallel models.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn


def get_parallel_mlp_layers(
    model: nn.Module,
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, Tuple[nn.Module, str]]:
    """
    Get MLP layers from a parallel llama model.

    Args:
        model: ParallelLlamaForCausalLMRmPad or similar model.
        layer_indices: Optional list of layer indices. If None, returns all layers.

    Returns:
        Dict mapping layer names to (layer, projection_type) tuples.
        Example: {'blk0.gate_up': (gate_up_proj_layer, 'gate_up'),
                  'blk0.down': (down_proj_layer, 'down')}
    """
    # Get the model's decoder layers
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Could not find decoder layers in model")

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    result = {}
    for idx in layer_indices:
        if idx >= len(layers):
            raise ValueError(f"Layer index {idx} out of range (model has {len(layers)} layers)")

        mlp = layers[idx].mlp

        if hasattr(mlp, 'gate_up_proj'):
            result[f'blk{idx}.gate_up'] = (mlp.gate_up_proj, 'gate_up')

        if hasattr(mlp, 'down_proj'):
            result[f'blk{idx}.down'] = (mlp.down_proj, 'down')

    return result


def merge_kfac_factors(
    factor_files: List[str],
    output_path: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Merge K-FAC factor files from multiple passes into a single file.

    This is useful when K-FAC factors were collected in multiple passes
    (e.g., different layer groups) and need to be combined.

    Args:
        factor_files: List of paths to K-FAC factor files.
        output_path: Path to save the merged factors.

    Returns:
        Merged factors dict.
    """
    merged = {}

    for factor_file in factor_files:
        if not os.path.exists(factor_file):
            raise FileNotFoundError(f"Factor file not found: {factor_file}")

        data = torch.load(factor_file, map_location='cpu')

        for key, value in data.items():
            if key in merged:
                # Check for duplicates
                raise ValueError(f"Duplicate key {key} found in {factor_file}")
            merged[key] = value

    # Save merged factors
    torch.save(merged, output_path)
    print(f"Merged {len(merged)} layer factors to {output_path}")

    return merged


def split_merged_gate_up_factors(
    merged_factors: Dict[str, Dict[str, torch.Tensor]],
    intermediate_size: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Split K-FAC factors from merged gate_up projection into separate gate and up factors.

    When collecting factors from MergedColumnParallelLinear, the G matrix covers
    both gate and up projections. This function splits them for separate treatment.

    Args:
        merged_factors: Dict with keys like 'blk0.gate_up' containing A and G matrices.
        intermediate_size: The model's intermediate_size (gate and up output dimension).

    Returns:
        Dict with separate 'blk0.gate' and 'blk0.up' entries.
    """
    split_factors = {}

    for key, factors in merged_factors.items():
        if 'gate_up' in key:
            # This is a merged factor, split it
            A = factors['A']  # Shared input covariance
            G = factors['G']  # Combined output covariance [2*inter, 2*inter]

            # Split G into gate and up portions
            G_gate = G[:intermediate_size, :intermediate_size]
            G_up = G[intermediate_size:, intermediate_size:]

            base_key = key.replace('.gate_up', '')

            split_factors[f'{base_key}.gate'] = {
                'A': A.clone(),
                'G': G_gate,
                'n_tokens': factors.get('n_tokens', 0),
            }
            split_factors[f'{base_key}.up'] = {
                'A': A.clone(),
                'G': G_up,
                'n_tokens': factors.get('n_tokens', 0),
            }
        else:
            # Not a merged factor, keep as is
            split_factors[key] = factors

    return split_factors


def collect_kfac_for_parallel_model(
    model: nn.Module,
    dataloader,
    layer_indices: List[int],
    intermediate_size: int,
    device: str = "cuda",
    sample_labels: bool = False,
    layers_per_pass: int = 1,
    save_dir: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collect K-FAC factors from a parallel llama model.

    This is a high-level function that handles:
    - Setting up collectors for parallel MLP layers
    - Running forward/backward passes
    - Synchronizing factors across tensor parallel ranks
    - Optionally saving factors to disk

    Args:
        model: The parallel llama model.
        dataloader: DataLoader yielding batches with 'input_ids' and optionally 'attention_mask'.
        layer_indices: Which layers to collect factors from.
        intermediate_size: Model's intermediate_size.
        device: Device to run on.
        sample_labels: If True, use sampled labels instead of teacher forcing.
        layers_per_pass: How many layers to process per forward/backward pass.
        save_dir: Optional directory to save factors.

    Returns:
        Dict of K-FAC factors.
    """
    from tqdm import tqdm

    from verl.models.memorization_kfac.parallel_kfac import MergedKFACCollector, ParallelKFACCollector

    model.train()
    model.gradient_checkpointing_enable()

    all_factors = {}

    # Process layers in groups to manage memory
    for start in range(0, len(layer_indices), layers_per_pass):
        end = min(start + layers_per_pass, len(layer_indices))
        current_layers = layer_indices[start:end]

        # Disable gradients for all parameters
        for p in model.parameters():
            p.requires_grad_(False)

        # Get collectors for current layers
        collectors = {}

        if hasattr(model, 'model'):
            layers = model.model.layers
        else:
            layers = model.layers

        for idx in current_layers:
            mlp = layers[idx].mlp

            # Merged gate_up collector
            if hasattr(mlp, 'gate_up_proj'):
                mlp.gate_up_proj.weight.requires_grad_(True)
                collectors[f'blk{idx}'] = MergedKFACCollector(
                    layer=mlp.gate_up_proj,
                    layer_name=f'blk{idx}',
                    intermediate_size=intermediate_size,
                    device=device,
                )

            # Down projection collector
            if hasattr(mlp, 'down_proj'):
                mlp.down_proj.weight.requires_grad_(True)
                collectors[f'blk{idx}.down'] = ParallelKFACCollector(
                    layer=mlp.down_proj,
                    layer_name=f'blk{idx}.down',
                    layer_type='row',
                    device=device,
                )

        # Run data through model
        ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

        for batch in tqdm(dataloader, desc=f"K-FAC layers {current_layers}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = (input_ids != model.config.pad_token_id)

            # Create labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            labels[attention_mask == 0] = -100

            model.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1].float()

            if sample_labels:
                with torch.no_grad():
                    sampled = torch.multinomial(
                        torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1)),
                        1
                    ).squeeze(1)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    sampled
                )
            else:
                loss = ce(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, :-1].reshape(-1)
                )

            # Backward pass
            loss.backward()

        # Synchronize and extract factors
        for name, collector in collectors.items():
            collector.synchronize_factors()

            if isinstance(collector, MergedKFACCollector):
                # MergedKFACCollector returns dict with gate and up
                factors = collector.get_factors()
                all_factors.update(factors)
            else:
                all_factors[name] = collector.get_factors()

            collector.close()

        del collectors
        torch.cuda.empty_cache()

        print(f"Processed layers {current_layers}")

    # Optionally save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "kfac_factors.pt")
        torch.save(all_factors, save_path)
        print(f"Saved K-FAC factors to {save_path}")

    return all_factors


def apply_kfac_to_parallel_model(
    model: nn.Module,
    kfac_factors_path: str,
    layer_indices: List[int],
    variance_ratio: float = 0.9,
    method: str = "product",
    projections: List[str] = ["gate", "up", "down"],
) -> Dict[str, Dict]:
    """
    Apply K-FAC compression to a parallel llama model.

    High-level convenience function.

    Args:
        model: The parallel llama model.
        kfac_factors_path: Path to K-FAC factors file.
        layer_indices: Which layers to apply treatment to.
        variance_ratio: Variance ratio to retain (0-1).
        method: "product" for product-based selection, "separate" for separate A/G variance.
        projections: Which projections to treat.

    Returns:
        Compression stats from the treatment.
    """
    from verl.models.memorization_kfac.parallel_kfac import ParallelKFACTreatment

    treatment = ParallelKFACTreatment(
        model=model,
        layer_indices=layer_indices,
        kfac_factors_path=kfac_factors_path,
        projections=projections,
    )

    if method == "product":
        treatment.apply_kfac_by_product(variance_ratio)
    else:
        treatment.apply_kfac((variance_ratio, variance_ratio))

    return treatment.get_compression_summary()
