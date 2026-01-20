#!/usr/bin/env python
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
End-to-end K-FAC pipeline for verl's parallel llama models.

This script provides a complete pipeline for:
1. Collecting K-FAC factors from a model
2. Analyzing the collected factors
3. Applying K-FAC treatment to reduce memorization
4. Evaluating the treatment effects

Usage:
    # Collect K-FAC factors
    python run_kfac_pipeline.py collect --model meta-llama/Llama-2-7b-hf \
        --layers 20 24 28 31 --output_dir ./kfac_output

    # Analyze collected factors
    python run_kfac_pipeline.py analyze --factors_path ./kfac_output/kfac_factors.pt \
        --output_dir ./kfac_analysis

    # Apply treatment and evaluate
    python run_kfac_pipeline.py treat --model meta-llama/Llama-2-7b-hf \
        --factors_path ./kfac_output/kfac_factors.pt --variance_ratio 0.9

    # Run full pipeline
    python run_kfac_pipeline.py full --model meta-llama/Llama-2-7b-hf \
        --layers 20 24 28 31 --output_dir ./kfac_pipeline
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class KFACConfig:
    """Configuration for K-FAC pipeline."""

    # Model settings
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True

    # Data settings
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_samples: int = 1000
    seq_length: int = 512
    batch_size: int = 4

    # K-FAC collection settings
    target_layers: List[int] = None
    layers_per_pass: int = 2
    sample_labels: bool = False

    # Treatment settings
    variance_ratio: float = 0.9
    treatment_method: str = "product"  # "product" or "separate"
    projections: List[str] = None

    # Output settings
    output_dir: str = "./kfac_output"
    save_factors: bool = True
    save_analysis: bool = True

    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [20, 24, 28, 31]
        if self.projections is None:
            self.projections = ["gate", "up", "down"]


# ============================================================================
# Data Loading
# ============================================================================


class TextDataset(IterableDataset):
    """Simple text dataset for K-FAC collection."""

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        seq_length: int = 512,
        max_samples: int = 1000,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_samples = max_samples

        try:
            from datasets import load_dataset

            self.dataset = load_dataset(
                dataset_name, dataset_config, split=split, streaming=True
            )
            self.use_streaming = True
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_name}: {e}")
            print("Using dummy data instead")
            self.dataset = None
            self.use_streaming = False

    def __iter__(self):
        buffer = []
        count = 0

        if self.dataset is None:
            # Generate dummy data
            while count < self.max_samples:
                dummy_ids = torch.randint(100, 30000, (self.seq_length,))
                yield {"input_ids": dummy_ids}
                count += 1
            return

        for sample in self.dataset:
            text = sample.get("text", "")
            if not text.strip():
                continue

            tokens = self.tokenizer(text, add_special_tokens=False).input_ids
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length:
                yield {"input_ids": torch.tensor(buffer[: self.seq_length])}
                buffer = buffer[self.seq_length :]
                count += 1

                if count >= self.max_samples:
                    return


def create_dataloader(config: KFACConfig, tokenizer) -> DataLoader:
    """Create a dataloader for K-FAC collection."""
    dataset = TextDataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        seq_length=config.seq_length,
        max_samples=config.max_samples,
    )

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )


# ============================================================================
# K-FAC Factor Collection
# ============================================================================


class SimpleKFACCollector:
    """
    Simplified K-FAC collector that works with standard HuggingFace models.

    This collector is designed to work with both standard and parallel models.
    """

    def __init__(self, layer: nn.Module, layer_name: str, device: str = "cuda"):
        self.layer = layer
        self.layer_name = layer_name
        self.device = device

        # Get dimensions
        if hasattr(layer, "weight"):
            out_dim, in_dim = layer.weight.shape
        else:
            raise ValueError(f"Layer {layer_name} does not have a weight attribute")

        self.out_dim = out_dim
        self.in_dim = in_dim

        # Initialize accumulators
        self.A = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
        self.G = torch.zeros(out_dim, out_dim, dtype=torch.float32, device=device)
        self.n_tokens = 0

        self._input_buffer = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_pre_hook(module, inputs):
            if torch.is_grad_enabled() and len(inputs) > 0:
                x = inputs[0]
                if x.dim() == 3:
                    # [batch, seq, hidden] -> skip last for autoregressive
                    x = x[:, :-1, :].reshape(-1, x.size(-1))
                elif x.dim() == 2:
                    x = x[:-1, :]
                self._input_buffer = x.detach().float()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is None or self._input_buffer is None:
                return

            g = grad_output[0]
            if g.dim() == 3:
                g = g[:, :-1, :].reshape(-1, g.size(-1))
            elif g.dim() == 2:
                g = g[:-1, :]

            g = g.detach().float()
            x = self._input_buffer

            n = g.size(0)
            self.n_tokens += n

            # Accumulate factors
            self.A.add_(x.T @ x)
            self.G.add_(g.T @ g)

            self._input_buffer = None

        self._hooks.append(self.layer.register_forward_pre_hook(forward_pre_hook))
        self._hooks.append(self.layer.register_full_backward_hook(backward_hook))

    def get_factors(self) -> Dict[str, torch.Tensor]:
        if self.n_tokens == 0:
            raise RuntimeError(f"No tokens collected for {self.layer_name}")

        return {
            "A": (self.A / self.n_tokens).cpu(),
            "G": (self.G / self.n_tokens).cpu(),
            "n_tokens": self.n_tokens,
        }

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._input_buffer = None


def collect_kfac_factors(
    model: nn.Module,
    dataloader: DataLoader,
    target_layers: List[int],
    config: KFACConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collect K-FAC factors from a model.

    Args:
        model: The language model.
        dataloader: DataLoader providing training data.
        target_layers: List of layer indices to collect factors from.
        config: KFACConfig with settings.

    Returns:
        Dictionary mapping layer names to their K-FAC factors.
    """
    print(f"\n{'='*60}")
    print("K-FAC Factor Collection")
    print(f"{'='*60}")
    print(f"Target layers: {target_layers}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.seq_length}")
    print(f"Sample labels: {config.sample_labels}")

    model.train()

    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    all_factors = {}

    # Get model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find model layers")

    # Process layers in groups
    for start in range(0, len(target_layers), config.layers_per_pass):
        end = min(start + config.layers_per_pass, len(target_layers))
        current_layers = target_layers[start:end]

        print(f"\nProcessing layers: {current_layers}")

        # Disable all gradients
        for p in model.parameters():
            p.requires_grad_(False)

        # Setup collectors
        collectors = {}
        for idx in current_layers:
            if idx >= len(layers):
                print(f"Warning: Layer {idx} out of range, skipping")
                continue

            layer = layers[idx]
            mlp = layer.mlp if hasattr(layer, "mlp") else layer

            # Find MLP projections
            proj_names = []
            if hasattr(mlp, "gate_proj"):
                proj_names.append(("gate_proj", "gate"))
            if hasattr(mlp, "up_proj"):
                proj_names.append(("up_proj", "up"))
            if hasattr(mlp, "down_proj"):
                proj_names.append(("down_proj", "down"))
            if hasattr(mlp, "gate_up_proj"):
                proj_names.append(("gate_up_proj", "gate_up"))

            # Fallback for GPT-style models
            if hasattr(mlp, "c_fc"):
                proj_names.append(("c_fc", "up"))
            if hasattr(mlp, "c_proj"):
                proj_names.append(("c_proj", "down"))

            for attr_name, proj_type in proj_names:
                proj_layer = getattr(mlp, attr_name)
                proj_layer.weight.requires_grad_(True)

                key = f"blk{idx}.{proj_type}"
                collectors[key] = SimpleKFACCollector(
                    layer=proj_layer,
                    layer_name=key,
                    device=config.device,
                )

        if not collectors:
            print(f"Warning: No collectors created for layers {current_layers}")
            continue

        # Run forward/backward passes
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        total_tokens = 0

        for batch in tqdm(dataloader, desc=f"Collecting K-FAC (layers {current_layers})"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(config.device)

            # Create labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            model.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1].float()

            if config.sample_labels:
                # Use sampled labels (multinomial)
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1))
                    sampled = torch.multinomial(probs, 1).squeeze(1)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), sampled
                )
            else:
                # Use gold labels
                loss = ce_loss(
                    logits.reshape(-1, logits.size(-1)), labels[:, :-1].reshape(-1)
                )

            # Backward pass
            loss.backward()

            if attention_mask is not None:
                total_tokens += attention_mask[:, 1:].sum().item()
            else:
                total_tokens += input_ids.numel()

        # Extract factors
        for key, collector in collectors.items():
            try:
                factors = collector.get_factors()
                all_factors[key] = factors
                print(f"  {key}: A{tuple(factors['A'].shape)}, G{tuple(factors['G'].shape)}, n={factors['n_tokens']}")
            except RuntimeError as e:
                print(f"  {key}: Failed - {e}")
            finally:
                collector.close()

        del collectors
        torch.cuda.empty_cache()

    print(f"\nTotal factors collected: {len(all_factors)}")
    return all_factors


# ============================================================================
# K-FAC Analysis
# ============================================================================


def analyze_kfac_factors(
    factors: Dict[str, Dict[str, torch.Tensor]],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze K-FAC factors and generate statistics and visualizations.

    Args:
        factors: Dictionary of K-FAC factors.
        output_dir: Optional directory to save analysis results.

    Returns:
        Dictionary with analysis results.
    """
    print(f"\n{'='*60}")
    print("K-FAC Factor Analysis")
    print(f"{'='*60}")

    analysis = {"layers": {}, "summary": {}}

    for layer_name, layer_factors in factors.items():
        print(f"\n{layer_name}:")

        A = layer_factors["A"]
        G = layer_factors["G"]
        n_tokens = layer_factors.get("n_tokens", "N/A")

        # Compute eigendecomposition
        eva_A, evc_A = torch.linalg.eigh(A.float())
        eva_G, evc_G = torch.linalg.eigh(G.float())

        # Sort descending
        eva_A = eva_A.flip(0)
        eva_G = eva_G.flip(0)

        # Compute statistics
        layer_analysis = {
            "n_tokens": n_tokens,
            "A_shape": list(A.shape),
            "G_shape": list(G.shape),
            "A_eigenvalues": {
                "max": eva_A[0].item(),
                "min": eva_A[-1].item(),
                "mean": eva_A.mean().item(),
                "median": eva_A.median().item(),
                "sum": eva_A.sum().item(),
            },
            "G_eigenvalues": {
                "max": eva_G[0].item(),
                "min": eva_G[-1].item(),
                "mean": eva_G.mean().item(),
                "median": eva_G.median().item(),
                "sum": eva_G.sum().item(),
            },
        }

        # Compute variance explained at different thresholds
        cumsum_A = torch.cumsum(eva_A, dim=0) / eva_A.sum()
        cumsum_G = torch.cumsum(eva_G, dim=0) / eva_G.sum()

        thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        variance_ranks = {"A": {}, "G": {}}

        for thresh in thresholds:
            rank_A = (cumsum_A >= thresh).nonzero()[0].item() + 1 if (cumsum_A >= thresh).any() else len(eva_A)
            rank_G = (cumsum_G >= thresh).nonzero()[0].item() + 1 if (cumsum_G >= thresh).any() else len(eva_G)
            variance_ranks["A"][f"{int(thresh*100)}%"] = rank_A
            variance_ranks["G"][f"{int(thresh*100)}%"] = rank_G

        layer_analysis["variance_ranks"] = variance_ranks

        # Condition numbers
        eps = 1e-10
        layer_analysis["A_condition_number"] = (eva_A[0] / (eva_A[-1] + eps)).item()
        layer_analysis["G_condition_number"] = (eva_G[0] / (eva_G[-1] + eps)).item()

        # Effective rank (using entropy)
        def effective_rank(eigenvalues):
            p = eigenvalues / eigenvalues.sum()
            p = p[p > 1e-10]  # Filter near-zero
            entropy = -(p * p.log()).sum()
            return entropy.exp().item()

        layer_analysis["A_effective_rank"] = effective_rank(eva_A)
        layer_analysis["G_effective_rank"] = effective_rank(eva_G)

        analysis["layers"][layer_name] = layer_analysis

        # Print summary
        print(f"  A: shape={A.shape}, effective_rank={layer_analysis['A_effective_rank']:.1f}")
        print(f"     eigenvalues: max={eva_A[0]:.2e}, min={eva_A[-1]:.2e}")
        print(f"     90% variance at rank: {variance_ranks['A']['90%']}/{len(eva_A)}")
        print(f"  G: shape={G.shape}, effective_rank={layer_analysis['G_effective_rank']:.1f}")
        print(f"     eigenvalues: max={eva_G[0]:.2e}, min={eva_G[-1]:.2e}")
        print(f"     90% variance at rank: {variance_ranks['G']['90%']}/{len(eva_G)}")

        # Store eigenvalues for later use
        layer_analysis["_eva_A"] = eva_A.numpy() if HAS_MATPLOTLIB else None
        layer_analysis["_eva_G"] = eva_G.numpy() if HAS_MATPLOTLIB else None

    # Summary statistics
    all_eff_ranks_A = [l["A_effective_rank"] for l in analysis["layers"].values()]
    all_eff_ranks_G = [l["G_effective_rank"] for l in analysis["layers"].values()]

    analysis["summary"] = {
        "num_layers": len(factors),
        "mean_effective_rank_A": sum(all_eff_ranks_A) / len(all_eff_ranks_A) if all_eff_ranks_A else 0,
        "mean_effective_rank_G": sum(all_eff_ranks_G) / len(all_eff_ranks_G) if all_eff_ranks_G else 0,
    }

    # Generate visualizations
    if output_dir and HAS_MATPLOTLIB:
        os.makedirs(output_dir, exist_ok=True)
        _plot_eigenvalue_distributions(analysis, output_dir)
        _plot_variance_explained(analysis, output_dir)
        _plot_effective_ranks(analysis, output_dir)

        # Save analysis JSON
        analysis_json = {
            k: v for k, v in analysis.items() if not k.startswith("_")
        }
        for layer_name in analysis_json.get("layers", {}):
            layer_data = analysis_json["layers"][layer_name]
            layer_data.pop("_eva_A", None)
            layer_data.pop("_eva_G", None)

        with open(os.path.join(output_dir, "analysis.json"), "w") as f:
            json.dump(analysis_json, f, indent=2, default=str)

        print(f"\nAnalysis saved to {output_dir}")

    return analysis


def _plot_eigenvalue_distributions(analysis: Dict, output_dir: str):
    """Plot eigenvalue distributions for all layers."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for layer_name, layer_data in analysis["layers"].items():
        eva_A = layer_data.get("_eva_A")
        eva_G = layer_data.get("_eva_G")

        if eva_A is not None:
            axes[0].semilogy(eva_A, label=layer_name, alpha=0.7)
        if eva_G is not None:
            axes[1].semilogy(eva_G, label=layer_name, alpha=0.7)

    axes[0].set_title("Activation Covariance (A) Eigenvalues")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue (log scale)")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Gradient Covariance (G) Eigenvalues")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Eigenvalue (log scale)")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eigenvalue_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_variance_explained(analysis: Dict, output_dir: str):
    """Plot cumulative variance explained."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for layer_name, layer_data in analysis["layers"].items():
        eva_A = layer_data.get("_eva_A")
        eva_G = layer_data.get("_eva_G")

        if eva_A is not None:
            cumsum_A = np.cumsum(eva_A) / eva_A.sum()
            axes[0].plot(np.arange(len(cumsum_A)) / len(cumsum_A), cumsum_A, label=layer_name, alpha=0.7)

        if eva_G is not None:
            cumsum_G = np.cumsum(eva_G) / eva_G.sum()
            axes[1].plot(np.arange(len(cumsum_G)) / len(cumsum_G), cumsum_G, label=layer_name, alpha=0.7)

    for ax in axes:
        ax.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="90% threshold")
        ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, label="95% threshold")

    axes[0].set_title("Cumulative Variance Explained (A)")
    axes[0].set_xlabel("Fraction of Components")
    axes[0].set_ylabel("Variance Explained")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Cumulative Variance Explained (G)")
    axes[1].set_xlabel("Fraction of Components")
    axes[1].set_ylabel("Variance Explained")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "variance_explained.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_effective_ranks(analysis: Dict, output_dir: str):
    """Plot effective ranks across layers."""
    layers = list(analysis["layers"].keys())
    eff_ranks_A = [analysis["layers"][l]["A_effective_rank"] for l in layers]
    eff_ranks_G = [analysis["layers"][l]["G_effective_rank"] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width / 2, eff_ranks_A, width, label="A (Activation)", alpha=0.8)
    bars2 = ax.bar(x + width / 2, eff_ranks_G, width, label="G (Gradient)", alpha=0.8)

    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank by Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effective_ranks.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# K-FAC Treatment
# ============================================================================


def apply_kfac_treatment(
    model: nn.Module,
    factors: Dict[str, Dict[str, torch.Tensor]],
    config: KFACConfig,
) -> Dict[str, Any]:
    """
    Apply K-FAC treatment to reduce memorization.

    Args:
        model: The language model.
        factors: K-FAC factors dictionary.
        config: KFACConfig with settings.

    Returns:
        Treatment statistics.
    """
    print(f"\n{'='*60}")
    print("K-FAC Treatment Application")
    print(f"{'='*60}")
    print(f"Variance ratio: {config.variance_ratio}")
    print(f"Method: {config.treatment_method}")

    # Get model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find model layers")

    treatment_stats = {}

    for layer_name, layer_factors in factors.items():
        # Parse layer name to get index and projection type
        # Format: blk{idx}.{proj_type}
        parts = layer_name.split(".")
        if len(parts) < 2:
            continue

        try:
            idx = int(parts[0].replace("blk", ""))
            proj_type = parts[1]
        except (ValueError, IndexError):
            print(f"Warning: Could not parse layer name {layer_name}")
            continue

        if idx >= len(layers):
            continue

        layer = layers[idx]
        mlp = layer.mlp if hasattr(layer, "mlp") else layer

        # Find the projection layer
        if proj_type == "gate" and hasattr(mlp, "gate_proj"):
            proj_layer = mlp.gate_proj
        elif proj_type == "up" and hasattr(mlp, "up_proj"):
            proj_layer = mlp.up_proj
        elif proj_type == "down" and hasattr(mlp, "down_proj"):
            proj_layer = mlp.down_proj
        elif proj_type == "gate_up" and hasattr(mlp, "gate_up_proj"):
            proj_layer = mlp.gate_up_proj
        else:
            print(f"Warning: Could not find projection {proj_type} for layer {idx}")
            continue

        # Apply treatment
        A = layer_factors["A"].to(config.device).float()
        G = layer_factors["G"].to(config.device).float()

        # Eigendecomposition
        eva_A, evc_A = torch.linalg.eigh(A)
        eva_G, evc_G = torch.linalg.eigh(G)

        # Sort descending
        idx_A = eva_A.argsort(descending=True)
        eva_A, evc_A = eva_A[idx_A], evc_A[:, idx_A]

        idx_G = eva_G.argsort(descending=True)
        eva_G, evc_G = eva_G[idx_G], evc_G[:, idx_G]

        # Compute ranks for target variance
        def get_rank_for_variance(evals, target):
            cumsum = torch.cumsum(evals, dim=0) / evals.sum()
            if target >= 1.0:
                return len(evals)
            mask = cumsum >= target
            if mask.any():
                return mask.nonzero()[0].item() + 1
            return len(evals)

        rA = get_rank_for_variance(eva_A, config.variance_ratio)
        rG = get_rank_for_variance(eva_G, config.variance_ratio)

        # Project weight
        W = proj_layer.weight.data.float()
        Ug = evc_G[:, :rG]
        Ua = evc_A[:, :rA]

        W_proj = Ug @ (Ug.T @ W @ Ua) @ Ua.T

        # Update weight
        with torch.no_grad():
            proj_layer.weight.copy_(W_proj.to(proj_layer.weight.dtype))

        # Record stats
        treatment_stats[layer_name] = {
            "original_params": W.numel(),
            "effective_params": rA * rG,
            "compression_ratio": (rA * rG) / W.numel(),
            "rank_A": rA,
            "rank_G": rG,
            "total_A": len(eva_A),
            "total_G": len(eva_G),
        }

        print(f"  {layer_name}: rA={rA}/{len(eva_A)}, rG={rG}/{len(eva_G)}, compression={treatment_stats[layer_name]['compression_ratio']:.2%}")

    return treatment_stats


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    config: KFACConfig,
    description: str = "Evaluation",
) -> Dict[str, float]:
    """
    Evaluate model perplexity.

    Args:
        model: The language model.
        dataloader: DataLoader for evaluation.
        config: KFACConfig.
        description: Description for progress bar.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(config.device)

            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1].float()

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, :-1].reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )

            total_loss += loss.item()
            if attention_mask is not None:
                total_tokens += attention_mask[:, 1:].sum().item()
            else:
                total_tokens += (labels[:, :-1] != -100).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }


# ============================================================================
# Main Pipeline Functions
# ============================================================================


def run_collection(config: KFACConfig):
    """Run K-FAC factor collection."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=getattr(torch, config.dtype),
        device_map=config.device,
        trust_remote_code=config.trust_remote_code,
    )

    print("Creating dataloader...")
    dataloader = create_dataloader(config, tokenizer)

    # Collect factors
    factors = collect_kfac_factors(model, dataloader, config.target_layers, config)

    # Save factors
    if config.save_factors:
        os.makedirs(config.output_dir, exist_ok=True)
        factors_path = os.path.join(config.output_dir, "kfac_factors.pt")
        torch.save(factors, factors_path)
        print(f"\nFactors saved to {factors_path}")

    return factors


def run_analysis(factors_path: str, output_dir: str):
    """Run K-FAC factor analysis."""
    print(f"Loading factors from {factors_path}...")
    factors = torch.load(factors_path, map_location="cpu")

    analysis = analyze_kfac_factors(factors, output_dir)
    return analysis


def run_treatment(config: KFACConfig, factors_path: str):
    """Run K-FAC treatment and evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=getattr(torch, config.dtype),
        device_map=config.device,
        trust_remote_code=config.trust_remote_code,
    )

    print("Loading factors...")
    factors = torch.load(factors_path, map_location="cpu")

    print("Creating evaluation dataloader...")
    dataloader = create_dataloader(config, tokenizer)

    # Evaluate before treatment
    print("\nEvaluating model BEFORE treatment...")
    metrics_before = evaluate_model(model, dataloader, config, "Before Treatment")
    print(f"  Perplexity: {metrics_before['perplexity']:.2f}")

    # Apply treatment
    treatment_stats = apply_kfac_treatment(model, factors, config)

    # Evaluate after treatment
    print("\nEvaluating model AFTER treatment...")
    dataloader = create_dataloader(config, tokenizer)  # Reset dataloader
    metrics_after = evaluate_model(model, dataloader, config, "After Treatment")
    print(f"  Perplexity: {metrics_after['perplexity']:.2f}")

    # Summary
    print(f"\n{'='*60}")
    print("Treatment Summary")
    print(f"{'='*60}")
    print(f"Variance ratio: {config.variance_ratio}")
    print(f"Perplexity before: {metrics_before['perplexity']:.2f}")
    print(f"Perplexity after: {metrics_after['perplexity']:.2f}")
    print(f"Perplexity change: {metrics_after['perplexity'] - metrics_before['perplexity']:.2f}")

    return {
        "treatment_stats": treatment_stats,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }


def run_full_pipeline(config: KFACConfig):
    """Run the complete K-FAC pipeline."""
    print(f"\n{'='*60}")
    print("FULL K-FAC PIPELINE")
    print(f"{'='*60}")

    # Step 1: Collect factors
    print("\n[Step 1/4] Collecting K-FAC factors...")
    factors = run_collection(config)

    # Step 2: Analyze factors
    print("\n[Step 2/4] Analyzing K-FAC factors...")
    analysis_dir = os.path.join(config.output_dir, "analysis")
    analysis = analyze_kfac_factors(factors, analysis_dir)

    # Step 3: Apply treatment and evaluate
    print("\n[Step 3/4] Applying K-FAC treatment...")
    factors_path = os.path.join(config.output_dir, "kfac_factors.pt")
    results = run_treatment(config, factors_path)

    # Step 4: Save results
    print("\n[Step 4/4] Saving results...")
    results_path = os.path.join(config.output_dir, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": vars(config),
                "treatment_stats": results["treatment_stats"],
                "metrics_before": results["metrics_before"],
                "metrics_after": results["metrics_after"],
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {config.output_dir}")

    return results


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-FAC Pipeline for Memorization Reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect K-FAC factors")
    collect_parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    collect_parser.add_argument("--layers", type=int, nargs="+", default=[20, 24, 28, 31])
    collect_parser.add_argument("--output_dir", type=str, default="./kfac_output")
    collect_parser.add_argument("--batch_size", type=int, default=4)
    collect_parser.add_argument("--seq_length", type=int, default=512)
    collect_parser.add_argument("--max_samples", type=int, default=1000)
    collect_parser.add_argument("--sample_labels", action="store_true")
    collect_parser.add_argument("--device", type=str, default="cuda")
    collect_parser.add_argument("--dtype", type=str, default="bfloat16")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze K-FAC factors")
    analyze_parser.add_argument("--factors_path", type=str, required=True)
    analyze_parser.add_argument("--output_dir", type=str, default="./kfac_analysis")

    # Treat command
    treat_parser = subparsers.add_parser("treat", help="Apply K-FAC treatment")
    treat_parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    treat_parser.add_argument("--factors_path", type=str, required=True)
    treat_parser.add_argument("--variance_ratio", type=float, default=0.9)
    treat_parser.add_argument("--method", type=str, choices=["product", "separate"], default="product")
    treat_parser.add_argument("--batch_size", type=int, default=4)
    treat_parser.add_argument("--max_samples", type=int, default=500)
    treat_parser.add_argument("--device", type=str, default="cuda")
    treat_parser.add_argument("--dtype", type=str, default="bfloat16")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    full_parser.add_argument("--layers", type=int, nargs="+", default=[20, 24, 28, 31])
    full_parser.add_argument("--output_dir", type=str, default="./kfac_pipeline")
    full_parser.add_argument("--batch_size", type=int, default=4)
    full_parser.add_argument("--seq_length", type=int, default=512)
    full_parser.add_argument("--max_samples", type=int, default=1000)
    full_parser.add_argument("--variance_ratio", type=float, default=0.9)
    full_parser.add_argument("--sample_labels", action="store_true")
    full_parser.add_argument("--device", type=str, default="cuda")
    full_parser.add_argument("--dtype", type=str, default="bfloat16")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "collect":
        config = KFACConfig(
            model_name_or_path=args.model,
            target_layers=args.layers,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            max_samples=args.max_samples,
            sample_labels=args.sample_labels,
            device=args.device,
            dtype=args.dtype,
        )
        run_collection(config)

    elif args.command == "analyze":
        run_analysis(args.factors_path, args.output_dir)

    elif args.command == "treat":
        config = KFACConfig(
            model_name_or_path=args.model,
            variance_ratio=args.variance_ratio,
            treatment_method=args.method,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device,
            dtype=args.dtype,
        )
        run_treatment(config, args.factors_path)

    elif args.command == "full":
        config = KFACConfig(
            model_name_or_path=args.model,
            target_layers=args.layers,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            max_samples=args.max_samples,
            variance_ratio=args.variance_ratio,
            sample_labels=args.sample_labels,
            device=args.device,
            dtype=args.dtype,
        )
        run_full_pipeline(config)

    else:
        print("Please specify a command: collect, analyze, treat, or full")
        print("Use --help for more information")
        sys.exit(1)


if __name__ == "__main__":
    main()
