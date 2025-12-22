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
K-FAC (Kronecker-Factored Approximate Curvature) implementation for verl's
parallel llama models.

This module adapts the memorization_kfac library to work with:
- MergedColumnParallelLinear (combined gate_up_proj)
- RowParallelLinear (down_proj)
- Tensor parallelism and sequence parallelism

The key insight is that for tensor-parallel layers, K-FAC factors need to be
aggregated across the tensor parallel group to get the full covariance matrices.
"""

import heapq
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

try:
    from megatron.core import parallel_state as mpu
except ImportError:
    mpu = None


class ParallelKFACCollector:
    """
    Collect K-FAC factors (A and G matrices) from parallel linear layers.

    For tensor-parallel layers:
    - Column parallel (gate_up_proj): input dimension is replicated,
      output dimension is sharded. A is local, G needs all-reduce.
    - Row parallel (down_proj): input dimension is sharded,
      output dimension is replicated. A needs all-reduce, G is local.

    This collector handles the distributed aggregation automatically.
    """

    def __init__(
        self,
        layer: nn.Module,
        layer_name: str,
        layer_type: str = "column",  # "column" or "row"
        device: Optional[torch.device] = None,
    ):
        """
        Initialize K-FAC collector for a parallel linear layer.

        Args:
            layer: The parallel linear layer to collect factors from.
            layer_name: Name identifier for this layer.
            layer_type: "column" for ColumnParallelLinear, "row" for RowParallelLinear.
            device: Device for factor tensors.
        """
        self.layer = layer
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.device = device or next(layer.parameters()).device

        # Get dimensions from the weight tensor
        # For column parallel: weight shape is [out_features/tp, in_features]
        # For row parallel: weight shape is [out_features, in_features/tp]
        weight = layer.weight
        out_dim, in_dim = weight.shape

        # Adjust dimensions for tensor parallelism
        tp_size = self._get_tp_size()

        if layer_type == "column":
            # Column parallel: output is sharded, input is full
            self.full_out_dim = out_dim * tp_size
            self.full_in_dim = in_dim
            self.local_out_dim = out_dim
            self.local_in_dim = in_dim
        else:  # row
            # Row parallel: input is sharded, output is full
            self.full_out_dim = out_dim
            self.full_in_dim = in_dim * tp_size
            self.local_out_dim = out_dim
            self.local_in_dim = in_dim

        # Initialize factor accumulators
        # A: input covariance (in_dim x in_dim) - local for column, distributed for row
        # G: gradient covariance (out_dim x out_dim) - distributed for column, local for row
        self.A = torch.zeros(self.full_in_dim, self.full_in_dim, dtype=torch.float32, device=self.device)
        self.G = torch.zeros(self.full_out_dim, self.full_out_dim, dtype=torch.float32, device=self.device)
        self.n_tokens = 0

        self._input_buffer = None
        self._hooks = []
        self._register_hooks()

    def _get_tp_size(self) -> int:
        """Get tensor parallel world size."""
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_world_size()
            except Exception:
                pass
        return 1

    def _get_tp_rank(self) -> int:
        """Get tensor parallel rank."""
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_rank()
            except Exception:
                pass
        return 0

    def _get_tp_group(self):
        """Get tensor parallel process group."""
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_group()
            except Exception:
                pass
        return None

    def _register_hooks(self):
        """Register forward and backward hooks for factor collection."""

        def forward_pre_hook(module, inputs):
            """Capture activations before the layer."""
            if torch.is_grad_enabled():
                x = inputs[0]
                # Handle sequence dimension: skip last token for autoregressive
                if x.dim() == 3:
                    x = x[:, :-1, :]  # [batch, seq-1, hidden]
                    x = x.reshape(-1, x.size(-1))  # [batch*(seq-1), hidden]
                elif x.dim() == 2:
                    x = x[:-1, :]  # [seq-1, hidden]

                self._input_buffer = x.detach().float()

        def backward_hook(module, grad_input, grad_output):
            """Capture gradients after the backward pass."""
            if grad_output[0] is None or self._input_buffer is None:
                return

            g = grad_output[0]
            # Handle sequence dimension
            if g.dim() == 3:
                g = g[:, :-1, :]
                g = g.reshape(-1, g.size(-1))
            elif g.dim() == 2:
                g = g[:-1, :]

            g = g.detach().float()
            x = self._input_buffer

            n_samples = g.size(0)
            self.n_tokens += n_samples

            tp_size = self._get_tp_size()
            tp_rank = self._get_tp_rank()
            tp_group = self._get_tp_group()

            if self.layer_type == "column":
                # Column parallel: A is local (full input), G is sharded (need to handle)
                # Update A locally
                self.A.add_(x.T @ x)

                # For G, we have local gradients for local output dimensions
                # We need to place them in the correct shard location
                local_g = g.T @ g  # [local_out, local_out]

                if tp_size > 1:
                    # Place local G in the correct position of the full G matrix
                    start = tp_rank * self.local_out_dim
                    end = start + self.local_out_dim
                    self.G[start:end, start:end].add_(local_g)
                else:
                    self.G.add_(local_g)

            else:  # row parallel
                # Row parallel: A is sharded (need to handle), G is local (full output)
                # For A, we have local inputs for local input dimensions
                local_a = x.T @ x  # [local_in, local_in]

                if tp_size > 1:
                    # Place local A in the correct position of the full A matrix
                    start = tp_rank * self.local_in_dim
                    end = start + self.local_in_dim
                    self.A[start:end, start:end].add_(local_a)
                else:
                    self.A.add_(local_a)

                # Update G locally (full output dimension)
                self.G.add_(g.T @ g)

            self._input_buffer = None

        self._hooks.append(self.layer.register_forward_pre_hook(forward_pre_hook))
        self._hooks.append(self.layer.register_full_backward_hook(backward_hook))

    def synchronize_factors(self):
        """
        Synchronize K-FAC factors across tensor parallel ranks.

        Call this after all data has been processed to aggregate factors.
        """
        tp_size = self._get_tp_size()
        tp_group = self._get_tp_group()

        if tp_size > 1 and tp_group is not None:
            # All-reduce the factors that are distributed
            if self.layer_type == "column":
                # G is distributed, all-reduce it
                dist.all_reduce(self.G, group=tp_group)
            else:  # row
                # A is distributed, all-reduce it
                dist.all_reduce(self.A, group=tp_group)

            # Also sum up the token counts
            n_tokens_tensor = torch.tensor([self.n_tokens], dtype=torch.long, device=self.device)
            dist.all_reduce(n_tokens_tensor, group=tp_group)
            self.n_tokens = n_tokens_tensor.item() // tp_size  # Divide by tp_size since each rank sees all tokens

    def get_factors(self) -> Dict[str, torch.Tensor]:
        """
        Get normalized K-FAC factors.

        Returns:
            Dict with keys 'A', 'G', and 'n_tokens'.
        """
        if self.n_tokens == 0:
            raise RuntimeError("No tokens collected. Run forward/backward passes first.")

        return {
            'A': (self.A / self.n_tokens).cpu(),
            'G': (self.G / self.n_tokens).cpu(),
            'n_tokens': self.n_tokens,
        }

    def close(self):
        """Remove hooks and free buffers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._input_buffer = None


class MergedKFACCollector:
    """
    K-FAC collector for MergedColumnParallelLinear (gate_up_proj).

    This layer combines gate_proj and up_proj into a single weight matrix.
    We collect separate factors for the gate and up projections by splitting
    the gradient covariance matrix appropriately.
    """

    def __init__(
        self,
        layer: nn.Module,
        layer_name: str,
        intermediate_size: int,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize collector for merged gate_up projection.

        Args:
            layer: MergedColumnParallelLinear layer.
            layer_name: Base name (will create {name}.gate and {name}.up).
            intermediate_size: The intermediate size (used to split gate/up).
            device: Device for factor tensors.
        """
        self.layer = layer
        self.layer_name = layer_name
        self.intermediate_size = intermediate_size
        self.device = device or next(layer.parameters()).device

        # Get dimensions
        weight = layer.weight
        out_dim, in_dim = weight.shape  # [2*intermediate/tp, hidden]

        tp_size = self._get_tp_size()
        self.gate_dim = intermediate_size // tp_size
        self.up_dim = intermediate_size // tp_size

        # A is shared (same input activations)
        self.A = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=self.device)

        # Separate G for gate and up
        self.G_gate = torch.zeros(intermediate_size, intermediate_size, dtype=torch.float32, device=self.device)
        self.G_up = torch.zeros(intermediate_size, intermediate_size, dtype=torch.float32, device=self.device)

        self.n_tokens = 0
        self._input_buffer = None
        self._hooks = []
        self._register_hooks()

    def _get_tp_size(self) -> int:
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_world_size()
            except Exception:
                pass
        return 1

    def _get_tp_rank(self) -> int:
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_rank()
            except Exception:
                pass
        return 0

    def _get_tp_group(self):
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_group()
            except Exception:
                pass
        return None

    def _register_hooks(self):
        def forward_pre_hook(module, inputs):
            if torch.is_grad_enabled():
                x = inputs[0]
                if x.dim() == 3:
                    x = x[:, :-1, :]
                    x = x.reshape(-1, x.size(-1))
                elif x.dim() == 2:
                    x = x[:-1, :]
                self._input_buffer = x.detach().float()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is None or self._input_buffer is None:
                return

            g = grad_output[0]
            if g.dim() == 3:
                g = g[:, :-1, :]
                g = g.reshape(-1, g.size(-1))
            elif g.dim() == 2:
                g = g[:-1, :]

            g = g.detach().float()
            x = self._input_buffer

            n_samples = g.size(0)
            self.n_tokens += n_samples

            # Update A (shared input)
            self.A.add_(x.T @ x)

            # Split gradients into gate and up
            g_gate = g[:, :self.gate_dim]  # [samples, gate_dim]
            g_up = g[:, self.gate_dim:]     # [samples, up_dim]

            tp_rank = self._get_tp_rank()
            start = tp_rank * self.gate_dim
            end = start + self.gate_dim

            # Update G matrices (placed in correct shard location)
            self.G_gate[start:end, start:end].add_(g_gate.T @ g_gate)
            self.G_up[start:end, start:end].add_(g_up.T @ g_up)

            self._input_buffer = None

        self._hooks.append(self.layer.register_forward_pre_hook(forward_pre_hook))
        self._hooks.append(self.layer.register_full_backward_hook(backward_hook))

    def synchronize_factors(self):
        tp_size = self._get_tp_size()
        tp_group = self._get_tp_group()

        if tp_size > 1 and tp_group is not None:
            # G matrices are distributed, all-reduce them
            dist.all_reduce(self.G_gate, group=tp_group)
            dist.all_reduce(self.G_up, group=tp_group)

            n_tokens_tensor = torch.tensor([self.n_tokens], dtype=torch.long, device=self.device)
            dist.all_reduce(n_tokens_tensor, group=tp_group)
            self.n_tokens = n_tokens_tensor.item() // tp_size

    def get_factors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.n_tokens == 0:
            raise RuntimeError("No tokens collected.")

        return {
            f'{self.layer_name}.gate': {
                'A': (self.A / self.n_tokens).cpu(),
                'G': (self.G_gate / self.n_tokens).cpu(),
                'n_tokens': self.n_tokens,
            },
            f'{self.layer_name}.up': {
                'A': (self.A / self.n_tokens).cpu(),
                'G': (self.G_up / self.n_tokens).cpu(),
                'n_tokens': self.n_tokens,
            },
        }

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._input_buffer = None


class ParallelKFACTreatment:
    """
    Apply K-FAC compression to parallel llama MLP layers.

    This adapts the KFACTreatment class from memorization_kfac to work with
    verl's parallel linear layers. It handles:
    - MergedColumnParallelLinear (gate_up_proj)
    - RowParallelLinear (down_proj)
    - Tensor parallel weight distribution

    The treatment projects weights onto top eigenspaces to reduce memorization
    while preserving learned generalizations.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        kfac_factors_path: str,
        projections: List[str] = ["gate", "up", "down"],
        device: Optional[str] = None,
        keep_eigenvectors_on_cpu: bool = False,
    ):
        """
        Initialize K-FAC treatment for parallel llama model.

        Args:
            model: ParallelLlamaForCausalLMRmPad or similar model.
            layer_indices: List of layer indices to apply treatment to.
            kfac_factors_path: Path to K-FAC factors file.
            projections: Which projections to treat ("gate", "up", "down").
            device: Device for computations.
            keep_eigenvectors_on_cpu: Store eigenvectors on CPU to save GPU memory.
        """
        self.model = model
        self.layer_indices = layer_indices
        self.projections = projections
        self.device = device or str(next(model.parameters()).device)
        self.keep_eigenvectors_on_cpu = keep_eigenvectors_on_cpu

        # Load K-FAC factors
        self.kfac_data = torch.load(kfac_factors_path, map_location='cpu')

        # Build layer mapping
        self.layer_map = self._build_layer_map()

        # Store original weights
        self.original_weights = {}
        self._store_original_weights()

        # Prepare eigendecompositions
        self.kfac_info = {}
        self._prepare_kfac_info()

        # Track compression stats
        self.compression_stats = {}

    def _get_tp_size(self) -> int:
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_world_size()
            except Exception:
                pass
        return 1

    def _get_tp_rank(self) -> int:
        if mpu is not None and mpu.is_initialized():
            try:
                return mpu.get_tensor_model_parallel_rank()
            except Exception:
                pass
        return 0

    def _build_layer_map(self) -> Dict[str, Tuple[nn.Module, str]]:
        """Build mapping from layer names to (layer, projection_type)."""
        layer_map = {}

        # Get the model's decoder layers
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Could not find decoder layers in model")

        for idx in self.layer_indices:
            if idx >= len(layers):
                raise ValueError(f"Layer index {idx} out of range")

            mlp = layers[idx].mlp

            if "gate" in self.projections or "up" in self.projections:
                # MergedColumnParallelLinear contains both gate and up
                if hasattr(mlp, 'gate_up_proj'):
                    layer_map[f'blk{idx}.gate'] = (mlp.gate_up_proj, 'gate')
                    layer_map[f'blk{idx}.up'] = (mlp.gate_up_proj, 'up')

            if "down" in self.projections:
                if hasattr(mlp, 'down_proj'):
                    layer_map[f'blk{idx}.down'] = (mlp.down_proj, 'down')

        return layer_map

    def _store_original_weights(self):
        """Store original weights for restoration."""
        with torch.no_grad():
            for key, (layer, proj_type) in self.layer_map.items():
                weight_key = f"{key}_weight"
                if weight_key not in self.original_weights:
                    self.original_weights[weight_key] = layer.weight.data.detach().clone()

    def _get_kfac_key(self, layer_name: str) -> str:
        """Get the key in kfac_data for a layer."""
        # Try exact match first
        if layer_name in self.kfac_data:
            return layer_name

        # Try variations
        for key in self.kfac_data.keys():
            if layer_name.replace('.', '_') in key or key in layer_name:
                return key

        raise ValueError(f"K-FAC factors not found for {layer_name}")

    def _prepare_kfac_info(self):
        """Prepare eigendecompositions for all layers."""
        tp_size = self._get_tp_size()
        tp_rank = self._get_tp_rank()

        for layer_name, (layer, proj_type) in self.layer_map.items():
            kfac_key = self._get_kfac_key(layer_name)
            kfac_layer_data = self.kfac_data[kfac_key]

            # Get the weight for this specific projection
            weight = layer.weight.data
            out_dim, in_dim = weight.shape

            # For merged gate_up, we need to slice the weight appropriately
            if proj_type == 'gate':
                weight = weight[:out_dim // 2, :]
            elif proj_type == 'up':
                weight = weight[out_dim // 2:, :]

            device = self.device if not self.keep_eigenvectors_on_cpu else 'cpu'

            A = kfac_layer_data['A'].float().to(device)
            G = kfac_layer_data['G'].float().to(device)

            # Compute eigendecomposition
            eva_A, evc_A = torch.linalg.eigh(A)
            eva_G, evc_G = torch.linalg.eigh(G)

            # Sort descending
            idx_A = eva_A.argsort(descending=True)
            eva_A = eva_A[idx_A]
            evc_A = evc_A[:, idx_A]

            idx_G = eva_G.argsort(descending=True)
            eva_G = eva_G[idx_G]
            evc_G = evc_G[:, idx_G]

            if self.keep_eigenvectors_on_cpu:
                eva_A = eva_A.cpu()
                evc_A = evc_A.cpu()
                eva_G = eva_G.cpu()
                evc_G = evc_G.cpu()

            # Store with projection info
            self.kfac_info[layer_name] = {
                'proj_type': proj_type,
                'W_orig': weight.detach().clone(),
                'eva_A': eva_A,
                'evc_A': evc_A,
                'eva_G': eva_G,
                'evc_G': evc_G,
                'tp_size': tp_size,
                'tp_rank': tp_rank,
            }

            print(f"{layer_name}: W{tuple(weight.shape)}, G{tuple(G.shape)}, A{tuple(A.shape)}")

    def _compute_rank_for_variance(self, evals: torch.Tensor, target_variance: float) -> Tuple[int, float]:
        """Compute number of components to retain target variance."""
        total = evals.sum()
        cumsum = torch.cumsum(evals, dim=0)
        ratios = cumsum / total

        if target_variance >= 1.0:
            return len(evals), 1.0

        mask = ratios >= target_variance
        if mask.any():
            r = mask.nonzero()[0].item() + 1
            actual = ratios[r-1].item()
        else:
            r = len(evals)
            actual = 1.0

        return r, actual

    def _project_weight(self, info: Dict, rG: int, rA: int) -> torch.Tensor:
        """Project weight matrix to reduced eigenspace."""
        device = info['W_orig'].device

        Ug = info['evc_G'][:, :rG]
        Ua = info['evc_A'][:, :rA]

        if Ug.device != device:
            Ug = Ug.to(device)
        if Ua.device != device:
            Ua = Ua.to(device)

        W = info['W_orig'].float()

        # For parallel layers, we need to handle the sharded dimensions
        tp_size = info['tp_size']
        tp_rank = info['tp_rank']
        proj_type = info['proj_type']

        if tp_size > 1:
            if proj_type in ['gate', 'up']:
                # Column parallel: output is sharded
                # Slice Ug for this rank's portion
                out_per_rank = Ug.shape[0] // tp_size
                Ug_local = Ug[tp_rank * out_per_rank:(tp_rank + 1) * out_per_rank, :]
                W_proj = Ug_local @ (Ug.T @ W @ Ua) @ Ua.T
            else:  # down
                # Row parallel: input is sharded
                in_per_rank = Ua.shape[0] // tp_size
                Ua_local = Ua[tp_rank * in_per_rank:(tp_rank + 1) * in_per_rank, :]
                W_proj = Ug @ (Ug.T @ W @ Ua_local) @ Ua.T
        else:
            W_proj = Ug @ (Ug.T @ W @ Ua) @ Ua.T

        return W_proj

    def apply_kfac(self, variance_ratios: Union[Tuple[float, float], Dict[str, Tuple[float, float]]]):
        """
        Apply K-FAC compression to specified layers.

        Args:
            variance_ratios: Either (activation_variance, gradient_variance) for all layers,
                           or dict mapping layer names to variance ratios.
        """
        if isinstance(variance_ratios, tuple):
            ratios_dict = {name: variance_ratios for name in self.kfac_info.keys()}
        else:
            ratios_dict = variance_ratios

        with torch.no_grad():
            # Group by actual layer (gate and up share the same weight)
            processed_layers = set()

            for layer_name in self.kfac_info.keys():
                if layer_name not in ratios_dict:
                    continue

                layer, proj_type = self.layer_map[layer_name]

                var_A, var_G = ratios_dict[layer_name]
                info = self.kfac_info[layer_name]

                eva_A = info['eva_A'].to(self.device)
                eva_G = info['eva_G'].to(self.device)

                rA, actual_var_A = self._compute_rank_for_variance(eva_A, var_A)
                rG, actual_var_G = self._compute_rank_for_variance(eva_G, var_G)

                W_compressed = self._project_weight(info, rG, rA)

                # Update the appropriate slice of the layer weight
                weight = layer.weight
                out_dim = weight.shape[0]

                if proj_type == 'gate':
                    weight.data[:out_dim // 2, :] = W_compressed.to(weight.dtype)
                elif proj_type == 'up':
                    weight.data[out_dim // 2:, :] = W_compressed.to(weight.dtype)
                else:  # down
                    weight.data.copy_(W_compressed.to(weight.dtype))

                # Record stats
                self.compression_stats[layer_name] = {
                    'variance_ratios': (var_A, var_G),
                    'actual_variances': (actual_var_A, actual_var_G),
                    'ranks': (rA, rG),
                    'compression_ratio': (rA * rG) / info['W_orig'].numel(),
                }

                print(f"Applied K-FAC to {layer_name}:")
                print(f"  rA={rA} ({actual_var_A:.1%}), rG={rG} ({actual_var_G:.1%})")

    def apply_kfac_by_product(self, variance_ratio: Union[float, Dict[str, float]]):
        """
        Apply K-FAC compression using product-based variance retaining.

        Selects largest λ_i × μ_j eigenvalue products until target mass is retained.

        Args:
            variance_ratio: Single float for all layers or dict per layer.
        """
        if isinstance(variance_ratio, float):
            ratio_map = {name: variance_ratio for name in self.kfac_info.keys()}
        else:
            ratio_map = variance_ratio

        with torch.no_grad():
            for layer_name in self.kfac_info.keys():
                if layer_name not in ratio_map:
                    continue

                rho = ratio_map[layer_name]
                info = self.kfac_info[layer_name]
                layer, proj_type = self.layer_map[layer_name]

                eva_G = info['eva_G'].to(self.device)
                eva_A = info['eva_A'].to(self.device)

                # Get top pairs by product mass
                pairs = self._top_pairs_by_product(eva_G, eva_A, rho)

                # Project weight using selected pairs
                W_proj = self._project_weight_pairs(info, pairs)

                # Update weight
                weight = layer.weight
                out_dim = weight.shape[0]

                if proj_type == 'gate':
                    weight.data[:out_dim // 2, :] = W_proj.to(weight.dtype)
                elif proj_type == 'up':
                    weight.data[out_dim // 2:, :] = W_proj.to(weight.dtype)
                else:
                    weight.data.copy_(W_proj.to(weight.dtype))

                k = len(pairs)
                print(f"{layer_name}: {k} pairs ({k / info['W_orig'].numel():.2%})")

                self.compression_stats[layer_name] = {
                    'num_pairs': k,
                    'pair_ratio': k / info['W_orig'].numel(),
                    'rho': rho,
                }

    @staticmethod
    def _top_pairs_by_product(evals_G: torch.Tensor, evals_A: torch.Tensor, ratio: float) -> List[Tuple[int, int]]:
        """Select largest eigenvalue product pairs until target mass is reached."""
        m, n = len(evals_G), len(evals_A)
        total_mass = (evals_G.sum() * evals_A.sum()).item()
        target_mass = ratio * total_mass

        heap = [(-(evals_G[0] * evals_A[0]).item(), 0, 0)]
        seen = {(0, 0)}
        cum_mass = 0.0
        selected = []

        while cum_mass < target_mass and heap:
            neg_prod, i, j = heapq.heappop(heap)
            prod = -neg_prod
            cum_mass += prod
            selected.append((i, j))

            if i + 1 < m and (i + 1, j) not in seen:
                heapq.heappush(heap, (-(evals_G[i + 1] * evals_A[j]).item(), i + 1, j))
                seen.add((i + 1, j))
            if j + 1 < n and (i, j + 1) not in seen:
                heapq.heappush(heap, (-(evals_G[i] * evals_A[j + 1]).item(), i, j + 1))
                seen.add((i, j + 1))

        return selected

    def _project_weight_pairs(self, info: Dict, pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """Project weight using specific (i,j) eigenvector pairs."""
        device = info['W_orig'].device

        Ug = info['evc_G'].to(device)
        Ua = info['evc_A'].to(device)
        W = info['W_orig'].float()

        C = Ug.T @ W @ Ua

        mask = torch.zeros_like(C)
        if pairs:
            idx_i, idx_j = zip(*pairs)
            mask[idx_i, idx_j] = 1.0

        C_masked = C * mask
        W_proj = Ug @ C_masked @ Ua.T

        return W_proj

    def restore_original_weights(self):
        """Restore all layers to original weights."""
        with torch.no_grad():
            for layer_name, (layer, proj_type) in self.layer_map.items():
                weight_key = f"{layer_name}_weight"
                if weight_key in self.original_weights:
                    orig = self.original_weights[weight_key]

                    if proj_type == 'gate':
                        out_dim = layer.weight.shape[0]
                        layer.weight.data[:out_dim // 2, :] = orig[:out_dim // 2, :]
                    elif proj_type == 'up':
                        out_dim = layer.weight.shape[0]
                        layer.weight.data[out_dim // 2:, :] = orig[out_dim // 2:, :]
                    else:
                        layer.weight.data.copy_(orig)

        self.compression_stats = {}
        print("Restored original weights for all layers")

    def get_compression_summary(self) -> Dict:
        """Get summary of compression applied to each layer."""
        return self.compression_stats
