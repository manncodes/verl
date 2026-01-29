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
Gradient Analysis Module for Multi-Domain Training

This module provides utilities for computing gradient interference matrices
and domain-wise gradient metrics during multi-domain RL training.

Key concepts:
- Interference Matrix: Measures how gradients from one domain affect other domains
- Domain Gradient Metrics: Per-domain gradient magnitude, direction, and variance
"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class DomainGradientInfo:
    """Stores gradient information for a single domain."""

    domain_name: str
    gradient_magnitude: float = 0.0
    gradient_direction: Optional[torch.Tensor] = None  # Flattened normalized gradient
    sample_count: int = 0
    loss_sum: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to metrics dictionary."""
        return {
            f"gradient/{self.domain_name}/magnitude": self.gradient_magnitude,
            f"gradient/{self.domain_name}/sample_count": float(self.sample_count),
            f"gradient/{self.domain_name}/avg_loss": self.loss_sum / max(self.sample_count, 1),
        }


@dataclass
class GradientAnalysisResult:
    """Results from gradient analysis across domains."""

    domain_info: dict[str, DomainGradientInfo] = field(default_factory=dict)
    interference_matrix: Optional[torch.Tensor] = None  # Shape: (num_domains, num_domains)
    domain_names: list[str] = field(default_factory=list)
    overall_gradient_norm: float = 0.0

    def to_metrics_dict(self) -> dict[str, Any]:
        """Convert analysis results to a metrics dictionary for logging."""
        metrics = {}

        # Add overall gradient norm
        metrics["gradient/overall_norm"] = self.overall_gradient_norm

        # Add per-domain metrics
        for domain_name, info in self.domain_info.items():
            metrics.update(info.to_dict())

        # Add interference matrix metrics
        if self.interference_matrix is not None and len(self.domain_names) > 1:
            interference = self.interference_matrix.cpu().numpy()

            # Log individual interference values
            for i, domain_i in enumerate(self.domain_names):
                for j, domain_j in enumerate(self.domain_names):
                    metrics[f"interference/{domain_i}_to_{domain_j}"] = float(interference[i, j])

            # Compute summary statistics
            # Off-diagonal mean (cross-domain interference)
            n = len(self.domain_names)
            if n > 1:
                off_diag_mask = ~np.eye(n, dtype=bool)
                off_diag_values = interference[off_diag_mask]
                metrics["interference/cross_domain_mean"] = float(np.mean(off_diag_values))
                metrics["interference/cross_domain_std"] = float(np.std(off_diag_values))
                metrics["interference/cross_domain_min"] = float(np.min(off_diag_values))
                metrics["interference/cross_domain_max"] = float(np.max(off_diag_values))

                # Diagonal mean (within-domain alignment)
                diag_values = np.diag(interference)
                metrics["interference/within_domain_mean"] = float(np.mean(diag_values))

                # Conflict ratio: fraction of negative cross-domain similarities
                conflict_ratio = np.mean(off_diag_values < 0)
                metrics["interference/conflict_ratio"] = float(conflict_ratio)

        return metrics


class GradientAnalyzer:
    """
    Analyzes gradients across multiple domains for interference detection.

    This class provides methods to:
    1. Accumulate gradients per domain
    2. Compute interference matrices (cosine similarity between domain gradients)
    3. Track domain-wise gradient statistics

    Usage:
        analyzer = GradientAnalyzer(model)

        # During training loop
        for micro_batch in micro_batches:
            domains = micro_batch["data_source"]  # e.g., ["math", "code", "math", ...]
            loss.backward()
            analyzer.accumulate_domain_gradients(domains)

        # After mini-batch
        result = analyzer.compute_interference_matrix()
        metrics = result.to_metrics_dict()
        analyzer.reset()
    """

    def __init__(
        self,
        model: nn.Module,
        enabled: bool = True,
        layer_pattern: Optional[str] = None,
        reduce_across_ranks: bool = True,
    ):
        """
        Initialize the gradient analyzer.

        Args:
            model: The model whose gradients to analyze
            enabled: Whether to actually compute metrics (can be disabled for performance)
            layer_pattern: Optional pattern to filter which layers to analyze (e.g., "layers.")
            reduce_across_ranks: Whether to all-reduce gradients across distributed ranks
        """
        self.model = model
        self.enabled = enabled
        self.layer_pattern = layer_pattern
        self.reduce_across_ranks = reduce_across_ranks

        # Storage for accumulated gradients per domain
        self._domain_gradients: dict[str, torch.Tensor] = {}
        self._domain_sample_counts: dict[str, int] = defaultdict(int)
        self._domain_loss_sums: dict[str, float] = defaultdict(float)
        self._total_params = None

    def _get_total_params(self) -> int:
        """Get total number of parameters to track."""
        if self._total_params is None:
            self._total_params = sum(
                p.numel()
                for name, p in self.model.named_parameters()
                if p.requires_grad and (self.layer_pattern is None or self.layer_pattern in name)
            )
        return self._total_params

    def _flatten_gradients(self) -> torch.Tensor:
        """Flatten all model gradients into a single vector."""
        grads = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if self.layer_pattern is None or self.layer_pattern in name:
                    grads.append(p.grad.detach().view(-1))

        if not grads:
            return torch.zeros(1, device=next(self.model.parameters()).device)

        return torch.cat(grads)

    def accumulate_domain_gradients(
        self,
        domains: list[str],
        sample_losses: Optional[torch.Tensor] = None,
        scale_factor: float = 1.0,
    ):
        """
        Accumulate current gradients attributed to each domain.

        This should be called AFTER loss.backward() for a micro-batch.
        The gradients are attributed to domains based on the proportion of
        samples from each domain in the micro-batch.

        Args:
            domains: List of domain names for each sample in the micro-batch
            sample_losses: Optional per-sample losses for weighted attribution
            scale_factor: Scale factor for gradient (e.g., for gradient accumulation)
        """
        if not self.enabled:
            return

        # Get current gradients
        flat_grad = self._flatten_gradients()
        if flat_grad.numel() == 0:
            return

        # Count samples per domain
        domain_counts = defaultdict(int)
        for d in domains:
            domain_counts[d] += 1

        total_samples = len(domains)

        # Attribute gradients proportionally to each domain
        for domain, count in domain_counts.items():
            # Proportion of this micro-batch from this domain
            proportion = count / total_samples

            # Weighted gradient contribution from this domain
            domain_grad_contribution = flat_grad * proportion * scale_factor

            if domain not in self._domain_gradients:
                self._domain_gradients[domain] = torch.zeros_like(flat_grad)

            self._domain_gradients[domain] += domain_grad_contribution
            self._domain_sample_counts[domain] += count

            # Track losses if provided
            if sample_losses is not None:
                domain_indices = [i for i, d in enumerate(domains) if d == domain]
                if domain_indices:
                    domain_loss = sample_losses[domain_indices].sum().item()
                    self._domain_loss_sums[domain] += domain_loss

    def compute_interference_matrix(self) -> GradientAnalysisResult:
        """
        Compute the interference matrix and domain metrics.

        The interference matrix I[i,j] represents the cosine similarity between
        gradients from domain i and domain j:
        - I[i,j] = 1: Gradients are perfectly aligned (helpful)
        - I[i,j] = 0: Gradients are orthogonal (neutral)
        - I[i,j] = -1: Gradients are opposite (conflicting)

        Returns:
            GradientAnalysisResult containing interference matrix and per-domain info
        """
        if not self.enabled or not self._domain_gradients:
            return GradientAnalysisResult()

        domain_names = sorted(self._domain_gradients.keys())
        n_domains = len(domain_names)

        # Optionally reduce gradients across ranks
        if self.reduce_across_ranks and dist.is_initialized():
            for domain in domain_names:
                dist.all_reduce(self._domain_gradients[domain], op=dist.ReduceOp.SUM)

        # Compute per-domain info
        domain_info = {}
        domain_directions = {}

        for domain in domain_names:
            grad = self._domain_gradients[domain]
            magnitude = torch.norm(grad).item()

            # Normalize for direction
            if magnitude > 1e-8:
                direction = grad / magnitude
            else:
                direction = torch.zeros_like(grad)

            domain_directions[domain] = direction

            info = DomainGradientInfo(
                domain_name=domain,
                gradient_magnitude=magnitude,
                gradient_direction=direction,
                sample_count=self._domain_sample_counts[domain],
                loss_sum=self._domain_loss_sums[domain],
            )
            domain_info[domain] = info

        # Compute interference matrix (cosine similarity)
        if n_domains > 1:
            interference_matrix = torch.zeros(n_domains, n_domains, device=grad.device)

            for i, domain_i in enumerate(domain_names):
                for j, domain_j in enumerate(domain_names):
                    dir_i = domain_directions[domain_i]
                    dir_j = domain_directions[domain_j]

                    # Cosine similarity
                    similarity = torch.dot(dir_i, dir_j)
                    interference_matrix[i, j] = similarity
        else:
            interference_matrix = None

        # Compute overall gradient norm (sum of domain gradients)
        overall_grad = sum(self._domain_gradients.values())
        overall_norm = torch.norm(overall_grad).item()

        return GradientAnalysisResult(
            domain_info=domain_info,
            interference_matrix=interference_matrix,
            domain_names=domain_names,
            overall_gradient_norm=overall_norm,
        )

    def reset(self):
        """Reset accumulated gradients for the next batch."""
        self._domain_gradients.clear()
        self._domain_sample_counts.clear()
        self._domain_loss_sums.clear()


def compute_per_sample_gradients(
    model: nn.Module,
    micro_batch: dict,
    loss_fn: callable,
    domains: list[str],
) -> dict[str, torch.Tensor]:
    """
    Compute per-sample gradients and aggregate by domain.

    This is a more accurate but computationally expensive method that
    computes true per-sample gradients using gradient accumulation.

    Args:
        model: The model to compute gradients for
        micro_batch: Dictionary containing batch inputs
        loss_fn: Function that computes per-sample losses
        domains: List of domain names for each sample

    Returns:
        Dictionary mapping domain names to their aggregated gradient tensors
    """
    # This would require more invasive changes to the training loop
    # and is provided here as a reference implementation
    raise NotImplementedError(
        "Per-sample gradient computation requires vmap or per-sample backward, "
        "which is not implemented. Use accumulate_domain_gradients instead."
    )


def log_interference_matrix_as_heatmap(
    result: GradientAnalysisResult,
    step: int,
    logger_instance: Any,
) -> None:
    """
    Log interference matrix as a heatmap to wandb or other loggers.

    Args:
        result: GradientAnalysisResult from compute_interference_matrix
        step: Current training step
        logger_instance: Logger instance (wandb, tensorboard, etc.)
    """
    if result.interference_matrix is None or len(result.domain_names) < 2:
        return

    try:
        import wandb

        if wandb.run is not None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            matrix = result.interference_matrix.cpu().numpy()

            im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_xticks(range(len(result.domain_names)))
            ax.set_yticks(range(len(result.domain_names)))
            ax.set_xticklabels(result.domain_names, rotation=45, ha="right")
            ax.set_yticklabels(result.domain_names)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Cosine Similarity")

            # Add value annotations
            for i in range(len(result.domain_names)):
                for j in range(len(result.domain_names)):
                    text = ax.text(
                        j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10
                    )

            ax.set_title(f"Gradient Interference Matrix (Step {step})")
            ax.set_xlabel("Target Domain")
            ax.set_ylabel("Source Domain")

            plt.tight_layout()
            wandb.log({"interference/heatmap": wandb.Image(fig)}, step=step)
            plt.close(fig)
    except ImportError:
        pass  # wandb not available
    except Exception as e:
        logger.warning(f"Failed to log interference heatmap: {e}")
