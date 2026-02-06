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
TRACE Loophole Discovery: Unsupervised discovery of reward loopholes.

Once a model starts to learn reward hacking, hacking and non-hacking samples
separate into distinct clusters based on TRACE score. This module uses K-means
clustering on TRACE scores to automatically discover and characterize groups
of samples that may be exploiting loopholes.

Reference:
    Wang et al. (2025). "Is It Thinking or Cheating? Detecting Implicit
    Reward Hacking by Measuring Reasoning Effort." arXiv:2510.01367
    (Algorithm 1: K-means clustering on TRACE scores)
"""

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.trace.config import TRACEConfig

logger = logging.getLogger(__name__)

__all__ = ["TRACELoopholeDiscovery"]


class TRACELoopholeDiscovery:
    """Discovers reward loopholes via clustering on TRACE scores.

    The key observation from the paper is that once a model starts to hack,
    hacking and non-hacking samples form two distinct clusters when projected
    onto the TRACE score axis. By performing K-means clustering (k=2), we can:

    1. Separate hacking vs. non-hacking samples unsupervised
    2. Analyze the characteristics of hacking samples to identify loopholes
    3. Track loophole emergence during training

    Attributes:
        config: TRACE configuration.
        cluster_history: History of clustering results across training steps.
    """

    def __init__(self, config: TRACEConfig):
        self.config = config
        self.cluster_history: list[dict[str, Any]] = []

    def cluster_samples(
        self,
        trace_scores: torch.Tensor,
        n_clusters: Optional[int] = None,
        max_iter: int = 100,
    ) -> dict[str, Any]:
        """Cluster samples based on TRACE scores using K-means.

        Implements Algorithm 1 from the paper: K-means clustering on TRACE
        scores to separate hacking from non-hacking samples.

        Args:
            trace_scores: TRACE scores, shape (num_samples,).
            n_clusters: Number of clusters (default: config.n_clusters, typically 2).
            max_iter: Maximum K-means iterations.

        Returns:
            Dictionary with:
                "labels": Cluster assignment for each sample, shape (num_samples,).
                "centroids": Cluster centroids, shape (n_clusters,).
                "hacking_cluster": Index of the cluster with higher mean TRACE score.
                "legitimate_cluster": Index of the cluster with lower mean TRACE score.
                "hacking_mask": Boolean mask for hacking samples.
                "cluster_sizes": Number of samples per cluster.
                "cluster_means": Mean TRACE score per cluster.
                "separation": Distance between cluster centroids (higher = clearer signal).
        """
        if n_clusters is None:
            n_clusters = self.config.n_clusters

        scores = trace_scores.detach().cpu().numpy().reshape(-1, 1)
        n_samples = scores.shape[0]

        if n_samples < n_clusters:
            logger.warning(
                f"Not enough samples ({n_samples}) for {n_clusters} clusters. "
                "Returning single cluster."
            )
            return {
                "labels": np.zeros(n_samples, dtype=int),
                "centroids": np.array([scores.mean()]),
                "hacking_cluster": 0,
                "legitimate_cluster": 0,
                "hacking_mask": np.zeros(n_samples, dtype=bool),
                "cluster_sizes": {0: n_samples},
                "cluster_means": {0: float(scores.mean())},
                "separation": 0.0,
            }

        # K-means clustering
        labels, centroids = self._kmeans(scores, n_clusters, max_iter)

        # Identify hacking cluster (higher mean TRACE score)
        cluster_means = {}
        cluster_sizes = {}
        for k in range(n_clusters):
            mask = labels == k
            cluster_sizes[k] = int(mask.sum())
            cluster_means[k] = float(scores[mask].mean()) if mask.any() else 0.0

        # The hacking cluster has the higher centroid (more reasoning "shortcuts")
        hacking_cluster = max(range(n_clusters), key=lambda k: centroids[k])
        legitimate_cluster = min(range(n_clusters), key=lambda k: centroids[k])

        hacking_mask = labels == hacking_cluster

        # Compute separation between clusters
        if n_clusters >= 2:
            sorted_centroids = sorted(centroids.flatten())
            separation = sorted_centroids[-1] - sorted_centroids[0]
        else:
            separation = 0.0

        result = {
            "labels": labels,
            "centroids": centroids.flatten(),
            "hacking_cluster": hacking_cluster,
            "legitimate_cluster": legitimate_cluster,
            "hacking_mask": hacking_mask,
            "cluster_sizes": cluster_sizes,
            "cluster_means": cluster_means,
            "separation": separation,
        }

        self.cluster_history.append(result)
        return result

    def _kmeans(
        self,
        data: np.ndarray,
        n_clusters: int,
        max_iter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple K-means implementation for 1D data.

        Args:
            data: Input data, shape (n_samples, 1).
            n_clusters: Number of clusters.
            max_iter: Maximum iterations.

        Returns:
            labels: Cluster labels, shape (n_samples,).
            centroids: Cluster centroids, shape (n_clusters, 1).
        """
        n_samples = data.shape[0]

        # Initialize centroids using quantiles for better convergence
        quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
        centroids = np.percentile(data, quantiles, axis=0).reshape(n_clusters, 1)

        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            # Assignment step
            distances = np.abs(data - centroids.T)  # (n_samples, n_clusters)
            new_labels = distances.argmin(axis=1)

            # Check convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update step
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centroids[k] = data[mask].mean(axis=0)

        return labels, centroids

    def analyze_clusters(
        self,
        clustering_result: dict[str, Any],
        data_sources: Optional[list[str]] = None,
        prompt_ids: Optional[list] = None,
    ) -> dict[str, Any]:
        """Analyze the characteristics of discovered clusters.

        After clustering, this method analyzes the hacking cluster to find
        patterns that could indicate specific loopholes (e.g., certain data
        sources or question types being exploited).

        Args:
            clustering_result: Result from cluster_samples().
            data_sources: Data source identifiers for each sample.
            prompt_ids: Unique prompt identifiers (for grouping analysis).

        Returns:
            Dictionary with analysis results:
                "hacking_by_source": Fraction of hacking samples per data source.
                "source_vulnerability": Data sources ranked by hacking frequency.
                "loophole_candidates": Sources where hacking fraction > 50%.
        """
        hacking_mask = clustering_result["hacking_mask"]
        n_total = len(hacking_mask)
        n_hacking = hacking_mask.sum()

        analysis: dict[str, Any] = {
            "total_samples": n_total,
            "hacking_samples": int(n_hacking),
            "hacking_fraction": float(n_hacking) / max(n_total, 1),
            "separation": clustering_result["separation"],
        }

        # Analyze by data source
        if data_sources is not None:
            source_stats: dict[str, dict[str, int]] = defaultdict(
                lambda: {"total": 0, "hacking": 0}
            )
            for i, source in enumerate(data_sources):
                source_stats[source]["total"] += 1
                if hacking_mask[i]:
                    source_stats[source]["hacking"] += 1

            hacking_by_source = {}
            for source, stats in source_stats.items():
                frac = stats["hacking"] / max(stats["total"], 1)
                hacking_by_source[source] = {
                    "hacking_fraction": frac,
                    "total": stats["total"],
                    "hacking": stats["hacking"],
                }

            # Sort by hacking fraction descending
            source_vulnerability = sorted(
                hacking_by_source.items(),
                key=lambda x: x[1]["hacking_fraction"],
                reverse=True,
            )

            # Identify loophole candidates (sources with > 50% hacking)
            loophole_candidates = [
                (source, info)
                for source, info in source_vulnerability
                if info["hacking_fraction"] > 0.5
            ]

            analysis["hacking_by_source"] = hacking_by_source
            analysis["source_vulnerability"] = source_vulnerability
            analysis["loophole_candidates"] = loophole_candidates

            if loophole_candidates:
                logger.warning(
                    f"TRACE loophole candidates found: "
                    f"{[s for s, _ in loophole_candidates]}"
                )

        # Analyze by prompt (for grouped responses)
        if prompt_ids is not None:
            prompt_stats: dict[Any, dict[str, int]] = defaultdict(
                lambda: {"total": 0, "hacking": 0}
            )
            for i, pid in enumerate(prompt_ids):
                prompt_stats[pid]["total"] += 1
                if hacking_mask[i]:
                    prompt_stats[pid]["hacking"] += 1

            # Prompts where all responses are hacking
            fully_hacked_prompts = [
                pid
                for pid, stats in prompt_stats.items()
                if stats["hacking"] == stats["total"] and stats["total"] > 0
            ]
            analysis["fully_hacked_prompts"] = fully_hacked_prompts
            analysis["num_fully_hacked_prompts"] = len(fully_hacked_prompts)

        return analysis

    def get_metrics(self) -> dict[str, float]:
        """Get metrics from the most recent clustering for logging.

        Returns:
            Dictionary of metrics suitable for logging to wandb/tensorboard.
        """
        if not self.cluster_history:
            return {}

        latest = self.cluster_history[-1]
        metrics = {
            "trace_loophole/separation": latest["separation"],
            "trace_loophole/n_clusters": float(len(latest["cluster_sizes"])),
        }
        for k, size in latest["cluster_sizes"].items():
            metrics[f"trace_loophole/cluster_{k}_size"] = float(size)
        for k, mean in latest["cluster_means"].items():
            metrics[f"trace_loophole/cluster_{k}_mean"] = mean

        hacking_frac = float(latest["hacking_mask"].sum()) / max(
            len(latest["hacking_mask"]), 1
        )
        metrics["trace_loophole/hacking_fraction"] = hacking_frac

        return metrics
