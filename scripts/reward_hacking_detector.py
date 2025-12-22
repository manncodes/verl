# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Reward Hacking Detection Module for VERL

This module provides tools to detect reward hacking in RL training, specifically
targeting issues common in GRPO and IF-GRPO such as:
- Token/phrase repetition exploitation
- Sudden score anomalies
- Output pattern degradation

Usage:
    # CLI
    python reward_hacking_detector.py analyze /path/to/rollout_data_dir
    python reward_hacking_detector.py diff /path/to/rollout_data_dir --step1 100 --step2 200
    python reward_hacking_detector.py visualize /path/to/rollout_data_dir --output dashboard.html
    python reward_hacking_detector.py visualize /path/to/rollout_data_dir --type diversity

    # Python API
    from scripts.reward_hacking_detector import RewardHackingDetector, HackingVisualizer
    detector = RewardHackingDetector("/path/to/rollout_data_dir")
    results = detector.analyze()

    # Visualization (requires plotly: pip install plotly)
    visualizer = HackingVisualizer(detector)
    fig = visualizer.create_dashboard("dashboard.html")  # Full dashboard
    fig = visualizer.plot_score_trends()  # Individual plots
    fig = visualizer.plot_diversity_metrics()
    fig = visualizer.plot_indicator_heatmap()
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import typer
    HAS_TYPER = True
except ImportError:
    typer = None
    HAS_TYPER = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAS_PLOTLY = False


# =============================================================================
# Progress Tracking Utilities
# =============================================================================

class ProgressTracker:
    """Simple progress tracker with timing information."""

    def __init__(self, total: int, desc: str = "", disable: bool = False):
        self.total = total
        self.desc = desc
        self.disable = disable
        self.current = 0
        self.start_time = time.time()
        self.last_print_time = 0
        self.print_interval = 0.5  # Print at most every 0.5 seconds

    def update(self, n: int = 1, extra_info: str = ""):
        """Update progress by n steps."""
        self.current += n

        if self.disable:
            return

        current_time = time.time()
        if current_time - self.last_print_time < self.print_interval and self.current < self.total:
            return

        self.last_print_time = current_time
        self._print_progress(extra_info)

    def _print_progress(self, extra_info: str = ""):
        """Print progress bar to stderr."""
        elapsed = time.time() - self.start_time
        pct = self.current / max(1, self.total) * 100

        # Estimate remaining time
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f"ETA: {remaining:.1f}s"
        else:
            eta_str = "ETA: --"

        # Build progress bar
        bar_width = 30
        filled = int(bar_width * self.current / max(1, self.total))
        bar = "█" * filled + "░" * (bar_width - filled)

        # Format output
        info = f" | {extra_info}" if extra_info else ""
        line = f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({pct:.1f}%) {eta_str}{info}"

        # Pad to overwrite previous line
        sys.stderr.write(line.ljust(120) + "\r")
        sys.stderr.flush()

        if self.current >= self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def finish(self, message: str = ""):
        """Mark as complete with optional message."""
        self.current = self.total
        elapsed = time.time() - self.start_time

        if not self.disable:
            final_msg = f"{self.desc}: Done in {elapsed:.2f}s"
            if message:
                final_msg += f" | {message}"
            sys.stderr.write(f"\r{final_msg.ljust(120)}\n")
            sys.stderr.flush()


def log_step(message: str, verbose: bool = True):
    """Log a step message with timestamp."""
    if verbose:
        timestamp = time.strftime("%H:%M:%S")
        sys.stderr.write(f"[{timestamp}] {message}\n")
        sys.stderr.flush()


@dataclass
class Sample:
    """A single rollout sample from JSONL."""

    input: str
    output: str
    score: float
    step: int
    gts: Optional[Any] = None
    request_id: Optional[str] = None
    sample_idx: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, sample_idx: int = 0) -> "Sample":
        return cls(
            input=d.get("input", ""),
            output=d.get("output", ""),
            score=float(d.get("score", 0)),
            step=int(d.get("step", 0)),
            gts=d.get("gts"),
            request_id=d.get("request_id"),
            sample_idx=sample_idx,
            extra={k: v for k, v in d.items()
                   if k not in {"input", "output", "score", "step", "gts", "request_id"}},
        )


@dataclass
class RepetitionMetrics:
    """Metrics for detecting token/phrase repetition."""

    total_tokens: int = 0
    unique_tokens: int = 0
    token_repetition_ratio: float = 0.0
    max_consecutive_repeats: int = 0
    repeated_phrase: Optional[str] = None
    repeated_phrase_count: int = 0
    ngram_repetition_scores: dict = field(default_factory=dict)  # n -> repetition ratio

    @property
    def is_suspicious(self) -> bool:
        """Check if repetition metrics indicate reward hacking."""
        # High token repetition (low unique/total ratio)
        if self.total_tokens > 10 and self.token_repetition_ratio < 0.3:
            return True
        # Many consecutive repeats
        if self.max_consecutive_repeats > 5:
            return True
        # High n-gram repetition for any n
        for n, ratio in self.ngram_repetition_scores.items():
            if n >= 3 and ratio > 0.5:  # >50% repeated 3-grams is suspicious
                return True
        return False


@dataclass
class ScoreAnomaly:
    """Detected score anomaly."""

    step: int
    sample_idx: int
    score: float
    expected_range: tuple[float, float]
    anomaly_type: str  # "spike", "drop", "outlier"
    zscore: Optional[float] = None
    request_id: Optional[str] = None


@dataclass
class HackingIndicator:
    """A detected reward hacking indicator."""

    step: int
    sample_idx: int
    indicator_type: str  # "repetition", "score_anomaly", "pattern_collapse"
    severity: str  # "low", "medium", "high"
    description: str
    sample: Optional[Sample] = None
    metrics: Optional[dict] = None


@dataclass
class DiffResult:
    """Result of diffing rollouts between two steps."""

    step1: int
    step2: int
    samples_compared: int
    score_changes: list[dict] = field(default_factory=list)
    output_changes: list[dict] = field(default_factory=list)
    new_repetitions: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class RolloutLoader:
    """Lazy loader for rollout JSONL files."""

    def __init__(self, rollout_dir: str | Path):
        self.rollout_dir = Path(rollout_dir)
        if not self.rollout_dir.exists():
            raise FileNotFoundError(f"Rollout directory not found: {rollout_dir}")

        self._step_files: dict[int, Path] = {}
        self._cache: dict[int, list[Sample]] = {}
        self._scan_files()

    def _scan_files(self):
        """Scan directory for JSONL files and map steps."""
        for p in self.rollout_dir.glob("*.jsonl"):
            try:
                step = int(p.stem)
                self._step_files[step] = p
            except ValueError:
                continue

        if not self._step_files:
            raise ValueError(f"No valid JSONL files found in {self.rollout_dir}")

    @property
    def steps(self) -> list[int]:
        """Return sorted list of available steps."""
        return sorted(self._step_files.keys())

    @property
    def num_steps(self) -> int:
        return len(self._step_files)

    def load_step(self, step: int, use_cache: bool = True) -> list[Sample]:
        """Load all samples from a specific step."""
        if step not in self._step_files:
            raise KeyError(f"Step {step} not found. Available: {self.steps}")

        if use_cache and step in self._cache:
            return self._cache[step]

        samples = []
        with open(self._step_files[step], "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    d = json.loads(line)
                    samples.append(Sample.from_dict(d, sample_idx=idx))

        if use_cache:
            self._cache[step] = samples

        return samples

    def iter_steps(self, start: Optional[int] = None, end: Optional[int] = None) -> Iterator[tuple[int, list[Sample]]]:
        """Iterate over steps in order, optionally within a range."""
        for step in self.steps:
            if start is not None and step < start:
                continue
            if end is not None and step > end:
                break
            yield step, self.load_step(step)

    def clear_cache(self):
        """Clear the sample cache to free memory."""
        self._cache.clear()


class RepetitionAnalyzer:
    """Analyzes text for repetition patterns common in reward hacking."""

    def __init__(
        self,
        ngram_sizes: tuple[int, ...] = (2, 3, 4, 5),
        min_repeat_length: int = 3,
    ):
        self.ngram_sizes = ngram_sizes
        self.min_repeat_length = min_repeat_length

    def analyze(self, text: str) -> RepetitionMetrics:
        """Analyze text for repetition patterns."""
        # Tokenize (simple whitespace + punctuation split)
        tokens = self._tokenize(text)

        if not tokens:
            return RepetitionMetrics()

        metrics = RepetitionMetrics(
            total_tokens=len(tokens),
            unique_tokens=len(set(tokens)),
        )

        # Token-level repetition ratio
        metrics.token_repetition_ratio = metrics.unique_tokens / max(1, metrics.total_tokens)

        # Find max consecutive repeats
        metrics.max_consecutive_repeats = self._max_consecutive_repeats(tokens)

        # N-gram repetition analysis
        for n in self.ngram_sizes:
            if len(tokens) >= n:
                ratio, top_phrase, count = self._ngram_repetition(tokens, n)
                metrics.ngram_repetition_scores[n] = ratio

                # Track most repeated phrase (prefer longer n-grams)
                if count > metrics.repeated_phrase_count and len(top_phrase) >= self.min_repeat_length:
                    metrics.repeated_phrase = top_phrase
                    metrics.repeated_phrase_count = count

        return metrics

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Split on whitespace and punctuation, keeping meaningful tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return tokens

    def _max_consecutive_repeats(self, tokens: list[str]) -> int:
        """Find maximum consecutive identical tokens."""
        if not tokens:
            return 0

        max_repeats = 1
        current_repeats = 1

        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                current_repeats += 1
                max_repeats = max(max_repeats, current_repeats)
            else:
                current_repeats = 1

        return max_repeats

    def _ngram_repetition(self, tokens: list[str], n: int) -> tuple[float, str, int]:
        """Calculate n-gram repetition ratio and find most repeated phrase."""
        if len(tokens) < n:
            return 0.0, "", 0

        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counter = Counter(ngrams)

        total = len(ngrams)
        repeated = sum(c - 1 for c in counter.values() if c > 1)
        ratio = repeated / max(1, total)

        if counter:
            top_phrase, top_count = counter.most_common(1)[0]
            return ratio, top_phrase, top_count

        return ratio, "", 0


class ScoreAnalyzer:
    """Analyzes score distributions for anomalies."""

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        spike_threshold: float = 2.0,  # Factor increase from moving average
        window_size: int = 5,
    ):
        self.zscore_threshold = zscore_threshold
        self.spike_threshold = spike_threshold
        self.window_size = window_size

    def analyze_step(self, samples: list[Sample], historical_stats: Optional[dict] = None) -> list[ScoreAnomaly]:
        """Analyze scores within a single step for anomalies."""
        if not samples:
            return []

        scores = [s.score for s in samples]

        if np is None:
            # Fallback without numpy
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = variance ** 0.5
        else:
            mean = np.mean(scores)
            std = np.std(scores)

        anomalies = []

        for sample in samples:
            zscore = (sample.score - mean) / max(std, 1e-8)

            if abs(zscore) > self.zscore_threshold:
                anomalies.append(ScoreAnomaly(
                    step=sample.step,
                    sample_idx=sample.sample_idx,
                    score=sample.score,
                    expected_range=(mean - 2*std, mean + 2*std),
                    anomaly_type="outlier" if zscore > 0 else "low_outlier",
                    zscore=zscore,
                    request_id=sample.request_id,
                ))

        return anomalies

    def analyze_trend(
        self,
        step_stats: list[tuple[int, float, float]],  # (step, mean, std)
    ) -> list[dict]:
        """Analyze score trends across steps for sudden changes."""
        if len(step_stats) < 3:
            return []

        anomalies = []

        for i in range(self.window_size, len(step_stats)):
            window = step_stats[max(0, i - self.window_size):i]
            window_mean = sum(s[1] for s in window) / len(window)

            current_step, current_mean, _ = step_stats[i]

            # Detect sudden spikes
            if window_mean > 0 and current_mean / window_mean > self.spike_threshold:
                anomalies.append({
                    "step": current_step,
                    "type": "spike",
                    "current_mean": current_mean,
                    "window_mean": window_mean,
                    "factor": current_mean / window_mean,
                })
            elif window_mean > 0 and current_mean / window_mean < 1 / self.spike_threshold:
                anomalies.append({
                    "step": current_step,
                    "type": "drop",
                    "current_mean": current_mean,
                    "window_mean": window_mean,
                    "factor": current_mean / window_mean,
                })

        return anomalies


# =============================================================================
# NEW ANALYZERS: Compression, Diversity, Length, GRPO-specific
# =============================================================================

import zlib


@dataclass
class CompressionMetrics:
    """Metrics from compression-based analysis."""

    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 1.0  # compressed/original, lower = more repetitive
    is_suspicious: bool = False

    @classmethod
    def from_text(cls, text: str, threshold: float = 0.3) -> "CompressionMetrics":
        """Create metrics from text. Low ratio = high repetition."""
        if not text:
            return cls()

        original = text.encode("utf-8")
        compressed = zlib.compress(original, level=9)

        ratio = len(compressed) / len(original)

        return cls(
            original_size=len(original),
            compressed_size=len(compressed),
            compression_ratio=ratio,
            is_suspicious=ratio < threshold,
        )


@dataclass
class DiversityMetrics:
    """Metrics for output diversity across samples."""

    num_samples: int = 0
    vocab_size: int = 0
    avg_output_length: float = 0.0
    self_bleu_2: float = 0.0  # 2-gram self-BLEU
    self_bleu_4: float = 0.0  # 4-gram self-BLEU
    jaccard_similarity: float = 0.0  # Average pairwise Jaccard
    distinct_1: float = 0.0  # Distinct unigrams / total unigrams
    distinct_2: float = 0.0  # Distinct bigrams / total bigrams

    @property
    def is_mode_collapse(self) -> bool:
        """Check if metrics indicate mode collapse."""
        # High self-BLEU = low diversity = mode collapse
        if self.self_bleu_4 > 0.7:
            return True
        # High Jaccard = outputs too similar
        if self.jaccard_similarity > 0.6:
            return True
        # Low distinct-2 = repetitive vocabulary
        if self.num_samples > 5 and self.distinct_2 < 0.1:
            return True
        return False


@dataclass
class LengthMetrics:
    """Metrics for output length analysis."""

    mean_length: float = 0.0
    std_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    near_max_ratio: float = 0.0  # Fraction of outputs near max length
    length_entropy: float = 0.0  # Entropy of length distribution

    @property
    def is_length_hacking(self) -> bool:
        """Check if lengths indicate length exploitation."""
        # Most outputs at max length = budget stuffing
        if self.near_max_ratio > 0.5:
            return True
        # Very low length variance = formulaic outputs
        if self.mean_length > 50 and self.std_length < 5:
            return True
        return False


class CompressionAnalyzer:
    """Analyzes text using compression ratios (LZ-based)."""

    def __init__(self, suspicious_threshold: float = 0.3):
        self.suspicious_threshold = suspicious_threshold

    def analyze(self, text: str) -> CompressionMetrics:
        """Analyze single text for compression-based repetition."""
        return CompressionMetrics.from_text(text, self.suspicious_threshold)

    def analyze_batch(self, texts: list[str]) -> list[CompressionMetrics]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]

    def combined_compression(self, texts: list[str]) -> float:
        """Compress all texts together - high similarity = low ratio."""
        if not texts:
            return 1.0

        combined = "\n".join(texts).encode("utf-8")
        compressed = zlib.compress(combined, level=9)

        return len(compressed) / len(combined)


class DiversityAnalyzer:
    """Analyzes diversity across multiple outputs (mode collapse detection)."""

    def __init__(self, ngram_sizes: tuple[int, ...] = (2, 4)):
        self.ngram_sizes = ngram_sizes

    def analyze(self, outputs: list[str]) -> DiversityMetrics:
        """Analyze diversity across a list of outputs."""
        if not outputs:
            return DiversityMetrics()

        tokenized = [self._tokenize(o) for o in outputs]
        all_tokens = [t for tokens in tokenized for t in tokens]

        metrics = DiversityMetrics(
            num_samples=len(outputs),
            vocab_size=len(set(all_tokens)),
            avg_output_length=sum(len(o) for o in outputs) / len(outputs),
        )

        # Distinct-n metrics
        if all_tokens:
            metrics.distinct_1 = len(set(all_tokens)) / len(all_tokens)

            bigrams = list(zip(all_tokens[:-1], all_tokens[1:]))
            if bigrams:
                metrics.distinct_2 = len(set(bigrams)) / len(bigrams)

        # Self-BLEU (approximate)
        metrics.self_bleu_2 = self._self_bleu(tokenized, n=2)
        metrics.self_bleu_4 = self._self_bleu(tokenized, n=4)

        # Jaccard similarity
        metrics.jaccard_similarity = self._avg_jaccard(tokenized)

        return metrics

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: list[str], n: int) -> set[tuple]:
        """Get n-grams as a set."""
        if len(tokens) < n:
            return set()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def _self_bleu(self, tokenized_outputs: list[list[str]], n: int = 4) -> float:
        """Compute approximate self-BLEU using n-gram overlap."""
        if len(tokenized_outputs) < 2:
            return 0.0

        ngram_sets = [self._get_ngrams(tokens, n) for tokens in tokenized_outputs]

        overlaps = []
        for i, ng_i in enumerate(ngram_sets):
            if not ng_i:
                continue
            # Compare to all other outputs
            others = set().union(*[ng for j, ng in enumerate(ngram_sets) if j != i])
            if others:
                overlap = len(ng_i & others) / len(ng_i)
                overlaps.append(overlap)

        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    def _avg_jaccard(self, tokenized_outputs: list[list[str]]) -> float:
        """Compute average pairwise Jaccard similarity."""
        if len(tokenized_outputs) < 2:
            return 0.0

        sets = [set(tokens) for tokens in tokenized_outputs]
        similarities = []

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return sum(similarities) / len(similarities) if similarities else 0.0


class LengthAnalyzer:
    """Analyzes output lengths for exploitation patterns."""

    def __init__(self, near_max_threshold: float = 0.9):
        self.near_max_threshold = near_max_threshold

    def analyze(self, outputs: list[str]) -> LengthMetrics:
        """Analyze length distribution of outputs."""
        if not outputs:
            return LengthMetrics()

        lengths = [len(o) for o in outputs]

        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        std = variance ** 0.5

        max_len = max(lengths)
        near_max_count = sum(1 for l in lengths if l >= max_len * self.near_max_threshold)

        # Compute entropy of length distribution (binned)
        length_entropy = self._length_entropy(lengths)

        return LengthMetrics(
            mean_length=mean,
            std_length=std,
            min_length=min(lengths),
            max_length=max_len,
            near_max_ratio=near_max_count / len(lengths),
            length_entropy=length_entropy,
        )

    def _length_entropy(self, lengths: list[int], num_bins: int = 10) -> float:
        """Compute entropy of length distribution."""
        if not lengths:
            return 0.0

        min_l, max_l = min(lengths), max(lengths)
        if min_l == max_l:
            return 0.0  # All same length

        bin_size = (max_l - min_l) / num_bins
        bins = [0] * num_bins

        for l in lengths:
            bin_idx = min(int((l - min_l) / bin_size), num_bins - 1)
            bins[bin_idx] += 1

        # Compute entropy
        total = len(lengths)
        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / total
                entropy -= p * (p if np is None else float(np.log2(p)))

        # Normalize by max entropy
        max_entropy = float(np.log2(num_bins)) if np else 3.32  # log2(10)
        return entropy / max_entropy if max_entropy > 0 else 0.0


class GRPOAnalyzer:
    """GRPO-specific analysis for advantage and group dynamics."""

    def analyze_groups(
        self,
        samples: list[Sample],
        group_key: str = "request_id",
    ) -> dict:
        """Analyze GRPO group dynamics."""
        # Group samples by key (typically request_id or uid)
        groups: dict[str, list[Sample]] = defaultdict(list)

        for s in samples:
            key = getattr(s, group_key, None) or s.extra.get(group_key, str(s.sample_idx))
            groups[key].append(s)

        results = {
            "num_groups": len(groups),
            "avg_group_size": sum(len(g) for g in groups.values()) / max(1, len(groups)),
            "winner_concentration": 0.0,
            "advantage_variance": 0.0,
            "same_output_ratio": 0.0,
        }

        if len(groups) < 2:
            return results

        # Analyze each group
        winner_positions = []
        intra_group_variances = []
        same_output_groups = 0

        for group_id, group_samples in groups.items():
            if len(group_samples) < 2:
                continue

            scores = [s.score for s in group_samples]

            # Winner position (which index wins)
            winner_idx = scores.index(max(scores))
            winner_positions.append(winner_idx)

            # Score variance within group
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            intra_group_variances.append(variance)

            # Check if outputs are identical
            outputs = [s.output for s in group_samples]
            if len(set(outputs)) == 1:
                same_output_groups += 1

        # Winner concentration (does same position always win?)
        if winner_positions:
            counts = Counter(winner_positions)
            results["winner_concentration"] = max(counts.values()) / len(winner_positions)

        # Average intra-group variance
        if intra_group_variances:
            results["advantage_variance"] = sum(intra_group_variances) / len(intra_group_variances)

        # Ratio of groups with identical outputs
        multi_sample_groups = sum(1 for g in groups.values() if len(g) > 1)
        if multi_sample_groups > 0:
            results["same_output_ratio"] = same_output_groups / multi_sample_groups

        return results


class RewardHackingDetector:
    """Main class for detecting reward hacking in rollout data."""

    def __init__(
        self,
        rollout_dir: str | Path,
        repetition_analyzer: Optional[RepetitionAnalyzer] = None,
        score_analyzer: Optional[ScoreAnalyzer] = None,
        compression_analyzer: Optional[CompressionAnalyzer] = None,
        diversity_analyzer: Optional[DiversityAnalyzer] = None,
        length_analyzer: Optional[LengthAnalyzer] = None,
        grpo_analyzer: Optional[GRPOAnalyzer] = None,
    ):
        self.loader = RolloutLoader(rollout_dir)
        self.repetition_analyzer = repetition_analyzer or RepetitionAnalyzer()
        self.score_analyzer = score_analyzer or ScoreAnalyzer()
        self.compression_analyzer = compression_analyzer or CompressionAnalyzer()
        self.diversity_analyzer = diversity_analyzer or DiversityAnalyzer()
        self.length_analyzer = length_analyzer or LengthAnalyzer()
        self.grpo_analyzer = grpo_analyzer or GRPOAnalyzer()

        # Results storage
        self.indicators: list[HackingIndicator] = []
        self.step_stats: list[tuple[int, float, float]] = []  # (step, mean, std)
        self.repetition_metrics_by_step: dict[int, list[RepetitionMetrics]] = {}
        self.diversity_metrics_by_step: dict[int, DiversityMetrics] = {}
        self.length_metrics_by_step: dict[int, LengthMetrics] = {}
        self.grpo_metrics_by_step: dict[int, dict] = {}

    @property
    def steps(self) -> list[int]:
        return self.loader.steps

    def analyze_step(self, step: int, verbose: bool = False) -> list[HackingIndicator]:
        """Analyze a single step for reward hacking indicators."""
        samples = self.loader.load_step(step)
        indicators = []
        repetition_metrics = []

        # Analyze each sample for repetition
        for sample in samples:
            metrics = self.repetition_analyzer.analyze(sample.output)
            repetition_metrics.append(metrics)

            if metrics.is_suspicious:
                severity = self._assess_repetition_severity(metrics)
                indicators.append(HackingIndicator(
                    step=step,
                    sample_idx=sample.sample_idx,
                    indicator_type="repetition",
                    severity=severity,
                    description=self._format_repetition_description(metrics),
                    sample=sample,
                    metrics={
                        "token_repetition_ratio": metrics.token_repetition_ratio,
                        "max_consecutive_repeats": metrics.max_consecutive_repeats,
                        "repeated_phrase": metrics.repeated_phrase,
                        "repeated_phrase_count": metrics.repeated_phrase_count,
                    },
                ))

        self.repetition_metrics_by_step[step] = repetition_metrics

        # Analyze compression for each sample
        for sample in samples:
            comp_metrics = self.compression_analyzer.analyze(sample.output)
            if comp_metrics.is_suspicious:
                indicators.append(HackingIndicator(
                    step=step,
                    sample_idx=sample.sample_idx,
                    indicator_type="compression",
                    severity="high" if comp_metrics.compression_ratio < 0.2 else "medium",
                    description=f"Low compression ratio {comp_metrics.compression_ratio:.2f} "
                               f"({comp_metrics.compressed_size}/{comp_metrics.original_size} bytes)",
                    sample=sample,
                    metrics={"compression_ratio": comp_metrics.compression_ratio},
                ))

        # Analyze scores for anomalies
        score_anomalies = self.score_analyzer.analyze_step(samples)
        for anomaly in score_anomalies:
            indicators.append(HackingIndicator(
                step=step,
                sample_idx=anomaly.sample_idx,
                indicator_type="score_anomaly",
                severity="high" if abs(anomaly.zscore or 0) > 4 else "medium",
                description=f"Score {anomaly.score:.4f} is {anomaly.anomaly_type} "
                           f"(z-score: {anomaly.zscore:.2f}, expected: {anomaly.expected_range})",
                sample=samples[anomaly.sample_idx] if anomaly.sample_idx < len(samples) else None,
                metrics={"score": anomaly.score, "zscore": anomaly.zscore},
            ))

        # Analyze diversity across all outputs in this step
        outputs = [s.output for s in samples]
        diversity_metrics = self.diversity_analyzer.analyze(outputs)
        self.diversity_metrics_by_step[step] = diversity_metrics

        if diversity_metrics.is_mode_collapse:
            indicators.append(HackingIndicator(
                step=step,
                sample_idx=-1,  # Step-level indicator
                indicator_type="mode_collapse",
                severity="high" if diversity_metrics.self_bleu_4 > 0.8 else "medium",
                description=f"Mode collapse detected: self-BLEU-4={diversity_metrics.self_bleu_4:.2f}, "
                           f"Jaccard={diversity_metrics.jaccard_similarity:.2f}, "
                           f"distinct-2={diversity_metrics.distinct_2:.2f}",
                metrics={
                    "self_bleu_4": diversity_metrics.self_bleu_4,
                    "jaccard_similarity": diversity_metrics.jaccard_similarity,
                    "distinct_2": diversity_metrics.distinct_2,
                    "vocab_size": diversity_metrics.vocab_size,
                },
            ))

        # Analyze length distribution
        length_metrics = self.length_analyzer.analyze(outputs)
        self.length_metrics_by_step[step] = length_metrics

        if length_metrics.is_length_hacking:
            indicators.append(HackingIndicator(
                step=step,
                sample_idx=-1,
                indicator_type="length_hacking",
                severity="medium",
                description=f"Length exploitation: {length_metrics.near_max_ratio:.0%} outputs near max length "
                           f"(mean={length_metrics.mean_length:.0f}, std={length_metrics.std_length:.1f})",
                metrics={
                    "mean_length": length_metrics.mean_length,
                    "std_length": length_metrics.std_length,
                    "near_max_ratio": length_metrics.near_max_ratio,
                },
            ))

        # GRPO-specific analysis
        grpo_metrics = self.grpo_analyzer.analyze_groups(samples)
        self.grpo_metrics_by_step[step] = grpo_metrics

        if grpo_metrics["winner_concentration"] > 0.8 and grpo_metrics["num_groups"] > 5:
            indicators.append(HackingIndicator(
                step=step,
                sample_idx=-1,
                indicator_type="grpo_concentration",
                severity="medium",
                description=f"GRPO winner concentration: {grpo_metrics['winner_concentration']:.0%} "
                           f"(same position wins across {grpo_metrics['num_groups']} groups)",
                metrics=grpo_metrics,
            ))

        if grpo_metrics["same_output_ratio"] > 0.3:
            indicators.append(HackingIndicator(
                step=step,
                sample_idx=-1,
                indicator_type="grpo_identical_outputs",
                severity="high" if grpo_metrics["same_output_ratio"] > 0.5 else "medium",
                description=f"GRPO identical outputs: {grpo_metrics['same_output_ratio']:.0%} of groups "
                           f"have identical responses",
                metrics=grpo_metrics,
            ))

        # Calculate step statistics
        scores = [s.score for s in samples]
        if scores:
            if np is not None:
                self.step_stats.append((step, float(np.mean(scores)), float(np.std(scores))))
            else:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                self.step_stats.append((step, mean, variance ** 0.5))

        self.indicators.extend(indicators)

        if verbose and indicators:
            print(f"Step {step}: Found {len(indicators)} indicators")

        return indicators

    def analyze(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """Analyze all steps for reward hacking indicators.

        Args:
            start_step: First step to analyze (inclusive)
            end_step: Last step to analyze (inclusive)
            verbose: Enable progress tracking and detailed logging
        """
        analysis_start = time.time()

        # Reset state
        self.indicators = []
        self.step_stats = []
        self.repetition_metrics_by_step = {}
        self.diversity_metrics_by_step = {}
        self.length_metrics_by_step = {}
        self.grpo_metrics_by_step = {}

        steps_to_analyze = [
            s for s in self.steps
            if (start_step is None or s >= start_step) and (end_step is None or s <= end_step)
        ]

        if verbose:
            log_step(f"Starting analysis of {len(steps_to_analyze)} steps "
                    f"(steps {steps_to_analyze[0]} to {steps_to_analyze[-1]})", verbose)

        # Progress tracker for steps
        progress = ProgressTracker(
            total=len(steps_to_analyze),
            desc="Analyzing steps",
            disable=not verbose
        )

        # Track running totals for progress info
        total_samples = 0
        total_indicators = 0

        for step in steps_to_analyze:
            step_start = time.time()

            indicators = self.analyze_step(step, verbose=False)  # Don't double-log

            # Update totals
            samples_in_step = len(self.loader.load_step(step, use_cache=True))
            total_samples += samples_in_step
            total_indicators += len(indicators)

            # Update progress with current stats
            step_time = time.time() - step_start
            progress.update(
                1,
                extra_info=f"step {step}: {samples_in_step} samples, "
                          f"{len(indicators)} issues, {step_time:.2f}s"
            )

        progress.finish(f"{total_samples} total samples, {total_indicators} indicators found")

        # Analyze cross-step trends
        if verbose:
            log_step("Analyzing cross-step trends...", verbose)

        trend_anomalies = self.score_analyzer.analyze_trend(self.step_stats)
        for anomaly in trend_anomalies:
            self.indicators.append(HackingIndicator(
                step=anomaly["step"],
                sample_idx=-1,  # Step-level, not sample-level
                indicator_type="score_trend",
                severity="high" if anomaly["factor"] > 3 else "medium",
                description=f"Score {anomaly['type']} detected: "
                           f"mean {anomaly['current_mean']:.4f} vs window avg {anomaly['window_mean']:.4f} "
                           f"(factor: {anomaly['factor']:.2f}x)",
                metrics=anomaly,
            ))

        analysis_time = time.time() - analysis_start

        if verbose:
            log_step(f"Analysis complete in {analysis_time:.2f}s", verbose)
            self._print_analysis_summary()

        return self.get_summary()

    def _print_analysis_summary(self):
        """Print a summary of analysis results to stderr."""
        summary = self.get_summary()

        sys.stderr.write("\n" + "=" * 60 + "\n")
        sys.stderr.write("ANALYSIS SUMMARY\n")
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write(f"Total indicators: {summary['total_indicators']}\n")
        sys.stderr.write(f"Steps with issues: {summary['steps_with_issues']}\n")

        if summary['by_type']:
            sys.stderr.write("\nBy Type:\n")
            for t, count in sorted(summary['by_type'].items(), key=lambda x: -x[1]):
                sys.stderr.write(f"  {t}: {count}\n")

        if summary['by_severity']:
            sys.stderr.write("\nBy Severity:\n")
            for s in ['high', 'medium', 'low']:
                if s in summary['by_severity']:
                    sys.stderr.write(f"  {s}: {summary['by_severity'][s]}\n")

        if summary['most_affected_steps']:
            sys.stderr.write("\nMost Affected Steps:\n")
            for step, count in summary['most_affected_steps'][:5]:
                sys.stderr.write(f"  Step {step}: {count} indicators\n")

        sys.stderr.write("=" * 60 + "\n\n")
        sys.stderr.flush()

    def diff_steps(
        self,
        step1: int,
        step2: int,
        match_by: str = "index",
        verbose: bool = False,
    ) -> DiffResult:
        """Compare rollouts between two steps to identify changes.

        Args:
            step1: First step to compare
            step2: Second step to compare
            match_by: How to match samples - "index" (positional) or "request_id"
            verbose: Enable progress tracking
        """
        diff_start = time.time()

        if verbose:
            log_step(f"Loading step {step1}...", verbose)
        samples1 = self.loader.load_step(step1)

        if verbose:
            log_step(f"Loading step {step2}...", verbose)
        samples2 = self.loader.load_step(step2)

        result = DiffResult(step1=step1, step2=step2, samples_compared=0)

        if match_by == "request_id":
            # Match by request_id
            map1 = {s.request_id: s for s in samples1 if s.request_id}
            map2 = {s.request_id: s for s in samples2 if s.request_id}
            common_ids = set(map1.keys()) & set(map2.keys())

            if verbose:
                log_step(f"Comparing {len(common_ids)} matched samples by request_id...", verbose)

            progress = ProgressTracker(len(common_ids), "Comparing samples", disable=not verbose)

            for rid in common_ids:
                s1, s2 = map1[rid], map2[rid]
                self._compare_samples(s1, s2, result)
                progress.update(1)

            progress.finish()
            result.samples_compared = len(common_ids)
        else:
            # Match by index position
            num_to_compare = min(len(samples1), len(samples2))

            if verbose:
                log_step(f"Comparing {num_to_compare} samples by index...", verbose)

            progress = ProgressTracker(num_to_compare, "Comparing samples", disable=not verbose)

            for i in range(num_to_compare):
                s1, s2 = samples1[i], samples2[i]
                self._compare_samples(s1, s2, result)
                progress.update(1)

            progress.finish()
            result.samples_compared = num_to_compare

        # Compute summary statistics
        if result.score_changes:
            changes = [c["change"] for c in result.score_changes]
            result.summary = {
                "mean_score_change": sum(changes) / len(changes),
                "max_score_increase": max(changes),
                "max_score_decrease": min(changes),
                "samples_with_score_increase": sum(1 for c in changes if c > 0),
                "samples_with_score_decrease": sum(1 for c in changes if c < 0),
                "new_repetition_cases": len(result.new_repetitions),
            }

        diff_time = time.time() - diff_start
        if verbose:
            log_step(f"Diff complete in {diff_time:.2f}s: "
                    f"{result.samples_compared} samples, "
                    f"{len(result.new_repetitions)} new repetitions", verbose)

        return result

    def _compare_samples(self, s1: Sample, s2: Sample, result: DiffResult):
        """Compare two samples and update diff result."""
        # Score change
        score_change = s2.score - s1.score
        result.score_changes.append({
            "sample_idx": s1.sample_idx,
            "request_id": s1.request_id,
            "score1": s1.score,
            "score2": s2.score,
            "change": score_change,
            "percent_change": (score_change / max(abs(s1.score), 1e-8)) * 100,
        })

        # Output change
        if s1.output != s2.output:
            result.output_changes.append({
                "sample_idx": s1.sample_idx,
                "request_id": s1.request_id,
                "output1_len": len(s1.output),
                "output2_len": len(s2.output),
                "output1_preview": s1.output[:200],
                "output2_preview": s2.output[:200],
            })

        # New repetition detection
        m1 = self.repetition_analyzer.analyze(s1.output)
        m2 = self.repetition_analyzer.analyze(s2.output)

        if not m1.is_suspicious and m2.is_suspicious:
            result.new_repetitions.append({
                "sample_idx": s1.sample_idx,
                "request_id": s1.request_id,
                "step1_metrics": {
                    "token_rep_ratio": m1.token_repetition_ratio,
                    "max_consec": m1.max_consecutive_repeats,
                },
                "step2_metrics": {
                    "token_rep_ratio": m2.token_repetition_ratio,
                    "max_consec": m2.max_consecutive_repeats,
                    "repeated_phrase": m2.repeated_phrase,
                },
            })

    def find_hacking_origin(self, indicator_type: str = "repetition") -> list[dict]:
        """Find the earliest step where each type of hacking appeared.

        Useful for identifying which training batch caused reward hacking.
        """
        origins = []
        seen_patterns = set()

        for indicator in sorted(self.indicators, key=lambda x: x.step):
            if indicator.indicator_type != indicator_type:
                continue

            # Create a pattern signature
            if indicator.metrics:
                pattern_key = indicator.metrics.get("repeated_phrase", "")
            else:
                pattern_key = ""

            if pattern_key and pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                origins.append({
                    "first_step": indicator.step,
                    "sample_idx": indicator.sample_idx,
                    "pattern": pattern_key,
                    "severity": indicator.severity,
                    "sample_input": indicator.sample.input[:200] if indicator.sample else None,
                    "sample_output": indicator.sample.output[:500] if indicator.sample else None,
                })

        return origins

    def get_summary(self) -> dict:
        """Get analysis summary."""
        by_type = defaultdict(list)
        by_severity = defaultdict(list)
        by_step = defaultdict(list)

        for ind in self.indicators:
            by_type[ind.indicator_type].append(ind)
            by_severity[ind.severity].append(ind)
            by_step[ind.step].append(ind)

        return {
            "total_indicators": len(self.indicators),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "steps_with_issues": len(by_step),
            "most_affected_steps": sorted(
                [(step, len(inds)) for step, inds in by_step.items()],
                key=lambda x: -x[1]
            )[:10],
            "hacking_origins": self.find_hacking_origin(),
        }

    def _assess_repetition_severity(self, metrics: RepetitionMetrics) -> str:
        """Assess severity of repetition."""
        severity_score = 0

        if metrics.token_repetition_ratio < 0.2:
            severity_score += 2
        elif metrics.token_repetition_ratio < 0.3:
            severity_score += 1

        if metrics.max_consecutive_repeats > 10:
            severity_score += 2
        elif metrics.max_consecutive_repeats > 5:
            severity_score += 1

        for n, ratio in metrics.ngram_repetition_scores.items():
            if n >= 3 and ratio > 0.7:
                severity_score += 2
            elif n >= 3 and ratio > 0.5:
                severity_score += 1

        if severity_score >= 4:
            return "high"
        elif severity_score >= 2:
            return "medium"
        return "low"

    def _format_repetition_description(self, metrics: RepetitionMetrics) -> str:
        """Format human-readable description of repetition."""
        parts = []

        parts.append(f"Token diversity: {metrics.token_repetition_ratio:.1%} "
                    f"({metrics.unique_tokens}/{metrics.total_tokens})")

        if metrics.max_consecutive_repeats > 1:
            parts.append(f"Max consecutive: {metrics.max_consecutive_repeats}")

        if metrics.repeated_phrase:
            parts.append(f"Repeated phrase ({metrics.repeated_phrase_count}x): "
                        f"'{metrics.repeated_phrase[:50]}'")

        return " | ".join(parts)

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a detailed text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("REWARD HACKING DETECTION REPORT")
        lines.append("=" * 80)
        lines.append("")

        summary = self.get_summary()

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total indicators found: {summary['total_indicators']}")
        lines.append(f"Steps with issues: {summary['steps_with_issues']}/{len(self.steps)}")
        lines.append("")

        lines.append("By Type:")
        for t, count in summary["by_type"].items():
            lines.append(f"  - {t}: {count}")
        lines.append("")

        lines.append("By Severity:")
        for s, count in summary["by_severity"].items():
            lines.append(f"  - {s}: {count}")
        lines.append("")

        lines.append("Most Affected Steps:")
        for step, count in summary["most_affected_steps"][:5]:
            lines.append(f"  - Step {step}: {count} indicators")
        lines.append("")

        if summary["hacking_origins"]:
            lines.append("HACKING ORIGINS (First Occurrences)")
            lines.append("-" * 40)
            for origin in summary["hacking_origins"][:10]:
                lines.append(f"\nStep {origin['first_step']}, Sample {origin['sample_idx']}")
                lines.append(f"  Severity: {origin['severity']}")
                if origin["pattern"]:
                    lines.append(f"  Pattern: '{origin['pattern']}'")
                if origin["sample_input"]:
                    lines.append(f"  Input: {origin['sample_input'][:100]}...")
                if origin["sample_output"]:
                    lines.append(f"  Output: {origin['sample_output'][:200]}...")

        lines.append("")
        lines.append("HIGH SEVERITY INDICATORS")
        lines.append("-" * 40)

        high_severity = [i for i in self.indicators if i.severity == "high"]
        for ind in high_severity[:20]:
            lines.append(f"\n[{ind.indicator_type}] Step {ind.step}, Sample {ind.sample_idx}")
            lines.append(f"  {ind.description}")
            if ind.sample:
                lines.append(f"  Output preview: {ind.sample.output[:150]}...")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report


# =============================================================================
# Visualization with Plotly
# =============================================================================

class HackingVisualizer:
    """Plotly-based visualization for reward hacking analysis."""

    def __init__(self, detector: RewardHackingDetector):
        if not HAS_PLOTLY:
            raise ImportError(
                "Plotly is required for visualization. "
                "Install with: pip install plotly"
            )
        self.detector = detector

    def plot_score_trends(
        self,
        title: str = "Score Trends Over Training Steps",
        show_std: bool = True,
    ) -> "go.Figure":
        """Plot mean scores and standard deviations over steps.

        Returns a Plotly figure showing score progression with optional
        error bands for standard deviation.
        """
        if not self.detector.step_stats:
            raise ValueError("No step stats available. Run analyze() first.")

        steps, means, stds = zip(*self.detector.step_stats)

        fig = go.Figure()

        # Main mean line
        fig.add_trace(go.Scatter(
            x=list(steps),
            y=list(means),
            mode='lines+markers',
            name='Mean Score',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=6),
        ))

        # Standard deviation band
        if show_std:
            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]

            fig.add_trace(go.Scatter(
                x=list(steps) + list(reversed(steps)),
                y=upper + list(reversed(lower)),
                fill='toself',
                fillcolor='rgba(33, 150, 243, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True,
            ))

        # Mark anomalies
        trend_anomalies = [
            ind for ind in self.detector.indicators
            if ind.indicator_type == "score_trend"
        ]
        if trend_anomalies:
            anomaly_steps = [a.step for a in trend_anomalies]
            anomaly_scores = [
                means[steps.index(s)] if s in steps else 0
                for s in anomaly_steps
            ]
            fig.add_trace(go.Scatter(
                x=anomaly_steps,
                y=anomaly_scores,
                mode='markers',
                name='Trend Anomalies',
                marker=dict(color='red', size=12, symbol='x'),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Training Step",
            yaxis_title="Score",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def plot_diversity_metrics(
        self,
        title: str = "Diversity Metrics Over Training",
    ) -> "go.Figure":
        """Plot diversity metrics (Self-BLEU, distinct-n, Jaccard) over steps.

        Returns a Plotly figure with multiple traces for different diversity
        metrics, useful for detecting mode collapse.
        """
        if not self.detector.diversity_metrics_by_step:
            raise ValueError("No diversity metrics available. Run analyze() first.")

        steps = sorted(self.detector.diversity_metrics_by_step.keys())

        metrics_data = {
            'self_bleu_4': [],
            'distinct_1': [],
            'distinct_2': [],
            'jaccard_similarity': [],
        }

        for step in steps:
            m = self.detector.diversity_metrics_by_step[step]
            metrics_data['self_bleu_4'].append(m.self_bleu_4)
            metrics_data['distinct_1'].append(m.distinct_1)
            metrics_data['distinct_2'].append(m.distinct_2)
            metrics_data['jaccard_similarity'].append(m.jaccard_similarity)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Self-BLEU-4 (lower is better)',
                'Jaccard Similarity (lower is better)',
                'Distinct-1 (higher is better)',
                'Distinct-2 (higher is better)',
            ),
        )

        # Self-BLEU (higher = more similar = bad)
        fig.add_trace(
            go.Scatter(x=steps, y=metrics_data['self_bleu_4'],
                      mode='lines+markers', name='Self-BLEU-4',
                      line=dict(color='#F44336')),
            row=1, col=1
        )
        # Add threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                     annotation_text="Mode collapse threshold", row=1, col=1)

        # Jaccard similarity
        fig.add_trace(
            go.Scatter(x=steps, y=metrics_data['jaccard_similarity'],
                      mode='lines+markers', name='Jaccard',
                      line=dict(color='#FF9800')),
            row=1, col=2
        )
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                     annotation_text="High similarity threshold", row=1, col=2)

        # Distinct-1
        fig.add_trace(
            go.Scatter(x=steps, y=metrics_data['distinct_1'],
                      mode='lines+markers', name='Distinct-1',
                      line=dict(color='#4CAF50')),
            row=2, col=1
        )

        # Distinct-2
        fig.add_trace(
            go.Scatter(x=steps, y=metrics_data['distinct_2'],
                      mode='lines+markers', name='Distinct-2',
                      line=dict(color='#2196F3')),
            row=2, col=2
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                     annotation_text="Low diversity threshold", row=2, col=2)

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=600,
            showlegend=False,
        )

        return fig

    def plot_indicator_heatmap(
        self,
        title: str = "Reward Hacking Indicators by Step",
    ) -> "go.Figure":
        """Create a heatmap of indicator counts by type and step.

        Returns a Plotly heatmap showing the density of different
        indicator types across training steps.
        """
        if not self.detector.indicators:
            raise ValueError("No indicators available. Run analyze() first.")

        # Collect indicator types and steps
        indicator_types = list(set(ind.indicator_type for ind in self.detector.indicators))
        steps = sorted(set(ind.step for ind in self.detector.indicators))

        # Build count matrix
        counts = {t: {s: 0 for s in steps} for t in indicator_types}
        for ind in self.detector.indicators:
            counts[ind.indicator_type][ind.step] += 1

        # Convert to matrix
        z_matrix = []
        for t in indicator_types:
            z_matrix.append([counts[t][s] for s in steps])

        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=[str(s) for s in steps],
            y=indicator_types,
            colorscale='Reds',
            hovertemplate='Step %{x}<br>Type: %{y}<br>Count: %{z}<extra></extra>',
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Training Step",
            yaxis_title="Indicator Type",
            template="plotly_white",
        )

        return fig

    def plot_severity_timeline(
        self,
        title: str = "Indicator Severity Timeline",
    ) -> "go.Figure":
        """Plot indicators over time colored by severity.

        Returns a scatter plot showing when indicators occurred,
        with color indicating severity level.
        """
        if not self.detector.indicators:
            raise ValueError("No indicators available. Run analyze() first.")

        severity_colors = {
            'low': '#4CAF50',
            'medium': '#FF9800',
            'high': '#F44336',
        }

        fig = go.Figure()

        for severity in ['low', 'medium', 'high']:
            inds = [i for i in self.detector.indicators if i.severity == severity]
            if inds:
                fig.add_trace(go.Scatter(
                    x=[i.step for i in inds],
                    y=[i.indicator_type for i in inds],
                    mode='markers',
                    name=severity.capitalize(),
                    marker=dict(
                        color=severity_colors[severity],
                        size=10 if severity == 'high' else 8,
                        symbol='circle' if severity != 'high' else 'x',
                    ),
                    text=[i.description[:50] for i in inds],
                    hovertemplate='Step %{x}<br>%{y}<br>%{text}<extra></extra>',
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Training Step",
            yaxis_title="Indicator Type",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        return fig

    def plot_length_distribution(
        self,
        steps: Optional[list[int]] = None,
        title: str = "Output Length Distribution Over Steps",
    ) -> "go.Figure":
        """Plot output length statistics over training steps.

        Returns a Plotly figure showing length mean/std and near-max ratio.
        """
        if not self.detector.length_metrics_by_step:
            raise ValueError("No length metrics available. Run analyze() first.")

        if steps is None:
            steps = sorted(self.detector.length_metrics_by_step.keys())

        means = []
        stds = []
        near_max_ratios = []

        for step in steps:
            if step in self.detector.length_metrics_by_step:
                m = self.detector.length_metrics_by_step[step]
                means.append(m.mean_length)
                stds.append(m.std_length)
                near_max_ratios.append(m.near_max_ratio)
            else:
                means.append(0)
                stds.append(0)
                near_max_ratios.append(0)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Mean Output Length', 'Near-Max Length Ratio'),
            shared_xaxes=True,
        )

        # Mean length with error bars
        fig.add_trace(
            go.Scatter(
                x=steps, y=means,
                mode='lines+markers',
                name='Mean Length',
                line=dict(color='#2196F3'),
                error_y=dict(type='data', array=stds, visible=True),
            ),
            row=1, col=1
        )

        # Near-max ratio (length hacking indicator)
        fig.add_trace(
            go.Scatter(
                x=steps, y=near_max_ratios,
                mode='lines+markers',
                name='Near-Max Ratio',
                line=dict(color='#F44336'),
                fill='tozeroy',
                fillcolor='rgba(244, 67, 54, 0.2)',
            ),
            row=2, col=1
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Length hacking threshold", row=2, col=1)

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=500,
            xaxis2_title="Training Step",
        )

        return fig

    def plot_grpo_metrics(
        self,
        title: str = "GRPO Group Dynamics",
    ) -> "go.Figure":
        """Plot GRPO-specific metrics over training.

        Returns a Plotly figure showing winner concentration and
        identical output ratio - indicators of GRPO-specific issues.
        """
        if not self.detector.grpo_metrics_by_step:
            raise ValueError("No GRPO metrics available. Run analyze() first.")

        steps = sorted(self.detector.grpo_metrics_by_step.keys())
        winner_conc = []
        same_output = []
        adv_var = []

        for step in steps:
            m = self.detector.grpo_metrics_by_step[step]
            winner_conc.append(m.get('winner_concentration', 0))
            same_output.append(m.get('same_output_ratio', 0))
            adv_var.append(m.get('advantage_variance', 0))

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Winner Concentration & Same Output Ratio',
                'Intra-Group Advantage Variance'
            ),
            shared_xaxes=True,
        )

        fig.add_trace(
            go.Scatter(x=steps, y=winner_conc,
                      mode='lines+markers', name='Winner Concentration',
                      line=dict(color='#9C27B0')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=same_output,
                      mode='lines+markers', name='Same Output Ratio',
                      line=dict(color='#F44336')),
            row=1, col=1
        )
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                     annotation_text="High concentration", row=1, col=1)

        fig.add_trace(
            go.Scatter(x=steps, y=adv_var,
                      mode='lines+markers', name='Advantage Variance',
                      line=dict(color='#2196F3'),
                      fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.2)'),
            row=2, col=1
        )

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=500,
            xaxis2_title="Training Step",
        )

        return fig

    def plot_compression_ratio(
        self,
        title: str = "Compression Ratio Distribution by Step",
    ) -> "go.Figure":
        """Plot compression ratio statistics as box plots per step.

        Low compression ratio indicates high repetition.
        """
        # We need to re-compute compression metrics per sample
        # This is expensive, so we'll use a sampling approach or
        # rely on cached data if available

        steps = sorted(self.detector.repetition_metrics_by_step.keys())

        fig = go.Figure()

        # Use token repetition ratio as a proxy (already computed)
        for step in steps:
            metrics = self.detector.repetition_metrics_by_step.get(step, [])
            if metrics:
                ratios = [m.token_repetition_ratio for m in metrics]
                fig.add_trace(go.Box(
                    y=ratios,
                    name=str(step),
                    marker_color='#2196F3',
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Training Step",
            yaxis_title="Token Diversity Ratio",
            template="plotly_white",
            showlegend=False,
        )

        # Add threshold line
        fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                     annotation_text="Suspicious threshold")

        return fig

    def create_dashboard(
        self,
        output_html: Optional[str] = None,
    ) -> "go.Figure":
        """Create a comprehensive dashboard with all visualizations.

        Returns a combined figure, optionally saving to HTML file.
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Score Trends', 'Diversity (Self-BLEU-4)',
                'Indicator Severity', 'Length Distribution',
                'GRPO Winner Concentration', 'Token Diversity',
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "box"}],
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
        )

        # Score trends (row 1, col 1)
        if self.detector.step_stats:
            steps, means, stds = zip(*self.detector.step_stats)
            fig.add_trace(
                go.Scatter(x=list(steps), y=list(means),
                          mode='lines+markers', name='Mean Score',
                          line=dict(color='#2196F3')),
                row=1, col=1
            )

        # Diversity - Self-BLEU (row 1, col 2)
        if self.detector.diversity_metrics_by_step:
            div_steps = sorted(self.detector.diversity_metrics_by_step.keys())
            bleu_vals = [self.detector.diversity_metrics_by_step[s].self_bleu_4 for s in div_steps]
            fig.add_trace(
                go.Scatter(x=div_steps, y=bleu_vals,
                          mode='lines+markers', name='Self-BLEU-4',
                          line=dict(color='#F44336')),
                row=1, col=2
            )
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange", row=1, col=2)

        # Severity timeline (row 2, col 1)
        severity_colors = {'low': '#4CAF50', 'medium': '#FF9800', 'high': '#F44336'}
        for severity in ['low', 'medium', 'high']:
            inds = [i for i in self.detector.indicators if i.severity == severity]
            if inds:
                fig.add_trace(
                    go.Scatter(
                        x=[i.step for i in inds],
                        y=[i.indicator_type for i in inds],
                        mode='markers', name=severity,
                        marker=dict(color=severity_colors[severity], size=6),
                    ),
                    row=2, col=1
                )

        # Length near-max ratio (row 2, col 2)
        if self.detector.length_metrics_by_step:
            len_steps = sorted(self.detector.length_metrics_by_step.keys())
            near_max = [self.detector.length_metrics_by_step[s].near_max_ratio for s in len_steps]
            fig.add_trace(
                go.Scatter(x=len_steps, y=near_max,
                          mode='lines+markers', name='Near-Max Ratio',
                          line=dict(color='#FF9800')),
                row=2, col=2
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=2)

        # GRPO winner concentration (row 3, col 1)
        if self.detector.grpo_metrics_by_step:
            grpo_steps = sorted(self.detector.grpo_metrics_by_step.keys())
            winner_conc = [self.detector.grpo_metrics_by_step[s].get('winner_concentration', 0) for s in grpo_steps]
            fig.add_trace(
                go.Scatter(x=grpo_steps, y=winner_conc,
                          mode='lines+markers', name='Winner Conc.',
                          line=dict(color='#9C27B0')),
                row=3, col=1
            )

        # Token diversity box plots (row 3, col 2)
        if self.detector.repetition_metrics_by_step:
            rep_steps = sorted(self.detector.repetition_metrics_by_step.keys())
            # Sample every N steps if too many
            sample_steps = rep_steps[::max(1, len(rep_steps) // 10)]
            for step in sample_steps:
                metrics = self.detector.repetition_metrics_by_step.get(step, [])
                if metrics:
                    ratios = [m.token_repetition_ratio for m in metrics]
                    fig.add_trace(
                        go.Box(y=ratios, name=str(step), marker_color='#2196F3',
                              showlegend=False),
                        row=3, col=2
                    )

        fig.update_layout(
            title="Reward Hacking Analysis Dashboard",
            template="plotly_white",
            height=900,
            showlegend=False,
        )

        if output_html:
            fig.write_html(output_html)
            print(f"Dashboard saved to {output_html}")

        return fig

    def show(self, fig: "go.Figure"):
        """Show a figure (opens in browser or notebook)."""
        fig.show()


# CLI Interface functions (defined separately for testability)
def _cli_analyze(rollout_dir, start_step=None, end_step=None, verbose=True, output=None):
    """Analyze rollout data for reward hacking indicators."""
    detector = RewardHackingDetector(rollout_dir)

    log_step(f"Loading rollouts from {rollout_dir}", verbose)
    log_step(f"Found {len(detector.steps)} steps: {detector.steps[0]} to {detector.steps[-1]}", verbose)

    results = detector.analyze(start_step=start_step, end_step=end_step, verbose=verbose)

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total indicators: {results['total_indicators']}")
    print(f"Steps with issues: {results['steps_with_issues']}")
    print("")

    print("By Type:")
    for t, count in results["by_type"].items():
        print(f"  {t}: {count}")

    print("\nBy Severity:")
    for s, count in results["by_severity"].items():
        print(f"  {s}: {count}")

    if results["hacking_origins"]:
        print("\nPotential Hacking Origins:")
        for origin in results["hacking_origins"][:5]:
            print(f"  Step {origin['first_step']}: '{origin['pattern'][:30]}...' ({origin['severity']})")

    if output:
        report = detector.generate_report(str(output))
        print(f"\nReport saved to {output}")

    return results


def _cli_diff(rollout_dir, step1, step2, match_by="index", verbose=True):
    """Diff rollouts between two steps."""
    detector = RewardHackingDetector(rollout_dir)

    log_step(f"Comparing step {step1} vs step {step2}...", verbose)
    result = detector.diff_steps(step1, step2, match_by=match_by, verbose=verbose)

    print(f"\nSamples compared: {result.samples_compared}")
    print("")

    if result.summary:
        print("Score Changes:")
        print(f"  Mean change: {result.summary['mean_score_change']:+.4f}")
        print(f"  Max increase: {result.summary['max_score_increase']:+.4f}")
        print(f"  Max decrease: {result.summary['max_score_decrease']:+.4f}")
        print(f"  Samples improved: {result.summary['samples_with_score_increase']}")
        print(f"  Samples degraded: {result.summary['samples_with_score_decrease']}")
        print(f"  New repetition cases: {result.summary['new_repetition_cases']}")

    if result.new_repetitions:
        print("\nNew Repetition Cases:")
        for rep in result.new_repetitions[:10]:
            print(f"  Sample {rep['sample_idx']}: "
                  f"token_ratio {rep['step1_metrics']['token_rep_ratio']:.2f} -> "
                  f"{rep['step2_metrics']['token_rep_ratio']:.2f}")
            if rep['step2_metrics'].get('repeated_phrase'):
                print(f"    Repeated: '{rep['step2_metrics']['repeated_phrase'][:40]}'")

    return result


def _cli_watch(rollout_dir, last_n=5, verbose=True):
    """Quick check of the most recent steps for hacking indicators."""
    watch_start = time.time()
    detector = RewardHackingDetector(rollout_dir)

    steps = detector.steps[-last_n:]
    log_step(f"Checking last {len(steps)} steps: {steps}", verbose)

    all_indicators = []
    progress = ProgressTracker(len(steps), "Scanning steps", disable=not verbose)

    for step in steps:
        step_start = time.time()
        indicators = detector.analyze_step(step)
        all_indicators.extend(indicators)
        step_time = time.time() - step_start

        high = sum(1 for i in indicators if i.severity == "high")
        med = sum(1 for i in indicators if i.severity == "medium")
        low = sum(1 for i in indicators if i.severity == "low")

        status = "OK" if not indicators else f"H:{high} M:{med} L:{low}"

        # Simple status line without rich dependency
        prefix = "[!]" if high > 0 else ("[?]" if indicators else "[ ]")

        progress.update(1, extra_info=f"step {step}: {status}")

    progress.finish()

    # Print summary
    print("\n" + "=" * 50)
    print("WATCH RESULTS")
    print("=" * 50)

    for step in steps:
        step_indicators = [i for i in all_indicators if i.step == step]
        high = sum(1 for i in step_indicators if i.severity == "high")
        med = sum(1 for i in step_indicators if i.severity == "medium")
        low = sum(1 for i in step_indicators if i.severity == "low")

        status = "OK" if not step_indicators else f"H:{high} M:{med} L:{low}"
        prefix = "[!]" if high > 0 else ("[?]" if step_indicators else "[ ]")
        print(f"{prefix} Step {step}: {status}")

        # Show high-severity details
        for ind in [i for i in step_indicators if i.severity == "high"][:3]:
            print(f"    -> {ind.description[:70]}")

    return all_indicators


def _cli_sample(rollout_dir, step, sample_idx):
    """Inspect a specific sample for repetition patterns."""
    detector = RewardHackingDetector(rollout_dir)
    samples = detector.loader.load_step(step)

    if sample_idx >= len(samples):
        print(f"Sample index {sample_idx} out of range (max: {len(samples) - 1})")
        return None

    sample = samples[sample_idx]
    metrics = detector.repetition_analyzer.analyze(sample.output)

    print(f"Step: {step}, Sample: {sample_idx}")
    print(f"Score: {sample.score}")
    if sample.request_id:
        print(f"Request ID: {sample.request_id}")
    print("")

    print("INPUT:")
    print("-" * 40)
    print(sample.input)
    print("")

    print("OUTPUT:")
    print("-" * 40)
    print(sample.output)
    print("")

    print("REPETITION ANALYSIS:")
    print("-" * 40)
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Unique tokens: {metrics.unique_tokens}")
    print(f"Token diversity: {metrics.token_repetition_ratio:.1%}")
    print(f"Max consecutive repeats: {metrics.max_consecutive_repeats}")

    if metrics.repeated_phrase:
        print(f"Most repeated phrase ({metrics.repeated_phrase_count}x): '{metrics.repeated_phrase}'")

    print("\nN-gram repetition ratios:")
    for n, ratio in sorted(metrics.ngram_repetition_scores.items()):
        bar = "#" * int(ratio * 20)
        print(f"  {n}-gram: {ratio:.1%} {bar}")

    print(f"\nSuspicious: {'YES' if metrics.is_suspicious else 'No'}")

    return sample, metrics


def _cli_visualize(
    rollout_dir,
    output_html="dashboard.html",
    plot_type="dashboard",
    start_step=None,
    end_step=None,
    verbose=True,
    show=False,
):
    """Generate Plotly visualizations of analysis results."""
    if not HAS_PLOTLY:
        print("Error: Plotly is required for visualization.")
        print("Install with: pip install plotly")
        return None

    # Run analysis first
    detector = RewardHackingDetector(rollout_dir)
    log_step(f"Loading and analyzing rollouts from {rollout_dir}", verbose)
    detector.analyze(start_step=start_step, end_step=end_step, verbose=verbose)

    # Create visualizer
    visualizer = HackingVisualizer(detector)

    plot_funcs = {
        "dashboard": lambda: visualizer.create_dashboard(output_html),
        "scores": visualizer.plot_score_trends,
        "diversity": visualizer.plot_diversity_metrics,
        "heatmap": visualizer.plot_indicator_heatmap,
        "timeline": visualizer.plot_severity_timeline,
        "length": visualizer.plot_length_distribution,
        "grpo": visualizer.plot_grpo_metrics,
        "compression": visualizer.plot_compression_ratio,
    }

    if plot_type not in plot_funcs:
        print(f"Unknown plot type: {plot_type}")
        print(f"Available: {', '.join(plot_funcs.keys())}")
        return None

    log_step(f"Generating {plot_type} visualization...", verbose)

    if plot_type == "dashboard":
        fig = visualizer.create_dashboard(output_html)
        print(f"Dashboard saved to {output_html}")
    else:
        fig = plot_funcs[plot_type]()
        if output_html:
            fig.write_html(output_html)
            print(f"Plot saved to {output_html}")

    if show:
        fig.show()

    return fig


# Typer CLI wrapper (only if typer is installed)
def _build_typer_app():
    """Build the typer CLI app."""
    if not HAS_TYPER:
        return None

    app = typer.Typer(help="Reward Hacking Detection Tool for VERL")

    @app.command()
    def analyze(
        rollout_dir: Path = typer.Argument(..., help="Path to rollout data directory"),
        start_step: Optional[int] = typer.Option(None, "--start", help="Start step (inclusive)"),
        end_step: Optional[int] = typer.Option(None, "--end", help="End step (inclusive)"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to file"),
    ):
        """Analyze rollout data for reward hacking indicators."""
        _cli_analyze(rollout_dir, start_step, end_step, verbose, output)

    @app.command()
    def diff(
        rollout_dir: Path = typer.Argument(..., help="Path to rollout data directory"),
        step1: int = typer.Option(..., "--step1", "-s1", help="First step to compare"),
        step2: int = typer.Option(..., "--step2", "-s2", help="Second step to compare"),
        match_by: str = typer.Option("index", "--match-by", help="Match by 'index' or 'request_id'"),
    ):
        """Diff rollouts between two steps."""
        _cli_diff(rollout_dir, step1, step2, match_by)

    @app.command()
    def watch(
        rollout_dir: Path = typer.Argument(..., help="Path to rollout data directory"),
        last_n: int = typer.Option(5, "--last", "-n", help="Analyze last N steps"),
    ):
        """Quick check of the most recent steps for hacking indicators."""
        _cli_watch(rollout_dir, last_n)

    @app.command()
    def sample(
        rollout_dir: Path = typer.Argument(..., help="Path to rollout data directory"),
        step: int = typer.Option(..., "--step", "-s", help="Step to inspect"),
        sample_idx: int = typer.Option(..., "--sample", "-i", help="Sample index"),
    ):
        """Inspect a specific sample for repetition patterns."""
        result = _cli_sample(rollout_dir, step, sample_idx)
        if result is None:
            raise typer.Exit(1)

    @app.command()
    def visualize(
        rollout_dir: Path = typer.Argument(..., help="Path to rollout data directory"),
        output: Path = typer.Option(
            "dashboard.html", "--output", "-o", help="Output HTML file"
        ),
        plot_type: str = typer.Option(
            "dashboard", "--type", "-t",
            help="Plot type: dashboard, scores, diversity, heatmap, timeline, length, grpo, compression"
        ),
        start_step: Optional[int] = typer.Option(None, "--start", help="Start step"),
        end_step: Optional[int] = typer.Option(None, "--end", help="End step"),
        verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
        show: bool = typer.Option(False, "--show", "-s", help="Open in browser after saving"),
    ):
        """Generate Plotly visualizations of reward hacking analysis."""
        result = _cli_visualize(
            rollout_dir,
            output_html=str(output),
            plot_type=plot_type,
            start_step=start_step,
            end_step=end_step,
            verbose=verbose,
            show=show,
        )
        if result is None:
            raise typer.Exit(1)

    return app


app = _build_typer_app()


def main():
    """Main entry point."""
    if app is None:
        print("Error: typer is required for CLI. Install with: pip install typer")
        print("")
        print("You can still use the Python API:")
        print("  from scripts.reward_hacking_detector import RewardHackingDetector")
        print("  detector = RewardHackingDetector('/path/to/rollouts')")
        print("  results = detector.analyze()")
        return

    app()


if __name__ == "__main__":
    main()
