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
    python reward_hacking_detector.py report /path/to/rollout_data_dir --output report.html

    # Python API
    from scripts.reward_hacking_detector import RewardHackingDetector
    detector = RewardHackingDetector("/path/to/rollout_data_dir")
    results = detector.analyze()
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

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


class RewardHackingDetector:
    """Main class for detecting reward hacking in rollout data."""

    def __init__(
        self,
        rollout_dir: str | Path,
        repetition_analyzer: Optional[RepetitionAnalyzer] = None,
        score_analyzer: Optional[ScoreAnalyzer] = None,
    ):
        self.loader = RolloutLoader(rollout_dir)
        self.repetition_analyzer = repetition_analyzer or RepetitionAnalyzer()
        self.score_analyzer = score_analyzer or ScoreAnalyzer()

        # Results storage
        self.indicators: list[HackingIndicator] = []
        self.step_stats: list[tuple[int, float, float]] = []  # (step, mean, std)
        self.repetition_metrics_by_step: dict[int, list[RepetitionMetrics]] = {}

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
        """Analyze all steps for reward hacking indicators."""
        self.indicators = []
        self.step_stats = []
        self.repetition_metrics_by_step = {}

        steps_to_analyze = [
            s for s in self.steps
            if (start_step is None or s >= start_step) and (end_step is None or s <= end_step)
        ]

        if verbose:
            print(f"Analyzing {len(steps_to_analyze)} steps...")

        for step in steps_to_analyze:
            self.analyze_step(step, verbose=verbose)

        # Analyze cross-step trends
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

        return self.get_summary()

    def diff_steps(self, step1: int, step2: int, match_by: str = "index") -> DiffResult:
        """Compare rollouts between two steps to identify changes.

        Args:
            step1: First step to compare
            step2: Second step to compare
            match_by: How to match samples - "index" (positional) or "request_id"
        """
        samples1 = self.loader.load_step(step1)
        samples2 = self.loader.load_step(step2)

        result = DiffResult(step1=step1, step2=step2, samples_compared=0)

        if match_by == "request_id":
            # Match by request_id
            map1 = {s.request_id: s for s in samples1 if s.request_id}
            map2 = {s.request_id: s for s in samples2 if s.request_id}
            common_ids = set(map1.keys()) & set(map2.keys())

            for rid in common_ids:
                s1, s2 = map1[rid], map2[rid]
                self._compare_samples(s1, s2, result)

            result.samples_compared = len(common_ids)
        else:
            # Match by index position
            for i in range(min(len(samples1), len(samples2))):
                s1, s2 = samples1[i], samples2[i]
                self._compare_samples(s1, s2, result)

            result.samples_compared = min(len(samples1), len(samples2))

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


# CLI Interface functions (defined separately for testability)
def _cli_analyze(rollout_dir, start_step=None, end_step=None, verbose=False, output=None):
    """Analyze rollout data for reward hacking indicators."""
    detector = RewardHackingDetector(rollout_dir)

    print(f"Loading rollouts from {rollout_dir}")
    print(f"Found {len(detector.steps)} steps: {detector.steps[0]} to {detector.steps[-1]}")
    print("")

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


def _cli_diff(rollout_dir, step1, step2, match_by="index"):
    """Diff rollouts between two steps."""
    detector = RewardHackingDetector(rollout_dir)

    print(f"Comparing step {step1} vs step {step2}...")
    result = detector.diff_steps(step1, step2, match_by=match_by)

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


def _cli_watch(rollout_dir, last_n=5):
    """Quick check of the most recent steps for hacking indicators."""
    detector = RewardHackingDetector(rollout_dir)

    steps = detector.steps[-last_n:]
    print(f"Checking last {len(steps)} steps: {steps}")
    print("")

    all_indicators = []
    for step in steps:
        indicators = detector.analyze_step(step)
        all_indicators.extend(indicators)
        high = sum(1 for i in indicators if i.severity == "high")
        med = sum(1 for i in indicators if i.severity == "medium")
        low = sum(1 for i in indicators if i.severity == "low")

        status = "OK" if not indicators else f"H:{high} M:{med} L:{low}"

        # Simple status line without rich dependency
        prefix = "[!]" if high > 0 else ("[?]" if indicators else "[ ]")
        print(f"{prefix} Step {step}: {status}")

        # Show high-severity details
        for ind in [i for i in indicators if i.severity == "high"][:3]:
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
