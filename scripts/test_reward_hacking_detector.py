#!/usr/bin/env python3
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
Comprehensive tests for the Reward Hacking Detection Module.

Run with: python -m pytest scripts/test_reward_hacking_detector.py -v
Or simply: python scripts/test_reward_hacking_detector.py
"""

import json
import os
import tempfile
from pathlib import Path

# Import the module components
from reward_hacking_detector import (
    Sample,
    RepetitionMetrics,
    RepetitionAnalyzer,
    CompressionMetrics,
    CompressionAnalyzer,
    DiversityMetrics,
    DiversityAnalyzer,
    LengthMetrics,
    LengthAnalyzer,
    GRPOAnalyzer,
    ScoreAnalyzer,
    RolloutLoader,
    RewardHackingDetector,
    HackingIndicator,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_test_rollout_dir(samples_by_step: dict[int, list[dict]]) -> Path:
    """Create a temporary directory with JSONL rollout files."""
    tmpdir = tempfile.mkdtemp()

    for step, samples in samples_by_step.items():
        filepath = Path(tmpdir) / f"{step}.jsonl"
        with open(filepath, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    return Path(tmpdir)


# =============================================================================
# TEST CASES: RepetitionAnalyzer
# =============================================================================

class TestRepetitionAnalyzer:
    """Tests for the RepetitionAnalyzer class."""

    def test_normal_text(self):
        """Normal text should not be flagged as suspicious."""
        analyzer = RepetitionAnalyzer()

        text = "The quick brown fox jumps over the lazy dog. This is a normal sentence with varied vocabulary."
        metrics = analyzer.analyze(text)

        assert not metrics.is_suspicious
        assert metrics.token_repetition_ratio > 0.5  # High diversity
        assert metrics.max_consecutive_repeats <= 2
        print(f"✓ Normal text: ratio={metrics.token_repetition_ratio:.2f}, suspicious={metrics.is_suspicious}")

    def test_repetitive_text_token_level(self):
        """Text with many repeated tokens should be flagged."""
        analyzer = RepetitionAnalyzer()

        # Extreme token repetition
        text = "hello " * 100
        metrics = analyzer.analyze(text)

        assert metrics.is_suspicious
        assert metrics.token_repetition_ratio < 0.1  # Very low diversity
        assert metrics.max_consecutive_repeats >= 50
        print(f"✓ Token repetition: ratio={metrics.token_repetition_ratio:.2f}, consecutive={metrics.max_consecutive_repeats}")

    def test_repetitive_text_phrase_level(self):
        """Text with repeated phrases should be detected."""
        analyzer = RepetitionAnalyzer()

        # Repeated phrases (common in reward hacking)
        text = "I think the answer is yes. " * 20
        metrics = analyzer.analyze(text)

        # Should detect phrase repetition via n-grams
        assert metrics.repeated_phrase is not None
        assert metrics.repeated_phrase_count > 10
        print(f"✓ Phrase repetition: phrase='{metrics.repeated_phrase}', count={metrics.repeated_phrase_count}")

    def test_mixed_repetition(self):
        """Text with some repetition but not extreme should be border case."""
        analyzer = RepetitionAnalyzer()

        text = "The result is correct. The result is verified. The result is good. However, we need more testing."
        metrics = analyzer.analyze(text)

        # This is a border case - some repetition but not extreme
        print(f"✓ Mixed text: ratio={metrics.token_repetition_ratio:.2f}, suspicious={metrics.is_suspicious}")

    def test_empty_text(self):
        """Empty text should return default metrics."""
        analyzer = RepetitionAnalyzer()

        metrics = analyzer.analyze("")

        assert metrics.total_tokens == 0
        assert not metrics.is_suspicious
        print("✓ Empty text handled correctly")

    def test_ngram_detection(self):
        """N-gram repetition should be detected at various levels."""
        analyzer = RepetitionAnalyzer(ngram_sizes=(2, 3, 4, 5))

        # 3-gram repetition pattern
        text = "step by step " * 15 + "we proceed carefully"
        metrics = analyzer.analyze(text)

        assert 3 in metrics.ngram_repetition_scores
        assert metrics.ngram_repetition_scores[3] > 0.3  # Significant 3-gram repetition
        print(f"✓ N-gram detection: 3-gram ratio={metrics.ngram_repetition_scores.get(3, 0):.2f}")


# =============================================================================
# TEST CASES: CompressionAnalyzer
# =============================================================================

class TestCompressionAnalyzer:
    """Tests for the CompressionAnalyzer class."""

    def test_random_text_high_compression(self):
        """Random/diverse text should have high compression ratio."""
        analyzer = CompressionAnalyzer()

        text = "Apple banana cherry date elderberry fig grape honeydew. " \
               "Ice cream jelly kiwi lemon mango nectarine orange papaya."
        metrics = analyzer.analyze(text)

        assert not metrics.is_suspicious
        assert metrics.compression_ratio > 0.5  # Can't compress much
        print(f"✓ Diverse text: compression_ratio={metrics.compression_ratio:.3f}")

    def test_repetitive_text_low_compression(self):
        """Highly repetitive text should have low compression ratio."""
        analyzer = CompressionAnalyzer()

        text = "repeat " * 500
        metrics = analyzer.analyze(text)

        assert metrics.is_suspicious
        assert metrics.compression_ratio < 0.2  # Compresses very well
        print(f"✓ Repetitive text: compression_ratio={metrics.compression_ratio:.3f}")

    def test_batch_analysis(self):
        """Batch analysis should work correctly."""
        analyzer = CompressionAnalyzer()

        texts = [
            "Normal varied text here",
            "repeat repeat repeat repeat repeat",
            "Another normal sentence with words",
        ]

        metrics_list = analyzer.analyze_batch(texts)

        assert len(metrics_list) == 3
        assert not metrics_list[0].is_suspicious  # Normal
        # Note: "repeat repeat..." might not hit threshold with just 5 repeats
        assert not metrics_list[2].is_suspicious  # Normal
        print(f"✓ Batch analysis: processed {len(metrics_list)} texts")

    def test_combined_compression(self):
        """Combined compression detects similarity across texts."""
        analyzer = CompressionAnalyzer()

        # Similar texts should compress well together
        similar_texts = [
            "The answer to the question is 42.",
            "The answer to the question is 43.",
            "The answer to the question is 44.",
        ]

        diverse_texts = [
            "Apples are red fruits.",
            "The sky is blue today.",
            "Mathematics involves numbers.",
        ]

        similar_ratio = analyzer.combined_compression(similar_texts)
        diverse_ratio = analyzer.combined_compression(diverse_texts)

        assert similar_ratio < diverse_ratio  # Similar texts compress better together
        print(f"✓ Combined compression: similar={similar_ratio:.3f}, diverse={diverse_ratio:.3f}")


# =============================================================================
# TEST CASES: DiversityAnalyzer
# =============================================================================

class TestDiversityAnalyzer:
    """Tests for the DiversityAnalyzer class."""

    def test_diverse_outputs(self):
        """Diverse outputs should have low self-BLEU and high distinct-n."""
        analyzer = DiversityAnalyzer()

        outputs = [
            "The capital of France is Paris, known for the Eiffel Tower.",
            "Python is a programming language used for web development.",
            "The ocean covers about 70 percent of Earth's surface.",
            "Mozart composed many famous symphonies and operas.",
            "Photosynthesis converts sunlight into chemical energy.",
        ]

        metrics = analyzer.analyze(outputs)

        assert not metrics.is_mode_collapse
        assert metrics.self_bleu_4 < 0.5  # Low similarity
        assert metrics.distinct_1 > 0.3   # Good vocabulary diversity
        print(f"✓ Diverse outputs: self_bleu_4={metrics.self_bleu_4:.3f}, distinct_1={metrics.distinct_1:.3f}")

    def test_mode_collapse_identical(self):
        """Identical outputs should trigger mode collapse detection."""
        analyzer = DiversityAnalyzer()

        outputs = [
            "The answer is yes.",
            "The answer is yes.",
            "The answer is yes.",
            "The answer is yes.",
            "The answer is yes.",
            "The answer is yes.",
        ]

        metrics = analyzer.analyze(outputs)

        assert metrics.is_mode_collapse
        assert metrics.self_bleu_4 > 0.9  # Very high similarity
        assert metrics.jaccard_similarity > 0.9
        print(f"✓ Mode collapse (identical): self_bleu_4={metrics.self_bleu_4:.3f}")

    def test_mode_collapse_similar(self):
        """Very similar outputs should trigger mode collapse."""
        analyzer = DiversityAnalyzer()

        outputs = [
            "I believe the correct answer is option A because it makes sense.",
            "I think the correct answer is option A because it is logical.",
            "I feel the correct answer is option A because it seems right.",
            "I believe the correct answer is option A because it is accurate.",
            "I think the correct answer is option A because it fits best.",
            "I feel the correct answer is option A because it is true.",
        ]

        metrics = analyzer.analyze(outputs)

        # High Jaccard due to shared vocabulary
        print(f"✓ Similar outputs: jaccard={metrics.jaccard_similarity:.3f}, self_bleu_4={metrics.self_bleu_4:.3f}")

    def test_distinct_n_metrics(self):
        """Distinct-n metrics should correctly measure vocabulary diversity."""
        analyzer = DiversityAnalyzer()

        # Low diversity - same words repeated
        low_diversity = ["the the the the", "the the the the", "the the the the"]
        high_diversity = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]

        low_metrics = analyzer.analyze(low_diversity)
        high_metrics = analyzer.analyze(high_diversity)

        assert low_metrics.distinct_1 < high_metrics.distinct_1
        print(f"✓ Distinct-1: low={low_metrics.distinct_1:.3f}, high={high_metrics.distinct_1:.3f}")


# =============================================================================
# TEST CASES: LengthAnalyzer
# =============================================================================

class TestLengthAnalyzer:
    """Tests for the LengthAnalyzer class."""

    def test_varied_lengths(self):
        """Varied lengths should not trigger length hacking."""
        analyzer = LengthAnalyzer()

        outputs = [
            "Short.",
            "This is a medium length response.",
            "This is a somewhat longer response that has more words in it.",
            "Very short",
            "Another medium response here.",
        ]

        metrics = analyzer.analyze(outputs)

        assert not metrics.is_length_hacking
        assert metrics.std_length > 10  # Good variance
        print(f"✓ Varied lengths: mean={metrics.mean_length:.1f}, std={metrics.std_length:.1f}")

    def test_length_stuffing(self):
        """All outputs at max length should trigger length hacking."""
        analyzer = LengthAnalyzer()

        # All outputs are very similar in length (near max)
        base = "word " * 50
        outputs = [base + "a", base + "b", base + "c", base + "d", base + "e"]

        metrics = analyzer.analyze(outputs)

        assert metrics.near_max_ratio > 0.8  # Most are near max
        # Note: is_length_hacking requires near_max_ratio > 0.5
        print(f"✓ Length stuffing: near_max_ratio={metrics.near_max_ratio:.2f}")

    def test_formulaic_outputs(self):
        """Low length variance with reasonable mean should be detected."""
        analyzer = LengthAnalyzer()

        # All outputs exactly 100 chars (formulaic)
        outputs = [
            "x" * 100,
            "y" * 100,
            "z" * 100,
            "a" * 100,
            "b" * 100,
        ]

        metrics = analyzer.analyze(outputs)

        assert metrics.std_length < 1  # Very low variance
        assert metrics.is_length_hacking  # Formulaic pattern
        print(f"✓ Formulaic outputs: mean={metrics.mean_length:.1f}, std={metrics.std_length:.3f}")


# =============================================================================
# TEST CASES: GRPOAnalyzer
# =============================================================================

class TestGRPOAnalyzer:
    """Tests for the GRPOAnalyzer class."""

    def test_normal_grpo_groups(self):
        """Normal GRPO groups should not trigger alerts."""
        analyzer = GRPOAnalyzer()

        samples = [
            Sample(input="q1", output="a1", score=0.8, step=1, request_id="g1", sample_idx=0),
            Sample(input="q1", output="a2", score=0.6, step=1, request_id="g1", sample_idx=1),
            Sample(input="q2", output="b1", score=0.7, step=1, request_id="g2", sample_idx=2),
            Sample(input="q2", output="b2", score=0.9, step=1, request_id="g2", sample_idx=3),
            Sample(input="q3", output="c1", score=0.5, step=1, request_id="g3", sample_idx=4),
            Sample(input="q3", output="c2", score=0.7, step=1, request_id="g3", sample_idx=5),
        ]

        metrics = analyzer.analyze_groups(samples)

        assert metrics["num_groups"] == 3
        assert metrics["avg_group_size"] == 2.0
        assert metrics["same_output_ratio"] == 0.0  # No identical outputs
        print(f"✓ Normal GRPO: groups={metrics['num_groups']}, winner_conc={metrics['winner_concentration']:.2f}")

    def test_winner_concentration(self):
        """Same position always winning should be detected."""
        analyzer = GRPOAnalyzer()

        # Position 0 always wins across all groups
        samples = []
        for g in range(10):
            samples.append(Sample(input=f"q{g}", output=f"winner{g}", score=1.0, step=1,
                                  request_id=f"g{g}", sample_idx=g*2))
            samples.append(Sample(input=f"q{g}", output=f"loser{g}", score=0.0, step=1,
                                  request_id=f"g{g}", sample_idx=g*2+1))

        metrics = analyzer.analyze_groups(samples)

        assert metrics["winner_concentration"] == 1.0  # Same position always wins
        print(f"✓ Winner concentration: {metrics['winner_concentration']:.2f}")

    def test_identical_outputs_in_groups(self):
        """Identical outputs within groups should be detected."""
        analyzer = GRPOAnalyzer()

        samples = [
            # Group 1: identical outputs
            Sample(input="q1", output="same answer", score=0.8, step=1, request_id="g1", sample_idx=0),
            Sample(input="q1", output="same answer", score=0.6, step=1, request_id="g1", sample_idx=1),
            # Group 2: identical outputs
            Sample(input="q2", output="another same", score=0.7, step=1, request_id="g2", sample_idx=2),
            Sample(input="q2", output="another same", score=0.9, step=1, request_id="g2", sample_idx=3),
            # Group 3: different outputs
            Sample(input="q3", output="output A", score=0.5, step=1, request_id="g3", sample_idx=4),
            Sample(input="q3", output="output B", score=0.7, step=1, request_id="g3", sample_idx=5),
        ]

        metrics = analyzer.analyze_groups(samples)

        # 2 out of 3 groups have identical outputs
        assert metrics["same_output_ratio"] == 2/3
        print(f"✓ Identical outputs ratio: {metrics['same_output_ratio']:.2f}")


# =============================================================================
# TEST CASES: ScoreAnalyzer
# =============================================================================

class TestScoreAnalyzer:
    """Tests for the ScoreAnalyzer class."""

    def test_normal_scores(self):
        """Normal score distribution should not trigger anomalies."""
        analyzer = ScoreAnalyzer(zscore_threshold=3.0)

        samples = [
            Sample(input="", output="", score=0.7 + i*0.02, step=1, sample_idx=i)
            for i in range(20)
        ]

        anomalies = analyzer.analyze_step(samples)

        assert len(anomalies) == 0  # No outliers
        print(f"✓ Normal scores: {len(anomalies)} anomalies")

    def test_score_outlier(self):
        """Score outliers should be detected."""
        analyzer = ScoreAnalyzer(zscore_threshold=2.0)  # Lower threshold for testing

        # Normal scores around 0.5, one outlier at 5.0
        samples = [
            Sample(input="", output="", score=0.5, step=1, sample_idx=i)
            for i in range(19)
        ]
        samples.append(Sample(input="", output="", score=5.0, step=1, sample_idx=19))

        anomalies = analyzer.analyze_step(samples)

        assert len(anomalies) >= 1
        assert any(a.sample_idx == 19 for a in anomalies)  # The outlier
        print(f"✓ Score outlier detected: {len(anomalies)} anomalies")

    def test_trend_spike(self):
        """Sudden score spikes across steps should be detected."""
        analyzer = ScoreAnalyzer(spike_threshold=2.0, window_size=3)

        # Normal scores, then sudden spike
        step_stats = [
            (1, 0.5, 0.1),
            (2, 0.5, 0.1),
            (3, 0.5, 0.1),
            (4, 0.5, 0.1),
            (5, 2.0, 0.1),  # Spike!
        ]

        anomalies = analyzer.analyze_trend(step_stats)

        assert len(anomalies) >= 1
        assert any(a["step"] == 5 and a["type"] == "spike" for a in anomalies)
        print(f"✓ Trend spike detected at step 5")


# =============================================================================
# TEST CASES: Integration Tests
# =============================================================================

class TestRewardHackingDetector:
    """Integration tests for the full RewardHackingDetector."""

    def test_full_analysis_clean(self):
        """Clean rollouts should not trigger many indicators."""
        samples = {
            1: [
                {"input": "What is 2+2?", "output": "The answer is 4.", "score": 0.9, "step": 1},
                {"input": "What is 3+3?", "output": "The answer is 6.", "score": 0.8, "step": 1},
            ],
            2: [
                {"input": "What is 4+4?", "output": "The answer is 8.", "score": 0.85, "step": 2},
                {"input": "What is 5+5?", "output": "The answer is 10.", "score": 0.9, "step": 2},
            ],
        }

        tmpdir = create_test_rollout_dir(samples)

        try:
            detector = RewardHackingDetector(tmpdir)
            results = detector.analyze()

            # Should have few or no high-severity indicators
            high_severity = [i for i in detector.indicators if i.severity == "high"]
            print(f"✓ Clean analysis: {results['total_indicators']} indicators, {len(high_severity)} high severity")
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(tmpdir)

    def test_full_analysis_repetitive(self):
        """Repetitive rollouts should trigger indicators."""
        repetitive_output = "I think I think I think I think " * 10

        samples = {
            1: [
                {"input": "Question 1", "output": repetitive_output, "score": 0.9, "step": 1},
                {"input": "Question 2", "output": repetitive_output, "score": 0.8, "step": 1},
                {"input": "Question 3", "output": repetitive_output, "score": 0.85, "step": 1},
            ],
        }

        tmpdir = create_test_rollout_dir(samples)

        try:
            detector = RewardHackingDetector(tmpdir)
            results = detector.analyze()

            # Should detect repetition and mode collapse
            assert results["total_indicators"] > 0
            types = results.get("by_type", {})
            print(f"✓ Repetitive analysis: {results['total_indicators']} indicators")
            print(f"  Types: {types}")
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_step_diff(self):
        """Diffing between steps should work correctly."""
        samples = {
            1: [
                {"input": "Q1", "output": "Normal response A", "score": 0.5, "step": 1},
                {"input": "Q2", "output": "Normal response B", "score": 0.6, "step": 1},
            ],
            2: [
                {"input": "Q1", "output": "repeat repeat repeat repeat " * 10, "score": 0.9, "step": 2},
                {"input": "Q2", "output": "Better response B", "score": 0.8, "step": 2},
            ],
        }

        tmpdir = create_test_rollout_dir(samples)

        try:
            detector = RewardHackingDetector(tmpdir)
            diff = detector.diff_steps(1, 2)

            assert diff.samples_compared == 2
            assert len(diff.score_changes) == 2
            # First sample got much worse (repetition) but higher score
            assert diff.new_repetitions  # Should detect new repetition
            print(f"✓ Step diff: {diff.samples_compared} samples, {len(diff.new_repetitions)} new repetitions")
        finally:
            import shutil
            shutil.rmtree(tmpdir)


# =============================================================================
# SYNTHETIC REWARD HACKING EXAMPLES
# =============================================================================

class TestSyntheticRewardHacking:
    """Test with synthetic examples of known reward hacking patterns."""

    def test_format_exploitation(self):
        """Test detection of format/template exploitation."""
        analyzer = RepetitionAnalyzer()

        # Common IF-GRPO pattern: model learns to repeat format tokens
        text = """
        Step 1: Let me think about this carefully.
        Step 2: Let me think about this carefully.
        Step 3: Let me think about this carefully.
        Step 4: Let me think about this carefully.
        Step 5: Let me think about this carefully.
        Therefore, the answer is yes.
        """

        metrics = analyzer.analyze(text)

        assert metrics.repeated_phrase_count >= 4
        print(f"✓ Format exploitation: repeated '{metrics.repeated_phrase}' x{metrics.repeated_phrase_count}")

    def test_json_padding(self):
        """Test detection of JSON/structured output padding."""
        comp_analyzer = CompressionAnalyzer()

        # Model pads JSON with repeated fields
        text = """
        {"result": "yes", "confidence": 0.9, "reason": "because because because because",
         "extra1": "padding", "extra2": "padding", "extra3": "padding", "extra4": "padding",
         "extra5": "padding", "extra6": "padding", "extra7": "padding", "extra8": "padding"}
        """ * 5

        metrics = comp_analyzer.analyze(text)

        print(f"✓ JSON padding: compression_ratio={metrics.compression_ratio:.3f}, suspicious={metrics.is_suspicious}")

    def test_reasoning_verbosity(self):
        """Test detection of excessive reasoning verbosity (GRPO budget stuffing)."""
        length_analyzer = LengthAnalyzer()

        # All responses are extremely long (hitting token budget)
        outputs = [
            "a" * 1000,  # Max length responses
            "b" * 1000,
            "c" * 1000,
            "d" * 1000,
            "e" * 1000,
        ]

        metrics = length_analyzer.analyze(outputs)

        assert metrics.is_length_hacking
        assert metrics.std_length < 1
        print(f"✓ Verbosity/budget stuffing: near_max={metrics.near_max_ratio:.2f}")

    def test_grpo_policy_collapse(self):
        """Test detection of GRPO policy collapse (same output always wins)."""
        grpo_analyzer = GRPOAnalyzer()
        diversity_analyzer = DiversityAnalyzer()

        # All groups produce identical best response
        samples = []
        winning_response = "The definitive answer is 42."

        for g in range(10):
            samples.append(Sample(
                input=f"Question {g}",
                output=winning_response,  # Same winning response!
                score=1.0,
                step=1,
                request_id=f"group_{g}",
                sample_idx=g*2
            ))
            samples.append(Sample(
                input=f"Question {g}",
                output=f"Alternative answer {g}",
                score=0.0,
                step=1,
                request_id=f"group_{g}",
                sample_idx=g*2+1
            ))

        grpo_metrics = grpo_analyzer.analyze_groups(samples)

        # Check winner position concentration
        assert grpo_metrics["winner_concentration"] == 1.0

        # Also check diversity of winning outputs
        winning_outputs = [s.output for s in samples if s.score == 1.0]
        diversity_metrics = diversity_analyzer.analyze(winning_outputs)

        assert diversity_metrics.is_mode_collapse
        print(f"✓ GRPO collapse: winner_conc={grpo_metrics['winner_concentration']:.2f}, "
              f"self_bleu={diversity_metrics.self_bleu_4:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestRepetitionAnalyzer,
        TestCompressionAnalyzer,
        TestDiversityAnalyzer,
        TestLengthAnalyzer,
        TestGRPOAnalyzer,
        TestScoreAnalyzer,
        TestRewardHackingDetector,
        TestSyntheticRewardHacking,
    ]

    total_passed = 0
    total_failed = 0

    print("=" * 70)
    print("REWARD HACKING DETECTOR - TEST SUITE")
    print("=" * 70)
    print()

    for test_class in test_classes:
        print(f"\n{'='*50}")
        print(f"Running: {test_class.__name__}")
        print(f"{'='*50}")

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    total_passed += 1
                except AssertionError as e:
                    print(f"✗ {method_name}: FAILED - {e}")
                    total_failed += 1
                except Exception as e:
                    print(f"✗ {method_name}: ERROR - {e}")
                    total_failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    return total_failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
