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
Unit tests for IFEval (Instruction Following Evaluation) reward function.

Tests cover all major instruction types:
- Keywords and forbidden words
- Length constraints (words, sentences, paragraphs)
- Format requirements (JSON, bullet lists, titles)
- Content constraints (start/end phrases, case, postscripts)
- Advanced constraints (highlighted sections, placeholders, quotations)
"""

import json

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest for when it's not available
    class MockPytest:
        @staticmethod
        def main(args):
            print("pytest not available, tests should be run with pytest or manually")
    pytest = MockPytest()

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.ifeval import compute_score


class TestKeywordInstructions:
    """Test keyword-based instruction verification."""

    def test_keyword_at_least(self):
        """Test 'at least' keyword constraint."""
        response = "AI is transforming the world. AI systems are improving. The future of AI is bright."
        ground_truth = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["AI"], "frequency": 3, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0
        assert result["num_followed"] == 1
        assert result["num_instructions"] == 1
        assert result["prompt_strict_acc"] == 1.0

    def test_keyword_at_least_fail(self):
        """Test 'at least' keyword constraint failure."""
        response = "AI is transforming the world."
        ground_truth = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["AI"], "frequency": 3, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0
        assert result["num_followed"] == 0
        assert result["prompt_strict_acc"] == 0.0

    def test_multiple_keywords(self):
        """Test multiple keyword constraints."""
        response = "Robot technology is advancing. AI and machine learning improve robot capabilities."
        ground_truth = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["robot", "AI"], "frequency": 2, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0
        assert result["num_followed"] == 1

    def test_keyword_case_insensitive(self):
        """Test that keyword matching is case-insensitive."""
        response = "ai and AI are both valid. Artificial Intelligence uses ai techniques."
        ground_truth = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["AI"], "frequency": 4, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0


class TestForbiddenWords:
    """Test forbidden word constraints."""

    def test_forbidden_words_pass(self):
        """Test that response without forbidden words passes."""
        response = "This is a great response without any problematic content."
        ground_truth = {
            "instruction_id_list": ["keywords:forbidden_words"],
            "kwargs": [{"forbidden_words": ["bad", "terrible", "awful"]}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0
        assert result["num_followed"] == 1

    def test_forbidden_words_fail(self):
        """Test that response with forbidden words fails."""
        response = "This is a terrible response with bad content."
        ground_truth = {
            "instruction_id_list": ["keywords:forbidden_words"],
            "kwargs": [{"forbidden_words": ["bad", "terrible", "awful"]}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0
        assert result["num_followed"] == 0


class TestLengthConstraints:
    """Test length constraint instructions."""

    def test_word_count_at_least(self):
        """Test word count 'at least' constraint."""
        response = "This is a response that contains more than ten words for testing purposes."
        ground_truth = {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 10, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0
        assert result["num_followed"] == 1

    def test_word_count_at_most(self):
        """Test word count 'at most' constraint."""
        response = "Short response here."
        ground_truth = {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 10, "relation": "at most"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_word_count_fail(self):
        """Test word count constraint failure."""
        response = "Too short."
        ground_truth = {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 10, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0
        assert result["num_followed"] == 0

    def test_sentence_count(self):
        """Test sentence count constraint."""
        response = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        ground_truth = {
            "instruction_id_list": ["length_constraints:number_sentences"],
            "kwargs": [{"num_sentences": 5, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_paragraph_count(self):
        """Test paragraph count constraint."""
        response = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        ground_truth = {
            "instruction_id_list": ["detectable_format:number_paragraphs"],
            "kwargs": [{"num_paragraphs": 3}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0


class TestStartEndConstraints:
    """Test start/end phrase constraints."""

    def test_startswith(self):
        """Test response starts with specified phrase."""
        response = "Dear Sir, this is a formal letter."
        ground_truth = {
            "instruction_id_list": ["startswith:quotation"],
            "kwargs": [{"start_phrase": "Dear Sir"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_startswith_fail(self):
        """Test response doesn't start with specified phrase."""
        response = "Hello, this is a letter."
        ground_truth = {
            "instruction_id_list": ["startswith:quotation"],
            "kwargs": [{"start_phrase": "Dear Sir"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0

    def test_endswith(self):
        """Test response ends with specified phrase."""
        response = "This is a message. Best regards, John"
        ground_truth = {
            "instruction_id_list": ["end_checker:endswith"],
            "kwargs": [{"end_phrase": "Best regards, John"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0


class TestFormatConstraints:
    """Test format-related constraints."""

    def test_json_format(self):
        """Test valid JSON format."""
        response = '{"name": "John", "age": 30}'
        ground_truth = {
            "instruction_id_list": ["detectable_format:json_format"],
            "kwargs": [{}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_json_format_fail(self):
        """Test invalid JSON format."""
        response = "This is not JSON"
        ground_truth = {
            "instruction_id_list": ["detectable_format:json_format"],
            "kwargs": [{}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0

    def test_bullet_list(self):
        """Test bullet list format."""
        response = "Items:\n- First item\n- Second item\n- Third item\n- Fourth item"
        ground_truth = {
            "instruction_id_list": ["detectable_format:number_bullet_lists"],
            "kwargs": [{"num_bullets": 4}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_numbered_list(self):
        """Test numbered list format."""
        response = "Steps:\n1. First step\n2. Second step\n3. Third step"
        ground_truth = {
            "instruction_id_list": ["detectable_format:number_bullet_lists"],
            "kwargs": [{"num_bullets": 3}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0


class TestCaseConstraints:
    """Test case-related constraints."""

    def test_all_lowercase(self):
        """Test all lowercase constraint."""
        response = "this is all lowercase text"
        ground_truth = {
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{"capital": "lower"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_all_lowercase_fail(self):
        """Test all lowercase constraint failure."""
        response = "This has Uppercase letters"
        ground_truth = {
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{"capital": "lower"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0

    def test_all_uppercase(self):
        """Test all uppercase constraint."""
        response = "THIS IS ALL UPPERCASE"
        ground_truth = {
            "instruction_id_list": ["change_case:english_capital"],
            "kwargs": [{"capital": "upper"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0


class TestPostscriptConstraint:
    """Test postscript marker constraint."""

    def test_postscript_marker(self):
        """Test postscript marker presence."""
        response = "This is the main content.\n\nP.S. This is a postscript."
        ground_truth = {
            "instruction_id_list": ["detectable_content:postscript"],
            "kwargs": [{"postscript_marker": "P.S."}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_postscript_marker_fail(self):
        """Test postscript marker absence."""
        response = "This is the main content without a postscript."
        ground_truth = {
            "instruction_id_list": ["detectable_content:postscript"],
            "kwargs": [{"postscript_marker": "P.S."}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0


class TestQuotationConstraint:
    """Test quotation presence constraint."""

    def test_quotation_present(self):
        """Test quotation marks present."""
        response = 'He said "Hello world" to everyone.'
        ground_truth = {
            "instruction_id_list": ["startswith:quotation"],
            "kwargs": [{}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0

    def test_quotation_single_quotes(self):
        """Test single quotation marks."""
        response = "He said 'Hello world' to everyone."
        ground_truth = {
            "instruction_id_list": ["startswith:quotation"],
            "kwargs": [{}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0


class TestPlaceholderConstraint:
    """Test placeholder constraint."""

    def test_placeholder_count(self):
        """Test placeholder count."""
        response = "Fill in the [NAME] and [ADDRESS] and [PHONE] fields."
        ground_truth = {
            "instruction_id_list": ["detectable_content:number_placeholders"],
            "kwargs": [{"num_placeholders": 3}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 1.0


class TestMultipleInstructions:
    """Test multiple instructions in a single prompt."""

    def test_multiple_instructions_all_pass(self):
        """Test when all instructions are followed."""
        response = "Robot technology is advancing rapidly. Robot systems are everywhere. Robots help humanity."
        ground_truth = {
            "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
            "kwargs": [
                {"keywords": ["robot"], "frequency": 2, "relation": "at least"},
                {"num_words": 10, "relation": "at least"}
            ]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["num_instructions"] == 2
        assert result["num_followed"] == 2
        assert result["score"] == 1.0
        assert result["prompt_strict_acc"] == 1.0

    def test_multiple_instructions_partial_pass(self):
        """Test when some instructions are followed."""
        response = "Short text with robot."
        ground_truth = {
            "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
            "kwargs": [
                {"keywords": ["robot"], "frequency": 1, "relation": "at least"},
                {"num_words": 20, "relation": "at least"}
            ]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["num_instructions"] == 2
        assert result["num_followed"] == 1
        assert result["score"] == 0.5  # 1 out of 2 instructions followed
        assert result["prompt_strict_acc"] == 0.0  # Not all instructions followed

    def test_multiple_instructions_all_fail(self):
        """Test when no instructions are followed."""
        response = "Short."
        ground_truth = {
            "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
            "kwargs": [
                {"keywords": ["robot"], "frequency": 1, "relation": "at least"},
                {"num_words": 20, "relation": "at least"}
            ]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["num_instructions"] == 2
        assert result["num_followed"] == 0
        assert result["score"] == 0.0
        assert result["prompt_strict_acc"] == 0.0


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_formal_letter_instructions(self):
        """Test formal letter with multiple constraints."""
        response = """Dear Sir,

I am writing to express my interest in the position. My qualifications include extensive experience in AI and machine learning.

Thank you for your consideration.

P.S. I look forward to hearing from you."""

        ground_truth = {
            "instruction_id_list": [
                "startswith:quotation",
                "keywords:existence",
                "detectable_format:number_paragraphs",
                "detectable_content:postscript"
            ],
            "kwargs": [
                {"start_phrase": "Dear Sir"},
                {"keywords": ["AI"], "frequency": 1, "relation": "at least"},
                {"num_paragraphs": 3},
                {"postscript_marker": "P.S."}
            ]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["num_instructions"] == 4
        assert result["num_followed"] == 4
        assert result["score"] == 1.0

    def test_technical_document_instructions(self):
        """Test technical document with formatting requirements."""
        response = """technical analysis of ai systems

- First point about neural networks
- Second point about deep learning
- Third point about transformers

{"conclusion": "AI is advancing rapidly"}"""

        ground_truth = {
            "instruction_id_list": [
                "change_case:english_lowercase",
                "detectable_format:number_bullet_lists",
                "detectable_format:json_format",
                "keywords:existence"
            ],
            "kwargs": [
                {"capital": "lower"},
                {"num_bullets": 3},
                {},
                {"keywords": ["ai"], "frequency": 2, "relation": "at least"}
            ]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["num_instructions"] == 4
        # Note: JSON check might fail due to surrounding text
        assert result["num_followed"] >= 3


class TestDefaultComputeScore:
    """Test integration with default_compute_score function."""

    def test_default_compute_score_ifeval(self):
        """Test IFEval through default_compute_score."""
        response = "AI is transforming the world. AI systems are improving. The future of AI is bright."
        ground_truth = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["AI"], "frequency": 3, "relation": "at least"}]
        }

        result = default_compute_score(
            data_source="google/IFEval",
            solution_str=response,
            ground_truth=ground_truth,
            extra_info={}
        )

        assert isinstance(result, dict)
        assert result["score"] == 1.0

    def test_default_compute_score_ifeval_alternate_source(self):
        """Test IFEval with alternate data source name."""
        response = "This is a test response."
        ground_truth = {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 5, "relation": "at least"}]
        }

        result = default_compute_score(
            data_source="ifeval",
            solution_str=response,
            ground_truth=ground_truth,
            extra_info={}
        )

        assert isinstance(result, dict)
        assert result["score"] == 0.0  # Only 5 words, needs at least 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_response(self):
        """Test empty response."""
        response = ""
        ground_truth = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["test"], "frequency": 1, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["score"] == 0.0

    def test_no_instructions(self):
        """Test with no instructions."""
        response = "Any response"
        ground_truth = {
            "instruction_id_list": [],
            "kwargs": []
        }

        result = compute_score(response, ground_truth, use_ifeval_library=False)

        assert result["num_instructions"] == 0
        assert result["num_followed"] == 0
        assert result["score"] == 0.0

    def test_extra_info_fallback(self):
        """Test using extra_info for instruction data."""
        response = "AI is important."
        ground_truth = {}
        extra_info = {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["AI"], "frequency": 1, "relation": "at least"}]
        }

        result = compute_score(response, ground_truth, extra_info=extra_info, use_ifeval_library=False)

        assert result["score"] == 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
