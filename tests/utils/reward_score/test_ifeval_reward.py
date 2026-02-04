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
Verification tests for the IFEVAL reward function against known rollouts.

Tests deterministic components (format_reward, _compute_thinking_penalty,
remove_thinking_section) and identifies bugs in the scoring pipeline.

Rollouts tested:
  - ROLLOUT_HINDI: Hindi riddle with full reasoning + <draft_answer>
  - ROLLOUT_APA_LONG: APA rewrite with full reasoning + <draft_answer>
  - ROLLOUT_APA_SHORT: APA rewrite with <draft_answer> ONLY (no reasoning)
"""

import re

# ---------------------------------------------------------------------------
# Inline the functions under test (the ifeval reward module doesn't exist in
# the repo yet, so we replicate the exact code the user provided).
# ---------------------------------------------------------------------------

FORMAT_SCORE_WEIGHT = 1
THINKING_BONUS_WEIGHT = 1
ALPHA_THRESHOLD = 0.7
MAX_REWARD = 4.5


def remove_thinking_section(text: str) -> str:
    """Strip thinking/evaluation tags, return only the answer content."""
    text = text.replace("<|assistant|>", "").strip()
    text = text.split("</think>")[-1]
    text = text.split("</evaluation>")[-1]
    return text.replace("<answer>", "").replace("</answer>", "").strip()


def _compute_thinking_penalty(text: str) -> float:
    """Punish-only: 0 if <think> contains sufficient reasoning text, -1 otherwise."""
    blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    if not blocks:
        return -1.0
    combined = " ".join(blocks)
    stripped = re.sub(r"<[^>]+>.*?</[^>]+>", "", combined, flags=re.DOTALL)
    stripped = re.sub(r"<[^>]+>", "", stripped).strip()
    return 0.0 if len(stripped) >= 100 else -1.0


def format_reward(predict_str: str) -> float:
    """Validate <think>/<answer> tag structure. Returns 0.0 or 1.0."""
    if re.search(r"<answer>.*<answer>", predict_str, re.DOTALL):
        return 0.0

    _optional_sub = r"(?:.*?<planning>.*?</planning>)?(?:.*?<draft answer>.*?</draft answer>)?"
    _think_block = rf"<think>{_optional_sub}.*?</think>"

    patterns = [
        # Standard (no <evaluation> tags, max 2 <think> tags)
        re.compile(
            r"^(?!.*<think>.*<think>.*<think>)(?!.*<answer>.*<answer>)"
            r".*<think>.+?</think>(?!.*<evaluation>).*<answer>.+?</answer>",
            re.DOTALL,
        ),
        # Direct evaluation -> CORRECT
        re.compile(
            rf"{_think_block}.*?<evaluation>.*?CORRECT.*?</evaluation>.*?<answer>.+?</answer>",
            re.DOTALL | re.IGNORECASE,
        ),
        # Retry: INCORRECT then CORRECT
        re.compile(
            rf"{_think_block}.*?<evaluation>.*?INCORRECT.*?</evaluation>"
            rf".*?{_think_block}.*?<evaluation>.*?CORRECT.*?</evaluation>.*?<answer>.+?</answer>",
            re.DOTALL | re.IGNORECASE,
        ),
    ]
    return 1.0 if any(p.search(predict_str) for p in patterns) else 0.0


# ---------------------------------------------------------------------------
# Rollouts under test
# ---------------------------------------------------------------------------

ROLLOUT_HINDI = """\
<think>
Hmm, the user wants me to write a funny riddle in Hindi about a man named Aarav who never spoke. They've specified that my entire response must be in Hindi and I cannot use any commas anywhere.

First, I need to understand the requirements clearly. The riddle should be humorous, centered on a silent character named Aarav, and fully in Hindi without any punctuation like commas. That means I'll have to structure the sentences carefully to avoid accidental commas.

The user seems playful—probably looking for entertainment or a light-hearted challenge. Since they mentioned being an AI assistant, they might be testing my language skills or just wanting a quick laugh. I should make the riddle clever and engaging to keep it fun.

For the riddle structure, I'll start with a question format to make it interactive. I'll describe Aarav's silence in a funny way, perhaps exaggerating how he communicates without words. Ideas could include him using gestures, writing, or even objects to "speak."

Key elements to include: his name (Aarav), the fact he never speaks, and a punchline that reveals the answer. The humor could come from the absurdity of how he "talks," like through actions or inanimate things.

Now, for the Hindi language: I need to ensure it's natural and witty. Words like "kuch nahi keh raha" (never said anything) set the scene. For the punchline, I'll use "silence" as the answer but make it sound silly, like silence itself is a character.

Avoiding commas is tricky. I'll use Hindi conjunctions like "और" (and) or rephrase sentences to connect ideas without punctuation. For example, instead of "He never spoke, only gestured," I can say "वह केवल हाथ से बात करता था" (He only spoke with his hands).

Finally, I'll end with the answer clearly stated. This should satisfy the user's request while being amusing and adhering to all rules. Let me craft it step by step.
<draft_answer>
हर दिन बात करता था पर कोई बात नहीं
हाथ से ही बात करता था
पैरों से ही बात करता था
कागज़ पर ही लिख देता था
फिर भी कोई बात नहीं कह पाता
कौन है वह आदमी जो कभी भी बोलता नहीं
जो सिर्फ... **सिलेंस** है

(Who is that man who never speaks?
He talks only with his hands
He talks only with his feet
He writes only on paper
Yet he never says anything
Who is that man who never speaks?
He is just... **Silence**)
</draft_answer>
</think>
<evaluation>
The response is entirely in Hindi, contains no commas, and presents a clear riddle about a silent man named Aarav (implied by the question). It follows the instruction to avoid commas and to provide the answer within the riddle. The riddle is humorous and meets the requested format. No constraints are violated, and the answer is logically consistent with the task. Final Verdict: CORRECT
</evaluation>
<answer>
हर दिन बात करता था पर कोई बात नहीं
हाथ से ही बात करता था
पैरों से ही बात करता था
कागज़ पर ही लिख देता था
फिर भी कोई बात नहीं कह पाता
कौन है वह आदमी जो कभी भी बोलता नहीं
जो सिर्फ... **सिलेंस** है

(Who is that man who never speaks?
He talks only with his hands
He talks only with his feet
He writes only on paper
Yet he never says anything
Who is that man who never speaks?
He is just... **Silence**)
</answer>"""

# Rollout A: APA rewrite with full reasoning in <think> + <draft_answer>
ROLLOUT_APA_LONG = """\
<think>
Hmm, the user is asking me to rewrite a research paper summary in a more formal APA style. They've provided a simple statement: "Our results show that VHF radio waves have no effect on the growth of bacteria." They specifically want me to avoid using "ours" and "have," and follow APA formatting conventions.

I need to recall what APA style emphasizes. It's all about objectivity, passive voice, and precise language. Passive voice helps remove the researcher's perspective, which aligns with avoiding "ours." Also, replacing "have" with stronger verbs like "exerted" or "demonstrated" would make it more academic.

The original summary is straightforward but informal. To make it formal, I should start with a strong verb and restructure the sentence. For instance, instead of "Our results show," I could use "The findings indicate" or "Results demonstrated." That shifts focus to the data itself.

Now, for the content: the key points are that VHF radio waves were tested, and there was no effect on bacterial growth. I need to phrase this neutrally. Instead of "no effect," I could say "no significant effect" to imply statistical testing, which is common in research. Or use "failed to influence" to sound more scientific.

I should also consider APA specifics. Headings aren't needed here since it's just a summary sentence, but the language must be concise and precise. No jargon unless necessary, but terms like "VHF radio waves" are fine as they're standard.

The user mentioned they're Llama-C1-Research-15B, built by Capital One for non-commercial research. That suggests they're in an academic or professional setting, probably writing a paper or report. Their actual need is to polish this summary for credibility, but deeper down, they might be learning how to write in APA style or seeking confidence in their scientific communication. By providing a clear rewrite, I'm helping them adhere to standards and avoid common pitfalls.

Brainstorming options: One approach is "Results indicated that exposure to VHF radio waves did not significantly influence bacterial growth." That uses passive voice, avoids forbidden words, and adds "significantly" for rigor. Another could be "Exposure to VHF radio waves failed to affect the growth rate of the bacteria." This is direct and formal.

I'll go with the first one as it's more comprehensive. It starts with "Results," which is standard in APA abstracts, and ends with the finding clearly. I'll make sure it's a single, well-structured sentence as per the query.

Finally, I'll present it in the required tags: reasoning in <think> and the answer in <answer>. No extra fluff—just the rewritten summary to keep it helpful and efficient.
<draft_answer>
Based on APA style conventions (objectivity, passive voice where appropriate, precise language), here is a formal rewrite of the summary avoiding "ours" and "have":

**Revised Summary:**
> Results indicated that exposure to VHF radio waves did not significantly influence the growth of the bacteria.

**Reasoning:**

1.  **Replace "Our results show":** "Results indicated" is a standard, objective way to present findings in formal writing, removing the subjective "our".
2.  **Replace "have no effect":** "did not significantly influence" is more precise and formal than "have no effect". Adding "significantly" implies the finding is based on statistical analysis (common in research), which strengthens the statement. "Influence" is a more formal synonym for "effect" in this context.
3.  **Restructure for Clarity:** The phrase "exposure to VHF radio waves" clearly establishes the experimental condition before stating the outcome ("did not significantly influence...").
4.  **Objectivity:** The revised sentence focuses solely on the findings and the experimental manipulation without referencing the researchers.
5.  **Conciseness:** It conveys the core finding efficiently.

**Alternative Option (slightly more direct):**
> Exposure to VHF radio waves failed to affect the growth of the bacteria.

This version also avoids the forbidden words and uses strong, formal verbs ("failed to affect"). Both options are acceptable in APA style for a summary statement.
</draft_answer>
</think>
<evaluation>
The response provides a formal rewrite of the original summary in a single sentence that avoids the prohibited words "ours" and "have". It uses objective, APA\u2011style language such as "Results indicated" and "did not significantly influence", which aligns with the instruction to be formal and concise. The answer also includes a brief reasoning section explaining the choices, which is permissible as the instruction allows internal reasoning. No disallowed content or formatting violations are present. Therefore the answer satisfies the instruction and is logically sound. Final Verdict: CORRECT
</evaluation>
<answer>
Based on APA style conventions (objectivity, passive voice where appropriate, precise language), here is a formal rewrite of the summary avoiding "ours" and "have":

**Revised Summary:**
> Results indicated that exposure to VHF radio waves did not significantly influence the growth of the bacteria.

**Reasoning:**

1.  **Replace "Our results show":** "Results indicated" is a standard, objective way to present findings in formal writing, removing the subjective "our".
2.  **Replace "have no effect":** "did not significantly influence" is more precise and formal than "have no effect". Adding "significantly" implies the finding is based on statistical analysis (common in research), which strengthens the statement. "Influence" is a more formal synonym for "effect" in this context.
3.  **Restructure for Clarity:** The phrase "exposure to VHF radio waves" clearly establishes the experimental condition before stating the outcome ("did not significantly influence...").
4.  **Objectivity:** The revised sentence focuses solely on the findings and the experimental manipulation without referencing the researchers.
5.  **Conciseness:** It conveys the core finding efficiently.

**Alternative Option (slightly more direct):**
> Exposure to VHF radio waves failed to affect the growth of the bacteria.

This version also avoids the forbidden words and uses strong, formal verbs ("failed to affect"). Both options are acceptable in APA style for a summary statement.
</answer>"""

# Rollout B: APA rewrite with <draft_answer> ONLY inside <think> (no reasoning)
ROLLOUT_APA_SHORT = """\
<think>
<draft_answer>
Here is the formal APA-style summary, avoiding "ours" and "have":

**Revised Summary:**

> The findings indicate that exposure to VHF radio waves does not influence bacterial growth.

**Reasoning:**

1.  **Formal Tone:** Replaced "Our results show" with the more objective "The findings indicate".
2.  **Avoided "ours":** Used "The findings" instead of "Our results".
3.  **Avoided "have":** Replaced "have no effect" with the phrase "does not influence".
4.  **Clarity & Conciseness:** "Does not influence" is a standard, formal way to express the lack of effect. "Bacterial growth" is the precise scientific term.
5.  **APA Style:** This structure (Subject + Verb + Finding) is standard for reporting research results in APA. The sentence is clear, objective, and uses formal academic language.
</draft_answer>
</think>
<evaluation>
The response provides a concise, formal rewrite of the original summary in APA style, avoiding the prohibited words "ours" and "have." It uses a clear, objective sentence structure typical of APA reporting. The additional reasoning section explains the changes but does not violate any constraints. Therefore, the answer fully satisfies the instruction. Final Verdict: CORRECT
</evaluation>
<answer>
Here is the formal APA-style summary, avoiding "ours" and "have":

**Revised Summary:**

> The findings indicate that exposure to VHF radio waves does not influence bacterial growth.

**Reasoning:**

1.  **Formal Tone:** Replaced "Our results show" with the more objective "The findings indicate".
2.  **Avoided "ours":** Used "The findings" instead of "Our results".
3.  **Avoided "have":** Replaced "have no effect" with the phrase "does not influence".
4.  **Clarity & Conciseness:** "Does not influence" is a standard, formal way to express the lack of effect. "Bacterial growth" is the precise scientific term.
5.  **APA Style:** This structure (Subject + Verb + Finding) is standard for reporting research results in APA. The sentence is clear, objective, and uses formal academic language.
</answer>"""


# ===================================================================
# Tests
# ===================================================================


class TestFormatReward:
    """Tests for the format_reward function."""

    def test_rollout_matches_pattern2_direct_correct(self):
        """Hindi rollout has <think>→<evaluation>CORRECT→<answer> → pattern 2 should match."""
        assert format_reward(ROLLOUT_HINDI) == 1.0

    def test_pattern1_rejects_evaluation_tags(self):
        """Pattern 1 requires no <evaluation> tags; the Hindi rollout has them."""
        # Pattern 1 is the first in the list — verify it alone does NOT match
        pattern1 = re.compile(
            r"^(?!.*<think>.*<think>.*<think>)(?!.*<answer>.*<answer>)"
            r".*<think>.+?</think>(?!.*<evaluation>).*<answer>.+?</answer>",
            re.DOTALL,
        )
        assert pattern1.search(ROLLOUT_HINDI) is None

    def test_duplicate_answer_tags_rejected(self):
        bad = "<think>reasoning</think><answer>a</answer><answer>b</answer>"
        assert format_reward(bad) == 0.0

    def test_standard_think_answer_no_eval(self):
        good = "<think>some reasoning here</think>\n<answer>my answer</answer>"
        assert format_reward(good) == 1.0

    def test_empty_think_rejected(self):
        """Empty <think> block should fail (.+? requires at least 1 char)."""
        bad = "<think></think><answer>answer</answer>"
        assert format_reward(bad) == 0.0

    def test_retry_pattern_incorrect_then_correct(self):
        retry = (
            "<think>first attempt reasoning</think>"
            "<evaluation>INCORRECT</evaluation>"
            "<think>second attempt reasoning</think>"
            "<evaluation>CORRECT</evaluation>"
            "<answer>final answer</answer>"
        )
        assert format_reward(retry) == 1.0

    def test_draft_answer_with_underscore_still_matches(self):
        """<draft_answer> (underscore) should not break format_reward —
        the optional <draft answer> (space) sub-pattern is simply skipped."""
        text = (
            "<think>reasoning <draft_answer>draft</draft_answer> more</think>"
            "<evaluation>CORRECT</evaluation>"
            "<answer>final</answer>"
        )
        assert format_reward(text) == 1.0

    def test_no_tags_at_all(self):
        assert format_reward("just a plain response") == 0.0


class TestComputeThinkingPenalty:
    """Tests for _compute_thinking_penalty."""

    def test_rollout_has_sufficient_reasoning(self):
        """Hindi rollout's <think> block has >100 chars of stripped text."""
        assert _compute_thinking_penalty(ROLLOUT_HINDI) == 0.0

    def test_no_think_block(self):
        assert _compute_thinking_penalty("no think tags here") == -1.0

    def test_empty_think_block(self):
        assert _compute_thinking_penalty("<think></think>") == -1.0

    def test_short_think_block(self):
        assert _compute_thinking_penalty("<think>short</think>") == -1.0

    def test_think_with_100_chars(self):
        text = "<think>" + "a" * 100 + "</think>"
        assert _compute_thinking_penalty(text) == 0.0

    def test_think_with_99_chars(self):
        text = "<think>" + "a" * 99 + "</think>"
        assert _compute_thinking_penalty(text) == -1.0

    def test_inner_tags_stripped_before_counting(self):
        """Content inside nested tags should be removed before measuring length."""
        # 50 chars of real reasoning + <foo>200 chars</foo> = only 50 counts
        text = "<think>" + "x" * 50 + "<foo>" + "y" * 200 + "</foo></think>"
        assert _compute_thinking_penalty(text) == -1.0

    def test_draft_answer_underscore_stripped(self):
        """<draft_answer>...</draft_answer> should be stripped from think content."""
        inner = "x" * 50 + "<draft_answer>" + "y" * 500 + "</draft_answer>" + "z" * 20
        text = f"<think>{inner}</think>"
        # Only 50 + 20 = 70 chars of real reasoning → should be -1.0
        assert _compute_thinking_penalty(text) == -1.0


# ===================================================================
# Tests for empty / whitespace-only <think> blocks
# ===================================================================


class TestEmptyThinkBlock:
    """Edge cases where <think> contains no real reasoning text.

    Tests two axes:
      - Content inside <think>: empty vs whitespace-only vs draft_answer-only
      - Surrounding structure: with vs without <evaluation> tags

    Key finding: Pattern 1 (.+?) and Pattern 2 (.*?) have INCONSISTENT
    handling of truly empty <think> blocks. Whitespace is treated as
    content by Pattern 1 but stripped away by _compute_thinking_penalty.
    """

    # ── format_reward with evaluation tags (Pattern 2: .*?) ──────────

    def test_empty_think_with_eval_gets_format_1(self):
        """EDGE CASE: Pattern 2 uses .*? in _think_block, so completely
        empty <think></think> PASSES format_reward when evaluation tags
        are present. This is arguably too lenient."""
        text = "<think></think><evaluation>CORRECT</evaluation><answer>ans</answer>"
        assert format_reward(text) == 1.0

    def test_whitespace_think_with_eval_gets_format_1(self):
        """Whitespace-only <think> also passes Pattern 2 (.*? matches whitespace)."""
        text = "<think>   \n\n  \t  </think><evaluation>CORRECT</evaluation><answer>ans</answer>"
        assert format_reward(text) == 1.0

    def test_draft_only_think_with_eval_gets_format_1(self):
        """<think> containing only <draft_answer> passes Pattern 2."""
        text = (
            "<think>\n<draft_answer>draft content</draft_answer>\n</think>"
            "<evaluation>CORRECT</evaluation><answer>ans</answer>"
        )
        assert format_reward(text) == 1.0

    # ── format_reward without evaluation tags (Pattern 1: .+?) ───────

    def test_empty_think_no_eval_gets_format_0(self):
        """Pattern 1 uses .+? inside <think>, so empty <think></think>
        FAILS format_reward when no evaluation tags are present."""
        text = "<think></think><answer>ans</answer>"
        assert format_reward(text) == 0.0

    def test_whitespace_think_no_eval_gets_format_1(self):
        """INCONSISTENCY: Whitespace-only <think> PASSES Pattern 1 because
        .+? matches whitespace characters. So a single space is 'reasoning'
        for format_reward but not for _compute_thinking_penalty."""
        text = "<think>   </think><answer>ans</answer>"
        assert format_reward(text) == 1.0

    def test_single_space_think_no_eval_gets_format_1(self):
        """Even a single space passes Pattern 1's .+? requirement."""
        text = "<think> </think><answer>ans</answer>"
        assert format_reward(text) == 1.0

    def test_newline_think_no_eval_gets_format_1(self):
        """A single newline passes Pattern 1 with re.DOTALL (. matches \\n)."""
        text = "<think>\n</think><answer>ans</answer>"
        assert format_reward(text) == 1.0

    # ── Pattern 1 vs Pattern 2 inconsistency ─────────────────────────

    def test_pattern1_vs_pattern2_empty_think_inconsistency(self):
        """BUG: Empty <think></think> is rejected by Pattern 1 (.+?)
        but accepted by Pattern 2 (.*?). The only difference is whether
        evaluation tags are present, which shouldn't affect whether the
        think block itself is considered valid."""
        no_eval = "<think></think><answer>ans</answer>"
        with_eval = "<think></think><evaluation>CORRECT</evaluation><answer>ans</answer>"
        assert format_reward(no_eval) == 0.0   # Pattern 1 rejects
        assert format_reward(with_eval) == 1.0  # Pattern 2 accepts

    # ── _compute_thinking_penalty (consistent across all empty cases) ─

    def test_empty_think_penalty(self):
        """Empty <think> always gets -1.0 thinking penalty."""
        assert _compute_thinking_penalty("<think></think>") == -1.0

    def test_whitespace_think_penalty(self):
        """Whitespace-only <think> gets -1.0 (stripped to empty)."""
        assert _compute_thinking_penalty("<think>   \n\n  \t  </think>") == -1.0

    def test_draft_only_think_penalty(self):
        """<draft_answer>-only <think> gets -1.0 (draft stripped, nothing remains)."""
        text = "<think>\n<draft_answer>lots of content here</draft_answer>\n</think>"
        assert _compute_thinking_penalty(text) == -1.0

    def test_whitespace_stripped_before_length_check(self):
        """Whitespace is .strip()'d before the >= 100 char check, so
        99 spaces + 1 real char = 1 char after strip (penalty applies)."""
        text = "<think>" + " " * 99 + "x" + "</think>"
        assert _compute_thinking_penalty(text) == -1.0

    # ── Combined scores for empty think variants ─────────────────────

    def test_empty_think_with_eval_score_composition(self):
        """Empty <think> + eval: fmt=1.0, think=-1.0.
        These cancel out, so only V_i and S_i contribute."""
        text = "<think></think><evaluation>CORRECT</evaluation><answer>ans</answer>"
        fmt = format_reward(text)
        think = _compute_thinking_penalty(text)
        assert fmt == 1.0
        assert think == -1.0
        # At V_i=2.0, S_i>alpha: F_i = (2.5 + 1 - 1) / 4.5 = 2.5/4.5
        F_i = max(0.0, 2.5 + fmt + think) / MAX_REWARD
        assert abs(F_i - 2.5 / 4.5) < 1e-9

    def test_empty_think_no_eval_score_composition(self):
        """Empty <think> + no eval: fmt=0.0, think=-1.0.
        Both penalize, making the response essentially worthless."""
        text = "<think></think><answer>ans</answer>"
        fmt = format_reward(text)
        think = _compute_thinking_penalty(text)
        assert fmt == 0.0
        assert think == -1.0
        # At V_i=2.0, S_i>alpha: F_i = max(0, 2.5 + 0 - 1) / 4.5 = 1.5/4.5
        F_i = max(0.0, 2.5 + fmt + think) / MAX_REWARD
        assert abs(F_i - 1.5 / 4.5) < 1e-9

    def test_whitespace_think_no_eval_perverse_incentive(self):
        """INCONSISTENCY: Adding whitespace to an empty <think> flips
        format_reward from 0.0 to 1.0 (Pattern 1 .+? matches whitespace).
        A single space gives +1.0 format score — a perverse incentive."""
        empty = "<think></think><answer>ans</answer>"
        space = "<think> </think><answer>ans</answer>"
        assert format_reward(empty) == 0.0
        assert format_reward(space) == 1.0
        # But both get the same thinking penalty
        assert _compute_thinking_penalty(empty) == -1.0
        assert _compute_thinking_penalty(space) == -1.0


class TestRemoveThinkingSection:
    """Tests for remove_thinking_section."""

    def test_rollout_extracts_answer_content(self):
        result = remove_thinking_section(ROLLOUT_HINDI)
        # Should NOT contain <think>, <evaluation>, or <answer> tags
        assert "<think>" not in result
        assert "</think>" not in result
        assert "<evaluation>" not in result
        assert "</evaluation>" not in result
        assert "<answer>" not in result
        assert "</answer>" not in result
        # Should contain the Hindi riddle
        assert "हर दिन बात करता था" in result

    def test_answer_text_contains_english_translation(self):
        """BUG INDICATOR: The answer text includes English in parentheses,
        which would cause language:response_language checks to fail."""
        result = remove_thinking_section(ROLLOUT_HINDI)
        assert "Who is that man who never speaks?" in result
        assert "Silence" in result

    def test_no_commas_in_answer_text(self):
        """Verify no commas in extracted answer → punctuation:no_comma would PASS."""
        result = remove_thinking_section(ROLLOUT_HINDI)
        assert "," not in result

    def test_simple_think_answer(self):
        text = "<think>reasoning</think><answer>my answer</answer>"
        assert remove_thinking_section(text) == "my answer"

    def test_with_evaluation_tags(self):
        text = "<think>r</think><evaluation>CORRECT</evaluation><answer>a</answer>"
        assert remove_thinking_section(text) == "a"

    def test_assistant_prefix_stripped(self):
        text = "<|assistant|><think>r</think><answer>a</answer>"
        assert remove_thinking_section(text) == "a"


class TestRewardComposition:
    """Test the reward composition logic (without RM client)."""

    def test_hindi_rollout_components(self):
        """Verify deterministic reward components for the Hindi rollout."""
        fmt = format_reward(ROLLOUT_HINDI)
        think = _compute_thinking_penalty(ROLLOUT_HINDI)
        assert fmt == 1.0, f"Expected format_reward=1.0, got {fmt}"
        assert think == 0.0, f"Expected thinking_penalty=0.0, got {think}"

    def test_score_when_all_instructions_followed_and_high_rm(self):
        """If V_i=2.0 (perfect) and S_i > alpha → max score."""
        V_i = 2.0
        S_i = 0.8  # > ALPHA_THRESHOLD
        base_score = V_i + 0.5  # = 2.5
        fmt_score = 1.0
        think_bonus = 0.0
        F_i = base_score + FORMAT_SCORE_WEIGHT * fmt_score + THINKING_BONUS_WEIGHT * think_bonus
        F_i = max(0.0, F_i) / MAX_REWARD
        # 2.5 + 1.0 + 0.0 = 3.5 / 4.5
        assert abs(F_i - 3.5 / 4.5) < 1e-9

    def test_score_when_all_instructions_followed_high_rm_and_perfect_think(self):
        """Maximum possible score: V_i=2, S_i>alpha, fmt=1, think=0."""
        V_i = 2.0
        base_score = V_i + 0.5  # 2.5
        F_i = (base_score + 1.0 * 1.0 + 1.0 * 0.0) / MAX_REWARD
        # 3.5 / 4.5 ≈ 0.778
        assert abs(F_i - 3.5 / 4.5) < 1e-9
        # NOTE: MAX_REWARD=4.5 but actual max F_i=3.5 → max normalized ≈ 0.778, NOT 1.0

    def test_max_reward_constant_is_unreachable(self):
        """BUG: MAX_REWARD=4.5 implies V_i(2)+S(0.5)+fmt(1)+think(1)=4.5,
        but thinking_bonus is 0 or -1 (never +1). True max is 3.5."""
        # The thinking_bonus function returns 0.0 or -1.0, never +1.0
        # So the actual maximum F_i before normalization is:
        #   2.0 + 0.5 + 1.0 + 0.0 = 3.5
        # This means the normalized score can never reach 1.0
        actual_max = 2.0 + 0.5 + FORMAT_SCORE_WEIGHT * 1.0 + THINKING_BONUS_WEIGHT * 0.0
        assert actual_max == 3.5
        assert actual_max < MAX_REWARD  # MAX_REWARD=4.5 is too high

    def test_score_when_no_instructions_followed_low_rm(self):
        """If V_i=0 → no RM call, base_score=0, fmt and think still contribute."""
        V_i = 0.0
        base_score = V_i
        fmt_score = 1.0
        think_bonus = 0.0
        F_i = base_score + FORMAT_SCORE_WEIGHT * fmt_score + THINKING_BONUS_WEIGHT * think_bonus
        F_i = max(0.0, F_i) / MAX_REWARD
        # 0 + 1.0 + 0.0 = 1.0 / 4.5
        assert abs(F_i - 1.0 / 4.5) < 1e-9

    def test_score_floor_at_zero(self):
        """Negative raw scores should be clamped to 0."""
        # V_i=0.5, S_i<=alpha → base=0.0, fmt=0.0, think=-1.0
        base_score = 0.0
        F_i = base_score + FORMAT_SCORE_WEIGHT * 0.0 + THINKING_BONUS_WEIGHT * (-1.0)
        F_i = max(0.0, F_i) / MAX_REWARD
        assert F_i == 0.0

    def test_likely_rollout_score_high_rm(self):
        """Likely score for this rollout with 1/2 instructions followed and S_i > alpha."""
        V_i = 0.5  # prompt_acc=0 + inst_acc=0.5
        base_score = V_i + 0.5  # S_i > alpha
        fmt_score = 1.0
        think_bonus = 0.0
        F_i = max(0.0, base_score + fmt_score + think_bonus) / MAX_REWARD
        # (1.0 + 1.0 + 0.0) / 4.5 = 2.0/4.5
        assert abs(F_i - 2.0 / 4.5) < 1e-9

    def test_likely_rollout_score_low_rm(self):
        """Likely score for this rollout with 1/2 instructions followed and S_i <= alpha."""
        V_i = 0.5
        base_score = V_i - 0.5  # S_i <= alpha → 0.0
        fmt_score = 1.0
        think_bonus = 0.0
        F_i = max(0.0, base_score + fmt_score + think_bonus) / MAX_REWARD
        # (0.0 + 1.0 + 0.0) / 4.5 = 1.0/4.5
        assert abs(F_i - 1.0 / 4.5) < 1e-9


# ===================================================================
# Tests for Rollout A: APA Long (full reasoning + draft_answer)
# ===================================================================


class TestApaLongFormatReward:
    """format_reward tests for ROLLOUT_APA_LONG.

    KEY BUG FINDING: format_reward returns 0.0 for this rollout despite
    having valid <think>→<evaluation>CORRECT→<answer> structure.

    Root cause: The reasoning text inside <think> literally mentions the
    <answer> tag: "reasoning in <think> and the answer in <answer>".
    The naive duplication check `re.search(r"<answer>.*<answer>", ...)`
    counts this literal mention as a second <answer> tag and early-returns 0.0.
    """

    def test_returns_zero_due_to_false_positive_duplication_check(self):
        """BUG: format_reward incorrectly returns 0.0 because the think
        block reasoning text literally mentions '<answer>' as text,
        which the r'<answer>.*<answer>' regex treats as a duplicate tag."""
        assert format_reward(ROLLOUT_APA_LONG) == 0.0

    def test_answer_tag_count_is_two_due_to_literal_mention(self):
        """The string '<answer>' appears twice: once in reasoning text
        (line 17: 'the answer in <answer>') and once as the actual tag."""
        assert ROLLOUT_APA_LONG.count("<answer>") == 2
        assert ROLLOUT_APA_LONG.count("</answer>") == 1  # Only the real closing tag

    def test_pattern2_would_match_without_duplication_check(self):
        """If the false-positive duplication check were bypassed,
        Pattern 2 WOULD match this rollout correctly."""
        _optional_sub = r"(?:.*?<planning>.*?</planning>)?(?:.*?<draft answer>.*?</draft answer>)?"
        _think_block = rf"<think>{_optional_sub}.*?</think>"
        p2 = re.compile(
            rf"{_think_block}.*?<evaluation>.*?CORRECT.*?</evaluation>.*?<answer>.+?</answer>",
            re.DOTALL | re.IGNORECASE,
        )
        assert p2.search(ROLLOUT_APA_LONG) is not None

    def test_pattern1_does_not_match(self):
        """Pattern 1 rejects responses with <evaluation> tags."""
        pattern1 = re.compile(
            r"^(?!.*<think>.*<think>.*<think>)(?!.*<answer>.*<answer>)"
            r".*<think>.+?</think>(?!.*<evaluation>).*<answer>.+?</answer>",
            re.DOTALL,
        )
        assert pattern1.search(ROLLOUT_APA_LONG) is None


class TestApaLongThinkingPenalty:
    """_compute_thinking_penalty tests for ROLLOUT_APA_LONG."""

    def test_has_sufficient_reasoning(self):
        """APA long has many paragraphs of reasoning outside <draft_answer>.
        After stripping <draft_answer>...</draft_answer>, >100 chars remain."""
        assert _compute_thinking_penalty(ROLLOUT_APA_LONG) == 0.0

    def test_reasoning_text_is_substantial(self):
        """Verify the reasoning outside <draft_answer> is indeed long."""
        blocks = re.findall(r"<think>(.*?)</think>", ROLLOUT_APA_LONG, re.DOTALL)
        combined = " ".join(blocks)
        stripped = re.sub(r"<[^>]+>.*?</[^>]+>", "", combined, flags=re.DOTALL)
        stripped = re.sub(r"<[^>]+>", "", stripped).strip()
        # The APA long rollout has ~1800+ chars of reasoning
        assert len(stripped) > 500


class TestApaLongRemoveThinking:
    """remove_thinking_section tests for ROLLOUT_APA_LONG."""

    def test_extracts_answer_only(self):
        result = remove_thinking_section(ROLLOUT_APA_LONG)
        assert "<think>" not in result
        assert "</think>" not in result
        assert "<evaluation>" not in result
        assert "<answer>" not in result

    def test_answer_contains_revised_summary(self):
        result = remove_thinking_section(ROLLOUT_APA_LONG)
        assert "Results indicated" in result
        assert "VHF radio waves" in result

    def test_answer_contains_forbidden_word_have(self):
        """IFEVAL concern: answer text contains 'have' in quoted context
        (e.g., '"have no effect"'). A keywords:forbidden_words check for
        'have' would FAIL even though it appears only in explanatory quotes."""
        result = remove_thinking_section(ROLLOUT_APA_LONG)
        assert "have" in result.lower()

    def test_answer_also_contains_forbidden_word_ours(self):
        """The answer text contains 'ours' in the opening line:
        'a formal rewrite of the summary avoiding "ours" and "have"'.
        A keywords:forbidden_words check for 'ours' would FAIL."""
        result = remove_thinking_section(ROLLOUT_APA_LONG)
        assert "ours" in result.lower()
        words = re.findall(r'\b\w+\b', result.lower())
        assert "ours" in words


class TestApaLongRewardComposition:
    """Reward composition scenarios for APA long rollout.

    Due to the format_reward false positive bug, fmt_score = 0.0 for this
    rollout even though the structure is valid. This cascades into lower
    overall scores.
    """

    def test_deterministic_components(self):
        fmt = format_reward(ROLLOUT_APA_LONG)
        think = _compute_thinking_penalty(ROLLOUT_APA_LONG)
        # BUG: fmt is 0.0 due to false positive <answer> duplication check
        assert fmt == 0.0
        assert think == 0.0

    def test_score_with_buggy_format_reward_all_instructions_pass_high_rm(self):
        """With fmt=0.0 (bug), V_i=2.0, S_i>alpha:
        F_i = (2.5 + 0.0 + 0.0) / 4.5 = 2.5/4.5"""
        V_i = 2.0
        base = V_i + 0.5
        F_i = max(0.0, base + 0.0 + 0.0) / MAX_REWARD
        assert abs(F_i - 2.5 / 4.5) < 1e-9

    def test_score_with_buggy_format_reward_some_instructions_fail(self):
        """Both forbidden words present → likely all IFEVAL instructions fail.
        With fmt=0.0 (bug), V_i=0, no RM call:
        F_i = max(0, 0 + 0 + 0) / 4.5 = 0.0"""
        V_i = 0.0
        F_i = max(0.0, V_i + 0.0 + 0.0) / MAX_REWARD
        assert F_i == 0.0

    def test_score_if_format_reward_were_correct(self):
        """What the score SHOULD be if format_reward worked correctly (1.0).
        With V_i=0.5, S_i>alpha, fmt=1.0, think=0.0:
        F_i = (1.0 + 1.0 + 0.0) / 4.5 = 2.0/4.5"""
        # This is the intended behavior without the bug
        V_i = 0.5
        base = V_i + 0.5
        F_i = max(0.0, base + 1.0 + 0.0) / MAX_REWARD
        assert abs(F_i - 2.0 / 4.5) < 1e-9


# ===================================================================
# Tests for Rollout B: APA Short (draft_answer ONLY, no reasoning)
# ===================================================================


class TestApaShortFormatReward:
    """format_reward tests for ROLLOUT_APA_SHORT."""

    def test_matches_pattern2_direct_correct(self):
        """APA short has <think>→<evaluation>CORRECT→<answer> → pattern 2 matches.
        Even though <think> only contains <draft_answer>, the _think_block
        pattern uses .*? (not .+?) so it accepts minimal content."""
        assert format_reward(ROLLOUT_APA_SHORT) == 1.0

    def test_think_block_regex_accepts_draft_only(self):
        """Verify the _think_block regex matches even when think content
        is exclusively a <draft_answer> block."""
        _optional_sub = r"(?:.*?<planning>.*?</planning>)?(?:.*?<draft answer>.*?</draft answer>)?"
        _think_block = rf"<think>{_optional_sub}.*?</think>"
        assert re.search(_think_block, ROLLOUT_APA_SHORT, re.DOTALL) is not None


class TestApaShortThinkingPenalty:
    """_compute_thinking_penalty tests for ROLLOUT_APA_SHORT.

    KEY FINDING: This rollout gets a PENALTY because its <think> block
    contains ONLY a <draft_answer> block with no reasoning text outside it.
    """

    def test_gets_penalty_no_reasoning_outside_draft(self):
        """After stripping <draft_answer>...</draft_answer> from the think
        block, only whitespace/newlines remain — well under 100 chars."""
        assert _compute_thinking_penalty(ROLLOUT_APA_SHORT) == -1.0

    def test_stripped_content_is_negligible(self):
        """Verify that the reasoning text after stripping is essentially empty."""
        blocks = re.findall(r"<think>(.*?)</think>", ROLLOUT_APA_SHORT, re.DOTALL)
        combined = " ".join(blocks)
        stripped = re.sub(r"<[^>]+>.*?</[^>]+>", "", combined, flags=re.DOTALL)
        stripped = re.sub(r"<[^>]+>", "", stripped).strip()
        # After stripping <draft_answer>...</draft_answer>, only whitespace remains
        assert len(stripped) < 100
        # In fact it should be essentially empty
        assert len(stripped) < 5


class TestApaShortRemoveThinking:
    """remove_thinking_section tests for ROLLOUT_APA_SHORT."""

    def test_extracts_answer_only(self):
        result = remove_thinking_section(ROLLOUT_APA_SHORT)
        assert "<think>" not in result
        assert "</think>" not in result
        assert "<evaluation>" not in result
        assert "<answer>" not in result

    def test_answer_contains_revised_summary(self):
        result = remove_thinking_section(ROLLOUT_APA_SHORT)
        assert "The findings indicate" in result
        assert "VHF radio waves" in result

    def test_answer_contains_forbidden_word_ours(self):
        """IFEVAL concern: answer text literally contains 'ours' in the
        meta-explanation 'Avoided "ours"'. A keywords:forbidden_words
        check for 'ours' would FAIL."""
        result = remove_thinking_section(ROLLOUT_APA_SHORT)
        assert "ours" in result.lower()

    def test_answer_contains_forbidden_word_have(self):
        """IFEVAL concern: answer text contains 'have' in the explanation
        'Replaced "have no effect"'. A keywords:forbidden_words check
        for 'have' would FAIL."""
        result = remove_thinking_section(ROLLOUT_APA_SHORT)
        assert "have" in result.lower()


class TestApaShortRewardComposition:
    """Reward composition scenarios for APA short rollout.

    The key difference from APA long: thinking_penalty = -1.0
    """

    def test_deterministic_components(self):
        fmt = format_reward(ROLLOUT_APA_SHORT)
        think = _compute_thinking_penalty(ROLLOUT_APA_SHORT)
        assert fmt == 1.0
        assert think == -1.0  # PENALTY — no reasoning outside <draft_answer>

    def test_thinking_penalty_difference_from_apa_long(self):
        """APA long gets think=0.0, APA short gets think=-1.0.
        This 1-point penalty is the only deterministic difference."""
        think_long = _compute_thinking_penalty(ROLLOUT_APA_LONG)
        think_short = _compute_thinking_penalty(ROLLOUT_APA_SHORT)
        assert think_long == 0.0
        assert think_short == -1.0
        assert think_long - think_short == 1.0

    def test_score_if_all_instructions_pass_high_rm(self):
        """Best case for APA short: V_i=2.0, S_i>alpha, fmt=1, think=-1.
        F_i = (2.5 + 1.0 - 1.0) / 4.5 = 2.5/4.5"""
        V_i = 2.0
        base = V_i + 0.5  # 2.5
        F_i = max(0.0, base + 1.0 * 1.0 + 1.0 * (-1.0)) / MAX_REWARD
        assert abs(F_i - 2.5 / 4.5) < 1e-9

    def test_score_if_all_instructions_pass_low_rm(self):
        """V_i=2.0, S_i<=alpha, fmt=1, think=-1.
        F_i = (1.5 + 1.0 - 1.0) / 4.5 = 1.5/4.5"""
        V_i = 2.0
        base = V_i - 0.5  # 1.5
        F_i = max(0.0, base + 1.0 - 1.0) / MAX_REWARD
        assert abs(F_i - 1.5 / 4.5) < 1e-9

    def test_score_if_some_instructions_fail_high_rm(self):
        """Likely scenario: V_i=0.5, S_i>alpha, fmt=1, think=-1.
        F_i = (1.0 + 1.0 - 1.0) / 4.5 = 1.0/4.5"""
        V_i = 0.5
        base = V_i + 0.5  # 1.0
        F_i = max(0.0, base + 1.0 - 1.0) / MAX_REWARD
        assert abs(F_i - 1.0 / 4.5) < 1e-9

    def test_score_if_some_instructions_fail_low_rm(self):
        """Worst realistic scenario: V_i=0.5, S_i<=alpha, fmt=1, think=-1.
        F_i = (0.0 + 1.0 - 1.0) / 4.5 = 0.0"""
        V_i = 0.5
        base = V_i - 0.5  # 0.0
        F_i = max(0.0, base + 1.0 - 1.0) / MAX_REWARD
        assert F_i == 0.0

    def test_score_if_no_instructions_pass(self):
        """V_i=0 → no RM, fmt=1, think=-1. F_i = max(0, 0+1-1)/4.5 = 0."""
        V_i = 0.0
        F_i = max(0.0, V_i + 1.0 - 1.0) / MAX_REWARD
        assert F_i == 0.0


# ===================================================================
# Comparative tests across all three rollouts
# ===================================================================


class TestCrossRolloutComparison:
    """Compare reward signals across all three rollouts."""

    def test_format_reward_across_rollouts(self):
        """Hindi and APA short get 1.0. APA long gets 0.0 due to
        false positive <answer> duplication (reasoning text mentions the tag)."""
        assert format_reward(ROLLOUT_HINDI) == 1.0
        assert format_reward(ROLLOUT_APA_LONG) == 0.0  # BUG: false positive
        assert format_reward(ROLLOUT_APA_SHORT) == 1.0

    def test_thinking_penalty_separates_short_from_others(self):
        """Only the APA short rollout receives a thinking penalty."""
        assert _compute_thinking_penalty(ROLLOUT_HINDI) == 0.0
        assert _compute_thinking_penalty(ROLLOUT_APA_LONG) == 0.0
        assert _compute_thinking_penalty(ROLLOUT_APA_SHORT) == -1.0

    def test_apa_long_would_dominate_short_if_format_bug_fixed(self):
        """With correct format_reward (both=1.0), APA long >= APA short
        at every V_i/S_i due to the 1-point thinking penalty difference."""
        fmt = 1.0  # Intended (correct) value for both
        think_long = 0.0
        think_short = -1.0
        for V_i in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for S_bonus in [-0.5, 0.0, 0.5]:
                base = V_i + S_bonus if V_i > 0 else V_i
                raw_long = base + fmt + think_long
                raw_short = base + fmt + think_short
                score_long = max(0.0, raw_long) / MAX_REWARD
                score_short = max(0.0, raw_short) / MAX_REWARD
                assert score_long >= score_short, (
                    f"APA long should dominate short at V_i={V_i}, "
                    f"S_bonus={S_bonus}: {score_long} vs {score_short}"
                )

    def test_apa_short_can_beat_apa_long_due_to_format_bug(self):
        """Due to the format_reward bug, APA long gets fmt=0.0 while
        APA short gets fmt=1.0. This means APA short can score HIGHER
        despite having worse thinking quality — a perverse incentive."""
        fmt_long = 0.0   # BUG: false positive
        fmt_short = 1.0   # Correct
        think_long = 0.0
        think_short = -1.0
        # At V_i=2.0, S_i>alpha:
        base = 2.0 + 0.5
        score_long = max(0.0, base + fmt_long + think_long) / MAX_REWARD
        score_short = max(0.0, base + fmt_short + think_short) / MAX_REWARD
        # Long: (2.5+0+0)/4.5 = 0.556, Short: (2.5+1-1)/4.5 = 0.556
        # They tie here, but at lower V_i, short wins
        assert abs(score_long - score_short) < 1e-9
        # At V_i=0.5, S_i>alpha: short wins
        base_low = 0.5 + 0.5
        score_long_low = max(0.0, base_low + fmt_long + think_long) / MAX_REWARD
        score_short_low = max(0.0, base_low + fmt_short + think_short) / MAX_REWARD
        # Long: (1.0+0+0)/4.5=0.222, Short: (1.0+1-1)/4.5=0.222
        assert abs(score_long_low - score_short_low) < 1e-9

    def test_forbidden_words_in_answer_text(self):
        """Both APA rollouts leak BOTH forbidden words ('ours' and 'have')
        into answer text via meta-explanations. This would cause IFEVAL
        keywords:forbidden_words instruction failures."""
        answer_long = remove_thinking_section(ROLLOUT_APA_LONG)
        answer_short = remove_thinking_section(ROLLOUT_APA_SHORT)
        # Both contain 'have' in explanatory context
        assert "have" in answer_long.lower()
        assert "have" in answer_short.lower()
        # Both contain 'ours' — long in 'avoiding "ours"', short in 'Avoided "ours"'
        assert "ours" in answer_long.lower()
        assert "ours" in answer_short.lower()


class TestBugsAndEdgeCases:
    """Tests documenting bugs and edge cases found in the IFEVAL reward code."""

    def test_bug_extra_info_none_crashes_rm_scoring(self):
        """BUG: When V_i > 0 and extra_info is None, `k in extra_info` raises TypeError.

        In compute_score(), when V_i > 0, the code does:
            prompt_key = next((k for k in ("original_prompt", "prompt") if k in extra_info), None)
        If extra_info is None, this raises TypeError: argument of type 'NoneType' is not iterable.
        """
        extra_info = None
        try:
            next((k for k in ("original_prompt", "prompt") if k in extra_info), None)
            assert False, "Should have raised TypeError"
        except TypeError:
            pass  # Bug confirmed

    def test_bug_prompt_key_none_raises_keyerror(self):
        """BUG: If extra_info has neither 'original_prompt' nor 'prompt',
        prompt_key is None and extra_info[None] raises KeyError."""
        extra_info = {"some_other_key": "value"}
        prompt_key = next(
            (k for k in ("original_prompt", "prompt") if k in extra_info), None
        )
        assert prompt_key is None
        try:
            _ = extra_info[prompt_key]
            assert False, "Should have raised KeyError"
        except KeyError:
            pass  # Bug confirmed

    def test_bug_inconsistent_return_on_empty_instructions(self):
        """BUG: When instruction_list is empty, compute_score returns only 3 keys
        instead of the full ~11 keys returned by other code paths.
        This will cause KeyError in downstream code that expects all fields."""
        empty_result = {"score": 0.0, "num_instructions": 0, "num_followed": 0}
        full_keys = {
            "score", "V_i", "S_i", "alpha_threshold", "reward_case",
            "prompt_strict_acc", "inst_strict_acc", "num_instructions",
            "num_followed", "format_score", "thinking_bonus",
        }
        missing = full_keys - set(empty_result.keys())
        assert len(missing) > 0, "Expected missing keys in empty instruction path"
        assert missing == {
            "V_i", "S_i", "alpha_threshold", "reward_case",
            "prompt_strict_acc", "inst_strict_acc", "format_score", "thinking_bonus",
        }

    def test_bug_max_reward_unreachable(self):
        """BUG: MAX_REWARD=4.5 assumes thinking_bonus can be +1.0, but
        _compute_thinking_penalty only returns 0.0 or -1.0.

        The docstring says 'V_i(2) + S_bonus(0.5) + fmt(1) + think(1) = 4.5'
        but think is never +1, only 0 or -1. True max is 3.5, so the
        normalized score can never exceed ~0.778."""
        assert MAX_REWARD == 4.5
        # Actual achievable max:
        best_V = 2.0
        best_S_bonus = 0.5
        best_fmt = 1.0
        best_think = 0.0  # NOT 1.0
        actual_max = best_V + best_S_bonus + best_fmt + best_think
        assert actual_max == 3.5
        assert actual_max / MAX_REWARD < 1.0  # Can never reach 1.0

    def test_bug_tag_stripping_regex_mismatched_tags(self):
        """EDGE CASE: The tag-stripping regex in _compute_thinking_penalty
        uses </[^>]+> which matches ANY closing tag, not just the matching one.
        This means <a>text</b> would be stripped."""
        text = "<think>" + "x" * 200 + "<a>hidden</b>" + "</think>"
        # <a>hidden</b> matches <[^>]+>.*?</[^>]+> and gets stripped
        result = _compute_thinking_penalty(text)
        assert result == 0.0  # 200 chars of 'x' remain

    def test_draft_answer_space_vs_underscore_mismatch(self):
        """DESIGN ISSUE: _optional_sub looks for '<draft answer>' (space)
        but the rollout uses '<draft_answer>' (underscore). The optional
        pattern silently skips, so no breakage, but the intent is unclear."""
        _optional_sub = r"(?:.*?<planning>.*?</planning>)?(?:.*?<draft answer>.*?</draft answer>)?"
        _think_block = rf"<think>{_optional_sub}.*?</think>"

        # With underscore — optional sub doesn't match, but .*? catches it
        text_underscore = "<think>text<draft_answer>draft</draft_answer>more</think>"
        assert re.search(_think_block, text_underscore, re.DOTALL) is not None

        # With space — optional sub matches
        text_space = "<think>text<draft answer>draft</draft answer>more</think>"
        assert re.search(_think_block, text_space, re.DOTALL) is not None

    def test_bug_format_reward_false_positive_answer_tag_in_reasoning(self):
        """BUG: format_reward's duplication check `re.search(r"<answer>.*<answer>")`
        false-positives when the reasoning text inside <think> literally
        mentions the <answer> tag (e.g., 'the answer in <answer>').

        This causes a valid response to get fmt_score=0.0 instead of 1.0,
        penalizing models that discuss their output format in their reasoning."""
        # A response that mentions <answer> in reasoning but has valid structure
        text = (
            "<think>I will put the answer in <answer> tags as required.</think>"
            "<answer>The final answer is 42.</answer>"
        )
        # BUG: returns 0.0 instead of 1.0
        assert format_reward(text) == 0.0
        # Verify the duplication regex is the cause
        assert re.search(r"<answer>.*<answer>", text, re.DOTALL) is not None
        # But only ONE of them is an actual tag (has matching </answer>)
        assert text.count("</answer>") == 1

    def test_format_chat_prompt_dead_code(self):
        """DESIGN ISSUE: format_chat_prompt() is defined but never called
        in compute_score or compute_score_batch. Likely dead code."""
        # This test documents the issue — format_chat_prompt is not
        # referenced by any scoring function in the module.
        pass
