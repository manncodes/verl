import re

def format_reward(predict_str: str) -> float:
    """
    Validates custom thinking formats with evaluation tags.
    Supports both direct correct evaluation and retry patterns.

    FIXED: Uses word boundaries to prevent matching CORRECT inside INCORRECT.
    """

    # Pattern 1: Direct correct evaluation
    # <think>...</think><evaluation>...CORRECT...</evaluation><answer>...</answer>
    # Use \b for word boundary to avoid matching "INCORRECT"
    pattern_direct = re.compile(
        r'<think>'
        r'(?:.*?<planning>.*?</planning>)?'  # optional planning
        r'(?:.*?<draft answer>.*?</draft answer>)?'  # optional draft
        r'.*?</think>'
        r'.*?<evaluation>.*?\bCORRECT\b.*?</evaluation>'  # FIXED: word boundary
        r'.*?<answer>.+?</answer>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern 2: Retry with incorrect then correct
    # <think>...</think><evaluation>INCORRECT</evaluation><think>...</think><evaluation>CORRECT</evaluation><answer>...</answer>
    pattern_retry = re.compile(
        r'<think>'
        r'(?:.*?<planning>.*?</planning>)?'
        r'(?:.*?<draft answer>.*?</draft answer>)?'
        r'.*?</think>'
        r'.*?<evaluation>.*?\bINCORRECT\b.*?</evaluation>'  # FIXED: word boundary
        r'.*?<think>'
        r'(?:.*?<planning>.*?</planning>)?'
        r'(?:.*?<draft answer>.*?</draft answer>)?'
        r'.*?</think>'
        r'.*?<evaluation>.*?\bCORRECT\b.*?</evaluation>'  # FIXED: word boundary
        r'.*?<answer>.+?</answer>',
        re.DOTALL | re.IGNORECASE
    )

    # Additional validation: ensure no duplicate opening tags where not expected
    duplicate_check = re.compile(
        r'<answer>.*<answer>|'  # no duplicate answer opens
        r'<evaluation>.*<evaluation>.*<evaluation>',  # max 2 evaluations
        re.DOTALL
    )

    # Check if string matches either valid pattern and has no invalid duplicates
    if duplicate_check.search(predict_str):
        return 0.0

    if pattern_direct.match(predict_str) or pattern_retry.match(predict_str):
        return 1.0

    return 0.0


# More robust version with additional validation
def format_reward_strict(predict_str: str) -> float:
    """
    Stricter validation with proper tag closure and content checks.

    FIXED: Uses word boundaries to prevent matching CORRECT inside INCORRECT.
    """

    def validate_tag_balance(text: str, tag: str) -> bool:
        """Check if a tag is properly opened and closed."""
        open_count = text.count(f'<{tag}>')
        close_count = text.count(f'</{tag}>')
        return open_count == close_count and open_count > 0

    # Basic tag balance validation
    required_tags = ['think', 'answer']
    for tag in required_tags:
        if not validate_tag_balance(predict_str, tag):
            return 0.0

    # Check for evaluation tag with CORRECT (must exist at least once)
    # FIXED: Use word boundary to avoid matching "INCORRECT"
    if not re.search(r'<evaluation>.*?\bCORRECT\b.*?</evaluation>', predict_str, re.DOTALL | re.IGNORECASE):
        return 0.0

    # Validate sequence order
    try:
        # Find last CORRECT evaluation position
        # FIXED: Use word boundary
        correct_eval = list(re.finditer(r'<evaluation>.*?\bCORRECT\b.*?</evaluation>',
                                       predict_str, re.DOTALL | re.IGNORECASE))[-1]

        # Find answer position
        answer_match = re.search(r'<answer>.+?</answer>', predict_str, re.DOTALL)

        if not answer_match:
            return 0.0

        # Answer must come after the final CORRECT evaluation
        if answer_match.start() < correct_eval.end():
            return 0.0

        # Check for think tag before evaluation
        think_before = predict_str.rfind('</think>', 0, correct_eval.start())
        if think_before == -1:
            return 0.0

    except (IndexError, AttributeError):
        return 0.0

    return 1.0


# Alternative approach: Use negative lookahead/lookbehind
def format_reward_v2(predict_str: str) -> float:
    """
    Alternative implementation using negative lookbehind.
    This ensures CORRECT is not preceded by "IN".
    """

    # Pattern 1: Direct correct evaluation
    # Use negative lookbehind to ensure CORRECT is not part of INCORRECT
    pattern_direct = re.compile(
        r'<think>'
        r'(?:.*?<planning>.*?</planning>)?'
        r'(?:.*?<draft answer>.*?</draft answer>)?'
        r'.*?</think>'
        r'.*?<evaluation>.*?(?<!IN)CORRECT.*?</evaluation>'  # Negative lookbehind
        r'.*?<answer>.+?</answer>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern 2: Retry pattern
    pattern_retry = re.compile(
        r'<think>'
        r'(?:.*?<planning>.*?</planning>)?'
        r'(?:.*?<draft answer>.*?</draft answer>)?'
        r'.*?</think>'
        r'.*?<evaluation>.*?INCORRECT.*?</evaluation>'
        r'.*?<think>'
        r'(?:.*?<planning>.*?</planning>)?'
        r'(?:.*?<draft answer>.*?</draft answer>)?'
        r'.*?</think>'
        r'.*?<evaluation>.*?(?<!IN)CORRECT.*?</evaluation>'  # Negative lookbehind
        r'.*?<answer>.+?</answer>',
        re.DOTALL | re.IGNORECASE
    )

    # Additional validation
    duplicate_check = re.compile(
        r'<answer>.*<answer>|'
        r'<evaluation>.*<evaluation>.*<evaluation>',
        re.DOTALL
    )

    if duplicate_check.search(predict_str):
        return 0.0

    if pattern_direct.match(predict_str) or pattern_retry.match(predict_str):
        return 1.0

    return 0.0
