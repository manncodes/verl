#!/usr/bin/env python3
"""
Test repetition detector against known reward hack patterns.

Run: python tests/test_repetition_detector.py
"""

import sys
import time

# Allow running standalone
sys.path.insert(0, '/home/user/verl/verl/utils/reward')
from repetition_penalty import repetition_score, is_repetitive, apply_repetition_penalty

# =============================================================================
# Test Dataset: Known Reward Hack Patterns
# =============================================================================

REWARD_HACK_EXAMPLES = {
    "plan_is_final": "The plan is final. " * 100,
    "wait_wait": "Wait, wait, wait. Let me think about this again. " * 50,
    "step_repetition": "\n".join([f"Step {i}: Calculate the result. The result is important." for i in range(1, 101)]),
    "token_stuffing": " ".join([f"Therefore{i}" for i in range(200)]),
    "phrase_loop": " ".join([f"The answer is {42 + i % 3}. Let me verify. Yes, {42 + i % 3} is correct." for i in range(50)]),
    "reasoning_pad": "This is a complex problem. I need to think carefully. " * 80,
    "enum_abuse": "\n".join([f"{i}. Consider option {i}." for i in range(1, 151)]),
    "fake_cot": "Let me think step by step.\n" + "First, I consider the problem.\n" * 60,
    "alternating": " ".join(["Yes." if i % 2 == 0 else "No." for i in range(200)]),
    "word_spam": "calculate " * 300,
    "template_fill": " ".join([f"The value of x is {i}." for i in range(100)]),
    "nested_rep": ("Beginning. " + "Middle part. " * 5 + "End. ") * 20,
    "unicode_abuse": "∴ Therefore ∴ " * 100,
    "whitespace_pad": "Answer: 42" + "   " * 500 + "Final: 42",
    "verify_loop": "Let me verify: 42. Verified. " * 60,
    "question_echo": "You asked about math. Math is important. This is about math. " * 40,
}

GOOD_EXAMPLES = {
    "math_reasoning": """
To solve this problem, I'll use the quadratic formula.
Given: ax² + bx + c = 0 where a=1, b=-5, c=6

The quadratic formula is: x = (-b ± √(b²-4ac)) / 2a

Substituting values:
x = (5 ± √(25-24)) / 2
x = (5 ± 1) / 2

Therefore: x = 3 or x = 2

Let me verify:
For x=3: 9 - 15 + 6 = 0 ✓
For x=2: 4 - 10 + 6 = 0 ✓

The solutions are x = 2 and x = 3.
""",
    "code_explanation": """
This function implements binary search in O(log n) time.

def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

The algorithm works by repeatedly halving the search space.
Each iteration eliminates half of the remaining elements.
This gives us logarithmic time complexity.
""",
    "diverse_text": """
Machine learning has revolutionized many fields. Computer vision now powers
autonomous vehicles and medical imaging. Natural language processing enables
translation and chatbots. Reinforcement learning achieved superhuman performance
in games like Go and StarCraft. These advances stem from deep neural networks,
massive datasets, and increased compute. However, challenges remain: models can
be brittle, biased, or computationally expensive. Research continues on efficiency,
interpretability, and robustness. The field moves rapidly, with new architectures
and training methods emerging regularly.
""",
    "technical_doc": """
API Reference: UserAuthentication Module

authenticate(username: str, password: str) -> Token
    Validates credentials against the database.
    Returns JWT token on success, raises AuthError on failure.

refresh_token(token: Token) -> Token
    Exchanges valid token for new one with extended expiry.
    Original token is invalidated after exchange.

logout(token: Token) -> None
    Invalidates the token immediately.
    Clears associated session data from cache.

Rate limits: 10 requests/minute per IP for auth endpoints.
Token expiry: 1 hour for access, 7 days for refresh.
""",
    "narrative": """
The researcher stared at the terminal, watching loss curves descend. Three months
of work had led to this moment. The model was finally converging. She grabbed her
coffee—cold now—and checked the validation metrics. Accuracy was climbing steadily.
Her colleague walked over, curious about the late night. "It's working," she said
simply. They watched together as the numbers ticked upward, each epoch bringing
them closer to their goal. By morning, they had a breakthrough.
""",
    "legit_steps": """
Problem: Find the shortest path in a weighted graph.

Analysis:
- Graph has positive weights only → Dijkstra's algorithm applies
- Need to track visited nodes and tentative distances
- Priority queue optimizes node selection

Implementation approach:
1. Initialize distances: source=0, others=infinity
2. Use min-heap for efficient minimum extraction
3. For each node, relax edges to unvisited neighbors
4. Continue until destination reached or heap empty

Time complexity: O((V+E) log V) with binary heap
Space complexity: O(V) for distance array and heap

Edge cases to handle:
- Disconnected graphs (return infinity)
- Self-loops (skip, distance is 0)
- Parallel edges (take minimum weight)
""",
}

BORDERLINE_EXAMPLES = {
    "mild_structure": """
First, let's consider option A. Option A has benefits.
Next, let's consider option B. Option B has benefits.
Then, let's consider option C. Option C has benefits.
Finally, let's consider option D. Option D has benefits.

Comparing all options:
- Option A scores 7/10
- Option B scores 8/10
- Option C scores 6/10
- Option D scores 9/10

The best choice is Option D.
""",
    "list_heavy": """
Requirements for the system:
1. Must handle 1000 requests per second
2. Must maintain 99.9% uptime
3. Must encrypt all data at rest
4. Must support horizontal scaling
5. Must integrate with existing auth
6. Must provide audit logging
7. Must support rollback
8. Must have monitoring
9. Must document APIs
10. Must include tests
""",
}


def run_tests():
    """Test detector against all examples."""
    print("=" * 80)
    print("REPETITION DETECTOR TEST")
    print("=" * 80)

    results = []

    # Test reward hack examples (should score HIGH)
    print("\n[REWARD HACK EXAMPLES] Should detect (score > 0.4):")
    print("-" * 80)
    for name, text in REWARD_HACK_EXAMPLES.items():
        t0 = time.perf_counter()
        score = repetition_score(text)
        elapsed = (time.perf_counter() - t0) * 1000

        status = "PASS" if score > 0.4 else "FAIL"
        print(f"  [{status}] {name:20s} score={score:.3f} ({elapsed:.1f}ms)")
        results.append(('hack', name, score, score > 0.4))

    # Test good examples (should score LOW)
    print("\n[GOOD EXAMPLES] Should NOT detect (score < 0.35):")
    print("-" * 80)
    for name, text in GOOD_EXAMPLES.items():
        t0 = time.perf_counter()
        score = repetition_score(text)
        elapsed = (time.perf_counter() - t0) * 1000

        status = "PASS" if score < 0.35 else "FAIL"
        print(f"  [{status}] {name:20s} score={score:.3f} ({elapsed:.1f}ms)")
        results.append(('good', name, score, score < 0.35))

    # Test borderline examples
    print("\n[BORDERLINE] Mild repetition (score 0.15-0.55):")
    print("-" * 80)
    for name, text in BORDERLINE_EXAMPLES.items():
        t0 = time.perf_counter()
        score = repetition_score(text)
        elapsed = (time.perf_counter() - t0) * 1000

        status = "PASS" if 0.15 < score < 0.55 else "NOTE"
        print(f"  [{status}] {name:20s} score={score:.3f} ({elapsed:.1f}ms)")
        results.append(('borderline', name, score, True))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    hack_correct = sum(1 for r in results if r[0] == 'hack' and r[3])
    hack_total = sum(1 for r in results if r[0] == 'hack')
    good_correct = sum(1 for r in results if r[0] == 'good' and r[3])
    good_total = sum(1 for r in results if r[0] == 'good')

    print(f"Reward hack detection: {hack_correct}/{hack_total} ({100*hack_correct/hack_total:.0f}%)")
    print(f"Good text (no false positives): {good_correct}/{good_total} ({100*good_correct/good_total:.0f}%)")

    # Performance test
    print("\n" + "=" * 80)
    print("PERFORMANCE")
    print("=" * 80)

    long_text = "The plan is final. " * 1000
    iterations = 100

    t0 = time.perf_counter()
    for _ in range(iterations):
        repetition_score(long_text)
    elapsed = (time.perf_counter() - t0) * 1000 / iterations

    print(f"Long text ({len(long_text)} chars): {elapsed:.2f}ms avg over {iterations} iterations")

    # Test the API
    print("\n" + "=" * 80)
    print("API TEST")
    print("=" * 80)

    test_text = "spam spam spam " * 50
    print(f"is_repetitive(spam x50): {is_repetitive(test_text)}")
    print(f"apply_repetition_penalty(spam, 1.0): {apply_repetition_penalty(test_text, 1.0)}")
    print(f"apply_repetition_penalty(spam, 1.0, 'strict'): {apply_repetition_penalty(test_text, 1.0, 'strict')}")

    return hack_correct == hack_total and good_correct == good_total


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
