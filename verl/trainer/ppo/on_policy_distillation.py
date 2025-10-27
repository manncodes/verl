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
On-Policy Distillation (GKD) implementation for VERL.

Based on "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
https://arxiv.org/abs/2306.13649

Key concepts:
- Generalized Knowledge Distillation (GKD): Trains student on its own generations with teacher feedback
- Addresses train-inference distribution mismatch by using on-policy data
- Generalizes JSD with beta parameter to interpolate between forward/reverse KL
- Significantly faster than RL (7-10x) with dense supervision (O(N) bits per episode)
"""

import torch
import torch.nn.functional as F
from typing import Optional

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss, agg_loss


def compute_generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute Generalized Jensen-Shannon Divergence loss.

    GKD uses generalized JSD which interpolates between forward and reverse KL:
    - beta = 0.0: Forward KL (mean-seeking)
    - beta = 1.0: Reverse KL (mode-seeking, recommended for distillation)
    - beta in (0,1): Mixture of both

    Args:
        student_logits: Student model logits, shape (batch, seq_len, vocab_size)
        teacher_logits: Teacher model logits, shape (batch, seq_len, vocab_size)
        beta: Interpolation parameter between 0 and 1
        temperature: Temperature for softmax (lower = sharper distribution)

    Returns:
        per_token_loss: Loss for each token, shape (batch, seq_len)
    """
    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Compute log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Compute probabilities for mixture
    student_probs = torch.exp(student_log_probs)
    teacher_probs = torch.exp(teacher_log_probs)

    # Mixture distribution: M = beta * teacher + (1-beta) * student
    mixture_probs = beta * teacher_probs + (1 - beta) * student_probs
    mixture_log_probs = torch.log(mixture_probs + 1e-10)  # numerical stability

    # Generalized JSD = beta * KL(teacher || mixture) + (1-beta) * KL(student || mixture)
    # For distillation loss, we want: KL(student || mixture)
    # This reduces to: sum_x student_probs * (log(student_probs) - log(mixture_probs))
    per_token_loss = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)

    return per_token_loss


def compute_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute Reverse KL divergence: KL(student || teacher).

    Reverse KL is mode-seeking - student focuses on modes of teacher distribution.
    This is the same as GKD with beta=1.0.

    Equivalent to: sum_x student(x) * log(student(x) / teacher(x))

    Args:
        student_logits: Student model logits, shape (batch, seq_len, vocab_size)
        teacher_logits: Teacher model logits, shape (batch, seq_len, vocab_size)
        temperature: Temperature for softmax

    Returns:
        per_token_loss: Loss for each token, shape (batch, seq_len)
    """
    # Apply temperature
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Compute log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Reverse KL: sum over vocabulary dimension
    per_token_loss = F.kl_div(
        teacher_log_probs,  # Note: pytorch's kl_div expects log-probs as first arg
        student_log_probs,
        reduction='none',
        log_target=True
    ).sum(dim=-1)

    return per_token_loss


def compute_forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute Forward KL divergence: KL(teacher || student).

    Forward KL is mean-seeking - student tries to cover all modes of teacher.
    This is the same as standard distillation loss or GKD with beta=0.0.

    Equivalent to: sum_x teacher(x) * log(teacher(x) / student(x))

    Args:
        student_logits: Student model logits, shape (batch, seq_len, vocab_size)
        teacher_logits: Teacher model logits, shape (batch, seq_len, vocab_size)
        temperature: Temperature for softmax

    Returns:
        per_token_loss: Loss for each token, shape (batch, seq_len)
    """
    # Apply temperature
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Compute log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Forward KL: sum over vocabulary dimension
    per_token_loss = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction='none',
        log_target=True
    ).sum(dim=-1)

    return per_token_loss


@register_policy_loss("on_policy_distillation")
def compute_on_policy_distillation_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[dict] = None,
    rollout_log_probs: torch.Tensor | None = None,
    teacher_logits: Optional[torch.Tensor] = None,
    student_logits: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute on-policy distillation loss using GKD.

    This implements Generalized Knowledge Distillation (GKD) which trains the student
    on its self-generated sequences with teacher feedback. This addresses the
    train-inference distribution mismatch and provides much denser supervision than RL.

    Key differences from RL:
    - RL: O(1) bits per episode (sparse reward at end)
    - GKD: O(N) bits per episode (dense supervision at every token)
    - Result: 7-10x faster convergence than RL

    Args:
        old_log_prob: Not used for distillation (kept for API compatibility)
        log_prob: Not used for distillation (kept for API compatibility)
        advantages: Not used for distillation (kept for API compatibility)
        response_mask: Mask for valid tokens, shape (batch, seq_len)
        loss_agg_mode: How to aggregate loss ("token-mean", "seq-mean-token-sum", etc.)
        config: Configuration dict with keys:
            - distillation_beta: Beta parameter for GKD (0.0=forward KL, 1.0=reverse KL)
            - distillation_temperature: Temperature for softmax (default 1.0)
            - distillation_type: "gkd", "reverse_kl", or "forward_kl"
        rollout_log_probs: Not used for distillation
        teacher_logits: Teacher model logits, shape (batch, seq_len, vocab_size)
        student_logits: Student model logits, shape (batch, seq_len, vocab_size)

    Returns:
        loss: Scalar distillation loss
        clipfrac: Always 0.0 (no clipping in distillation)
        kl: Approximate KL divergence between student and teacher
        clipfrac_lower: Always 0.0 (no clipping in distillation)

    Raises:
        ValueError: If teacher_logits or student_logits are not provided
    """
    if teacher_logits is None or student_logits is None:
        raise ValueError(
            "On-policy distillation requires both teacher_logits and student_logits. "
            "Make sure these are passed during training."
        )

    # Get config parameters
    beta = config.get("distillation_beta", 1.0) if config else 1.0
    temperature = config.get("distillation_temperature", 1.0) if config else 1.0
    distillation_type = config.get("distillation_type", "reverse_kl") if config else "reverse_kl"

    # Compute per-token loss based on distillation type
    if distillation_type == "gkd":
        per_token_loss = compute_generalized_jsd_loss(
            student_logits, teacher_logits, beta=beta, temperature=temperature
        )
    elif distillation_type == "reverse_kl":
        per_token_loss = compute_reverse_kl_loss(
            student_logits, teacher_logits, temperature=temperature
        )
    elif distillation_type == "forward_kl":
        per_token_loss = compute_forward_kl_loss(
            student_logits, teacher_logits, temperature=temperature
        )
    else:
        raise ValueError(
            f"Unknown distillation_type: {distillation_type}. "
            f"Supported types: 'gkd', 'reverse_kl', 'forward_kl'"
        )

    # Apply mask and aggregate
    masked_loss = per_token_loss * response_mask
    loss = agg_loss(loss_mat=masked_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # Compute approximate KL for logging (using reverse KL as metric)
    with torch.no_grad():
        approx_kl = compute_reverse_kl_loss(student_logits, teacher_logits, temperature=1.0)
        masked_kl = approx_kl * response_mask
        mean_kl = verl_F.masked_mean(masked_kl, response_mask)

    # Return compatible tuple (loss, clipfrac=0, kl, clipfrac_lower=0)
    clipfrac = torch.tensor(0.0, device=loss.device)
    clipfrac_lower = torch.tensor(0.0, device=loss.device)

    return loss, clipfrac, mean_kl, clipfrac_lower


@register_policy_loss("hybrid_rl_distillation")
def compute_hybrid_rl_distillation_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[dict] = None,
    rollout_log_probs: torch.Tensor | None = None,
    teacher_logits: Optional[torch.Tensor] = None,
    student_logits: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute hybrid loss combining RL and distillation.

    This combines policy gradient (RL) with knowledge distillation, allowing
    the model to learn from both rewards and teacher knowledge.

    Total loss = lambda_rl * RL_loss + lambda_distill * Distillation_loss

    Args:
        Same as compute_on_policy_distillation_loss
        Additional config keys:
            - hybrid_lambda_rl: Weight for RL loss (default 1.0)
            - hybrid_lambda_distill: Weight for distillation loss (default 1.0)

    Returns:
        loss: Combined RL + distillation loss
        clipfrac: From RL component
        kl: Combined KL (RL KL + distillation KL)
        clipfrac_lower: From RL component
    """
    if teacher_logits is None or student_logits is None:
        raise ValueError("Hybrid training requires both teacher_logits and student_logits.")

    # Get hybrid weights
    lambda_rl = config.get("hybrid_lambda_rl", 1.0) if config else 1.0
    lambda_distill = config.get("hybrid_lambda_distill", 1.0) if config else 1.0

    # Import vanilla RL loss (you can also use other RL losses)
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    # Compute RL loss
    rl_loss, clipfrac, rl_kl, clipfrac_lower = compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_log_probs
    )

    # Compute distillation loss
    distill_loss, _, distill_kl, _ = compute_on_policy_distillation_loss(
        old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config,
        rollout_log_probs, teacher_logits, student_logits
    )

    # Combine losses
    total_loss = lambda_rl * rl_loss + lambda_distill * distill_loss
    combined_kl = lambda_rl * rl_kl + lambda_distill * distill_kl

    return total_loss, clipfrac, combined_kl, clipfrac_lower
