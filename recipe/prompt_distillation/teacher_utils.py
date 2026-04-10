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
Teacher knowledge retrieval with system prompt injection for prompt distillation.

This module extends the GKD teacher utilities to support prompt distillation:
the teacher model receives input sequences WITH a system prompt prepended,
while the student model only sees the original input. The teacher's logprobs
for the system prompt tokens are discarded so the KL loss is computed only
on the tokens the student is expected to produce.
"""

import time
from types import SimpleNamespace

import torch

from verl import DataProto

teacher_topk_logps_padded, teacher_topk_indices_padded = None, None


def get_teacher_knowledge_with_system_prompt(
    batch: DataProto,
    teacher_client,
    system_prompt_ids: list[int],
    n_server_workers=1,
    is_async=False,
):
    """
    Retrieve teacher model's top-k predictions with system prompt prepended.

    The teacher sees: [system_prompt_ids] + [student_input_ids]
    The returned logprobs cover only the student's token positions (system prompt
    positions are stripped), so the KL loss aligns with the student's sequence.

    Args:
        batch (DataProto): Input batch containing input_ids and attention_mask
            from the student's rollout (no system prompt).
        teacher_client: Client for communicating with the teacher model server.
        system_prompt_ids (list[int]): Tokenized system prompt to prepend for the teacher.
        n_server_workers (int): Number of parallel workers for teacher model inference.
        is_async (bool): Whether to use asynchronous processing.

    Returns:
        If is_async=True: SimpleNamespace with get() method to process futures.
        If is_async=False: Processed DataProto containing teacher knowledge.
    """
    system_prompt_len = len(system_prompt_ids)

    # Extract valid token IDs from student sequences (removing padding)
    student_input_ids = []
    attention_mask = batch.batch["attention_mask"].to(torch.bool)

    for ids, mask in zip(batch.batch["input_ids"], attention_mask, strict=False):
        valid_ids = ids[mask].tolist()
        student_input_ids.append(valid_ids)

    # Prepend system prompt to each sequence for the teacher
    teacher_input_ids = []
    for valid_ids in student_input_ids:
        teacher_seq = system_prompt_ids + valid_ids
        teacher_input_ids.append(teacher_seq)

    all_teacher_topk_logps = []
    all_teacher_topk_indices = []

    batch_size = len(teacher_input_ids)
    assert batch_size % n_server_workers == 0
    micro_batch_size = batch_size // n_server_workers
    futures = []
    tik1 = time.time()
    tok1 = tik1

    def cb(future):
        nonlocal tok1
        tok1 = max(tok1, time.time())

    for i in range(0, batch_size, micro_batch_size):
        fut = teacher_client.submit(teacher_input_ids[i : i + micro_batch_size])
        fut.add_done_callback(cb)
        futures.append(fut)

    def handle_futures():
        for future in futures:
            try:
                _, teacher_topk_logps, teacher_topk_indices = future.result()
            except Exception as e:
                raise RuntimeError(f"Teacher request failed: {e}") from e

            all_teacher_topk_logps.extend(teacher_topk_logps)
            all_teacher_topk_indices.extend(teacher_topk_indices)

        tik2 = time.time()

        # Strip the system prompt positions from teacher logprobs.
        # The teacher returned logprobs for [system_prompt + student_input],
        # but we only want logprobs aligned with the student's positions.
        stripped_logps = []
        stripped_indices = []
        for logps, indices in zip(all_teacher_topk_logps, all_teacher_topk_indices):
            # logps shape: [teacher_seq_len, topk]
            # Strip the first system_prompt_len positions
            stripped_logps.append(logps[system_prompt_len:])
            stripped_indices.append(indices[system_prompt_len:])

        real_seq_lens = torch.tensor([x.size(0) for x in stripped_logps], dtype=torch.int32)

        topk = stripped_logps[0].size(-1)
        logp_dtype = stripped_logps[0].dtype
        idx_dtype = stripped_indices[0].dtype
        teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]

        global teacher_topk_logps_padded, teacher_topk_indices_padded
        if (
            teacher_topk_logps_padded is None
            or teacher_topk_logps_padded.dtype != logp_dtype
            or teacher_topk_logps_padded.shape != torch.Size(teacher_knowledge_shape)
        ):
            teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype)
        else:
            teacher_topk_logps_padded.zero_()

        if (
            teacher_topk_indices_padded is None
            or teacher_topk_indices_padded.dtype != idx_dtype
            or teacher_topk_indices_padded.shape != torch.Size(teacher_knowledge_shape)
        ):
            teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, dtype=idx_dtype)
        else:
            teacher_topk_indices_padded.zero_()

        # Pad back to student's sequence shape (aligned by attention_mask)
        for i in range(batch_size):
            teacher_topk_logps_padded[i][attention_mask[i]] = stripped_logps[i]
            teacher_topk_indices_padded[i][attention_mask[i]] = stripped_indices[i]

        output_batch = DataProto.from_single_dict(
            data={"real_seq_lens": real_seq_lens},
        )

        output_batch.non_tensor_batch.update(
            {
                "teacher_topk_logps": teacher_topk_logps_padded.numpy(),
                "teacher_topk_indices": teacher_topk_indices_padded.numpy(),
            }
        )

        tok2 = time.time()
        output_batch.meta_info["timing"] = {"get_teacher_knowledge": (tok1 - tik1) + (tok2 - tik2)}

        return output_batch

    if is_async:
        return SimpleNamespace(get=handle_futures)
    else:
        return handle_futures()
