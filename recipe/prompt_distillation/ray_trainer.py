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
On-policy prompt distillation trainer.

Extends the GKD OnPolicyDistillTrainer with system prompt injection for the teacher.
The student generates responses without the system prompt, while the teacher provides
top-k logprobs with the system prompt prepended. The KL divergence loss trains the
student to internalize the system prompt's knowledge into its weights.

Key differences from GKD:
- Constructor tokenizes and stores the system prompt
- _async_get_teacher_knowledge() prepends system prompt tokens to teacher queries
- All other logic (schedulers, fit(), init_workers(), weight sync) is inherited
"""

from typing import Optional

from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler

from recipe.gkd.ray_trainer import GenerationBatchFuture, OnPolicyDistillTrainer
from recipe.prompt_distillation.teacher_utils import get_teacher_knowledge_with_system_prompt
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

WorkerType = type[Worker]


class PromptDistillTrainer(OnPolicyDistillTrainer):
    """On-policy prompt distillation trainer.

    Extends OnPolicyDistillTrainer to inject a system prompt into teacher queries,
    enabling the student to learn the system prompt's knowledge through KL distillation.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        # Load and tokenize the system prompt
        self.system_prompt_ids = self._load_system_prompt(config, tokenizer)
        print(f"Prompt distillation: system prompt tokenized to {len(self.system_prompt_ids)} tokens")

    @staticmethod
    def _load_system_prompt(config, tokenizer):
        """Load the system prompt and tokenize it for the teacher.

        The system prompt is formatted as a chat message and tokenized so it can
        be prepended to teacher queries as raw token IDs.
        """
        prompt_distill_config = OmegaConf.select(config, "prompt_distillation")
        if prompt_distill_config is None:
            raise ValueError(
                "prompt_distillation config section is required. "
                "Set prompt_distillation.system_prompt_path or prompt_distillation.system_prompt"
            )

        system_prompt_path = prompt_distill_config.get("system_prompt_path")
        system_prompt = prompt_distill_config.get("system_prompt")

        if system_prompt_path is not None:
            with open(system_prompt_path) as f:
                system_prompt_text = f.read().strip()
        elif system_prompt is not None:
            system_prompt_text = system_prompt
        else:
            raise ValueError(
                "Either prompt_distillation.system_prompt_path or "
                "prompt_distillation.system_prompt must be provided"
            )

        # Tokenize the system prompt as a chat-formatted system message.
        # We use apply_chat_template to get the properly formatted prefix
        # that includes the system message tokens.
        system_messages = [{"role": "system", "content": system_prompt_text}]
        system_prompt_str = tokenizer.apply_chat_template(
            system_messages, add_generation_prompt=False, tokenize=False
        )
        system_prompt_ids = tokenizer.encode(system_prompt_str, add_special_tokens=False)

        return system_prompt_ids

    def _async_get_teacher_knowledge(self, future: GenerationBatchFuture):
        """Asynchronously obtain teacher knowledge with system prompt prepended.

        Overrides the GKD method to inject system prompt tokens into the teacher
        query. The teacher sees [system_prompt + student_input], while the student
        only sees [student_input]. The returned logprobs are aligned to the student's
        token positions (system prompt positions are stripped).
        """
        _, _, gen_batch_output = future.get()
        gen_batch_output.meta_info["response_length"] = self.config.data.max_response_length

        future.set_teacher_batch_output(
            get_teacher_knowledge_with_system_prompt(
                gen_batch_output,
                self.teacher_client,
                system_prompt_ids=self.system_prompt_ids,
                n_server_workers=self.n_server_workers,
                is_async=True,
            )
        )
        return future
