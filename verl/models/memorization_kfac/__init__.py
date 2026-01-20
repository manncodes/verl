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
Memorization K-FAC integration for verl's parallel llama models.

This module provides adapters to use the memorization_kfac library with verl's
custom split llama model implementations (ParallelLlamaForCausalLMRmPad, etc.).

Quick Start:
    # Run the full pipeline from command line:
    python -m verl.models.memorization_kfac.run_kfac_pipeline full \\
        --model meta-llama/Llama-2-7b-hf \\
        --layers 20 24 28 31 \\
        --output_dir ./kfac_output

    # Or use the Python API:
    from verl.models.memorization_kfac import KFACConfig, run_full_pipeline

    config = KFACConfig(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        target_layers=[20, 24, 28, 31],
        variance_ratio=0.9,
    )
    results = run_full_pipeline(config)
"""

from verl.models.memorization_kfac.parallel_kfac import (
    MergedKFACCollector,
    ParallelKFACCollector,
    ParallelKFACTreatment,
)
from verl.models.memorization_kfac.run_kfac_pipeline import (
    KFACConfig,
    analyze_kfac_factors,
    collect_kfac_factors,
    run_analysis,
    run_collection,
    run_full_pipeline,
    run_treatment,
)
from verl.models.memorization_kfac.utils import (
    apply_kfac_to_parallel_model,
    collect_kfac_for_parallel_model,
    get_parallel_mlp_layers,
    merge_kfac_factors,
    split_merged_gate_up_factors,
)

__all__ = [
    # Core classes
    "ParallelKFACCollector",
    "MergedKFACCollector",
    "ParallelKFACTreatment",
    # Configuration
    "KFACConfig",
    # Pipeline functions
    "run_full_pipeline",
    "run_collection",
    "run_analysis",
    "run_treatment",
    "collect_kfac_factors",
    "analyze_kfac_factors",
    # Utilities
    "get_parallel_mlp_layers",
    "merge_kfac_factors",
    "split_merged_gate_up_factors",
    "collect_kfac_for_parallel_model",
    "apply_kfac_to_parallel_model",
]
