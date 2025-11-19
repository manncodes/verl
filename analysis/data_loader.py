"""Data loader for rollout JSONL files."""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd


class RolloutDataLoader:
    """Load and parse rollout data from JSONL files."""

    def __init__(self, rollout_dir: str | Path):
        """Initialize data loader.

        Args:
            rollout_dir: Directory containing {iteration}.jsonl files
        """
        self.rollout_dir = Path(rollout_dir)
        self.iterations_data: Dict[int, List[Dict[str, Any]]] = {}

    def load_iterations(self, iterations: List[int] | None = None) -> None:
        """Load rollout data from specified iterations.

        Args:
            iterations: List of iteration numbers. If None, loads all available.
        """
        if iterations is None:
            # Auto-discover all iteration files
            iteration_files = sorted(self.rollout_dir.glob("*.jsonl"))
            iterations = [int(f.stem) for f in iteration_files]

        for itr in iterations:
            file_path = self.rollout_dir / f"{itr}.jsonl"
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping iteration {itr}")
                continue

            self.iterations_data[itr] = self._load_jsonl(file_path)

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single JSONL file."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    @staticmethod
    def compute_prompt_hash(prompt: str) -> str:
        """Create a stable hash for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()[:12]

    def get_prompt_groups(self) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """Group rollouts by prompt across iterations.

        Returns:
            {prompt_hash: {iteration: [rollouts]}}
        """
        prompt_groups = defaultdict(lambda: defaultdict(list))

        for itr, rollouts in self.iterations_data.items():
            for rollout in rollouts:
                prompt_hash = self.compute_prompt_hash(rollout['input'])
                prompt_groups[prompt_hash][itr].append(rollout)

        return dict(prompt_groups)

    def get_instruction_types(self) -> set:
        """Extract all unique instruction types from the data."""
        instruction_types = set()
        for rollouts in self.iterations_data.values():
            for rollout in rollouts:
                if 'gts' in rollout and 'instruction_id_list' in rollout['gts']:
                    instruction_types.update(rollout['gts']['instruction_id_list'])
        return instruction_types

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all rollout data to a pandas DataFrame."""
        rows = []
        for itr, rollouts in self.iterations_data.items():
            for rollout in rollouts:
                row = {
                    'iteration': itr,
                    'prompt_hash': self.compute_prompt_hash(rollout['input']),
                    'input': rollout['input'][:200],  # Truncate for display
                    'output': rollout['output'][:200],
                    'score': rollout.get('score', 0),
                    'V_i': rollout.get('V_i', 0),
                    'S_i': rollout.get('S_i', 0),
                    'alpha_threshold': rollout.get('alpha_threshold', 0.5),
                    'reward_case': rollout.get('reward_case', 'unknown'),
                    'prompt_strict_acc': rollout.get('prompt_strict_acc', 0),
                    'inst_strict_acc': rollout.get('inst_strict_acc', 0),
                    'num_instructions': rollout.get('num_instructions', 0),
                    'num_followed': rollout.get('num_followed', 0),
                    'format_score': rollout.get('format_score', 0),
                }

                # Add instruction types
                if 'gts' in rollout and 'instruction_id_list' in rollout['gts']:
                    row['instruction_types'] = ','.join(rollout['gts']['instruction_id_list'])
                else:
                    row['instruction_types'] = ''

                rows.append(row)

        return pd.DataFrame(rows)
