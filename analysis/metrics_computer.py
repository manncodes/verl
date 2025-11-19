"""Compute metrics for training analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from .data_loader import RolloutDataLoader


class MetricsComputer:
    """Compute training metrics from rollout data."""

    def __init__(self, loader: RolloutDataLoader):
        """Initialize metrics computer.

        Args:
            loader: RolloutDataLoader instance with loaded data
        """
        self.loader = loader
        self.df = loader.to_dataframe()
        self.prompt_groups = loader.get_prompt_groups()

    def compute_prompt_level_metrics(self) -> pd.DataFrame:
        """Compute metrics aggregated by prompt and iteration.

        Returns:
            DataFrame with columns: prompt_hash, iteration, mean_score, std_score,
                                   mean_V_i, mean_S_i, prompt_strict_acc, etc.
        """
        metrics = []

        for prompt_hash, itr_data in self.prompt_groups.items():
            for itr in sorted(itr_data.keys()):
                rollouts = itr_data[itr]

                scores = [r['score'] for r in rollouts]
                v_is = [r.get('V_i', 0) for r in rollouts]
                s_is = [r.get('S_i', 0) for r in rollouts]
                prompt_accs = [r.get('prompt_strict_acc', 0) for r in rollouts]

                # Get instruction types from first rollout (same for all)
                inst_types = []
                if rollouts and 'gts' in rollouts[0] and 'instruction_id_list' in rollouts[0]['gts']:
                    inst_types = rollouts[0]['gts']['instruction_id_list']

                metrics.append({
                    'prompt_hash': prompt_hash,
                    'iteration': itr,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'max_score': np.max(scores),
                    'min_score': np.min(scores),
                    'mean_V_i': np.mean(v_is),
                    'std_V_i': np.std(v_is),
                    'mean_S_i': np.mean(s_is),
                    'std_S_i': np.std(s_is),
                    'prompt_strict_acc': np.max(prompt_accs),  # Any rollout succeeded
                    'mean_prompt_acc': np.mean(prompt_accs),
                    'num_rollouts': len(rollouts),
                    'instruction_types': ','.join(inst_types),
                    'num_instructions': rollouts[0].get('num_instructions', 0) if rollouts else 0,
                })

        return pd.DataFrame(metrics)

    def compute_learning_dynamics(self) -> pd.DataFrame:
        """Compute learning dynamics per prompt (first success, forgetting, etc.).

        Returns:
            DataFrame with columns: prompt_hash, first_success_iter, num_forgetting_events,
                                   learning_rate, final_score, etc.
        """
        prompt_metrics = self.compute_prompt_level_metrics()
        dynamics = []

        for prompt_hash in prompt_metrics['prompt_hash'].unique():
            prompt_data = prompt_metrics[prompt_metrics['prompt_hash'] == prompt_hash].sort_values('iteration')

            # First success iteration (prompt_strict_acc = 1)
            success_iters = prompt_data[prompt_data['prompt_strict_acc'] == 1]['iteration']
            first_success = success_iters.iloc[0] if len(success_iters) > 0 else None

            # Forgetting events (accuracy drops from 1 to 0)
            accs = prompt_data['prompt_strict_acc'].values
            forgetting_events = sum((accs[i-1] == 1 and accs[i] == 0) for i in range(1, len(accs)))

            # Learning rate (slope of mean_score over iterations)
            iterations = prompt_data['iteration'].values
            scores = prompt_data['mean_score'].values
            if len(iterations) > 1:
                learning_rate = np.polyfit(iterations, scores, 1)[0]
            else:
                learning_rate = 0

            # Plateau detection (variance in last 3 iterations)
            if len(scores) >= 3:
                recent_variance = np.var(scores[-3:])
            else:
                recent_variance = np.var(scores)

            dynamics.append({
                'prompt_hash': prompt_hash,
                'first_success_iter': first_success,
                'iterations_to_learn': first_success if first_success else len(iterations),
                'num_forgetting_events': forgetting_events,
                'learning_rate': learning_rate,
                'final_score': scores[-1],
                'initial_score': scores[0],
                'score_improvement': scores[-1] - scores[0],
                'recent_variance': recent_variance,
                'is_plateau': recent_variance < 0.01,  # Threshold for plateau
                'instruction_types': prompt_data.iloc[0]['instruction_types'],
                'num_instructions': prompt_data.iloc[0]['num_instructions'],
            })

        return pd.DataFrame(dynamics)

    def compute_instruction_type_metrics(self) -> pd.DataFrame:
        """Compute metrics per instruction type and iteration.

        Returns:
            DataFrame with columns: instruction_type, iteration, success_rate,
                                   mean_contribution, etc.
        """
        metrics = []

        for itr in sorted(self.df['iteration'].unique()):
            itr_data = self.df[self.df['iteration'] == itr]

            # Group by instruction type
            inst_type_groups = defaultdict(list)
            for _, row in itr_data.iterrows():
                if row['instruction_types']:
                    for inst_type in row['instruction_types'].split(','):
                        inst_type_groups[inst_type].append(row)

            for inst_type, rows in inst_type_groups.items():
                # Success rate for this instruction type
                v_is = [r['V_i'] for r in rows]
                success_rates = [r['inst_strict_acc'] for r in rows]

                metrics.append({
                    'instruction_type': inst_type,
                    'iteration': itr,
                    'success_rate': np.mean(success_rates),
                    'mean_V_i': np.mean(v_is),
                    'std_V_i': np.std(v_is),
                    'num_samples': len(rows),
                })

        return pd.DataFrame(metrics)

    def compute_reward_case_distribution(self) -> pd.DataFrame:
        """Compute distribution of reward cases per iteration.

        Returns:
            DataFrame with columns: iteration, reward_case, count, percentage
        """
        dist = []
        for itr in sorted(self.df['iteration'].unique()):
            itr_data = self.df[self.df['iteration'] == itr]
            total = len(itr_data)

            case_counts = itr_data['reward_case'].value_counts()
            for case, count in case_counts.items():
                dist.append({
                    'iteration': itr,
                    'reward_case': case,
                    'count': count,
                    'percentage': count / total * 100,
                })

        return pd.DataFrame(dist)

    def compute_exploration_metrics(self) -> pd.DataFrame:
        """Compute exploration metrics (variance across rollouts) per prompt.

        Returns:
            DataFrame with columns: prompt_hash, iteration, score_variance,
                                   best_mean_gap, consistency_score
        """
        prompt_metrics = self.compute_prompt_level_metrics()

        # Add exploration-specific metrics
        prompt_metrics['best_mean_gap'] = prompt_metrics['max_score'] - prompt_metrics['mean_score']
        prompt_metrics['consistency_score'] = 1 / (1 + prompt_metrics['std_score'])  # Higher = more consistent

        return prompt_metrics[['prompt_hash', 'iteration', 'std_score', 'best_mean_gap', 'consistency_score']]

    def compute_vi_si_correlation(self) -> pd.DataFrame:
        """Compute correlation between V_i and S_i over iterations.

        Returns:
            DataFrame with V_i, S_i, iteration, reward_case for scatter plot
        """
        return self.df[['iteration', 'V_i', 'S_i', 'reward_case', 'prompt_hash']].copy()

    def compute_instruction_interference(self) -> pd.DataFrame:
        """Compute instruction interference matrix (which pairs co-occur and succeed/fail).

        Returns:
            DataFrame with columns: inst_type_1, inst_type_2, co_occurrence_count,
                                   joint_success_rate
        """
        interference = []

        # Get all prompts with multiple instructions
        multi_inst_data = self.df[self.df['num_instructions'] > 1]

        inst_pairs = defaultdict(lambda: {'count': 0, 'successes': 0})

        for _, row in multi_inst_data.iterrows():
            if not row['instruction_types']:
                continue

            inst_types = row['instruction_types'].split(',')
            # Generate all pairs
            for i, inst1 in enumerate(inst_types):
                for inst2 in inst_types[i+1:]:
                    pair = tuple(sorted([inst1, inst2]))
                    inst_pairs[pair]['count'] += 1
                    if row['prompt_strict_acc'] == 1:
                        inst_pairs[pair]['successes'] += 1

        for (inst1, inst2), data in inst_pairs.items():
            interference.append({
                'inst_type_1': inst1,
                'inst_type_2': inst2,
                'co_occurrence_count': data['count'],
                'joint_success_rate': data['successes'] / data['count'] if data['count'] > 0 else 0,
            })

        return pd.DataFrame(interference)

    def compute_upsampling_candidates(self, min_iterations: int = 5) -> pd.DataFrame:
        """Identify prompts that should be upsampled (hard, not learning well).

        Args:
            min_iterations: Minimum iterations before considering for upsampling

        Returns:
            DataFrame with prompts ranked by upsampling priority
        """
        dynamics = self.compute_learning_dynamics()

        # Filter prompts with enough iterations
        candidates = dynamics[dynamics['iterations_to_learn'] >= min_iterations].copy()

        # Compute upsampling score (higher = more need for upsampling)
        # Factors: low final score, slow learning rate, not yet successful
        candidates['upsampling_score'] = (
            (1 - candidates['final_score'].clip(0, 1)) * 2 +  # Low final score
            (-candidates['learning_rate'].clip(-1, 1)) * 1 +  # Slow/negative learning
            candidates['num_forgetting_events'] * 0.5  # Forgetting events
        )

        candidates = candidates.sort_values('upsampling_score', ascending=False)

        return candidates[['prompt_hash', 'upsampling_score', 'final_score', 'learning_rate',
                          'iterations_to_learn', 'num_forgetting_events', 'instruction_types']]

    def compute_score_distributions(self) -> Dict[int, List[float]]:
        """Get score distributions per iteration for violin plots.

        Returns:
            {iteration: [scores]}
        """
        distributions = {}
        for itr in sorted(self.df['iteration'].unique()):
            distributions[itr] = self.df[self.df['iteration'] == itr]['score'].tolist()
        return distributions
