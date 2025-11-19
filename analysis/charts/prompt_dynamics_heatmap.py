"""Prompt-level training dynamics heatmap."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Literal


def create_prompt_dynamics_heatmap(
    prompt_metrics: pd.DataFrame,
    learning_dynamics: pd.DataFrame,
    metric: Literal['mean_score', 'mean_V_i', 'mean_S_i', 'prompt_strict_acc'] = 'mean_score',
    sort_by: Literal['first_success', 'final_score', 'learning_rate'] = 'first_success'
) -> go.Figure:
    """Create a heatmap showing prompt learning dynamics over iterations.

    Args:
        prompt_metrics: DataFrame from compute_prompt_level_metrics()
        learning_dynamics: DataFrame from compute_learning_dynamics()
        metric: Which metric to visualize
        sort_by: How to sort prompts (rows)

    Returns:
        Plotly Figure object
    """
    # Pivot data to create matrix (prompts x iterations)
    pivot = prompt_metrics.pivot(
        index='prompt_hash',
        columns='iteration',
        values=metric
    )

    # Sort prompts based on learning dynamics
    if sort_by == 'first_success':
        # Sort by first success iteration (earliest learners first)
        sort_order = learning_dynamics.sort_values('first_success_iter', na_position='last')['prompt_hash']
    elif sort_by == 'final_score':
        sort_order = learning_dynamics.sort_values('final_score', ascending=False)['prompt_hash']
    elif sort_by == 'learning_rate':
        sort_order = learning_dynamics.sort_values('learning_rate', ascending=False)['prompt_hash']
    else:
        sort_order = pivot.index

    # Reorder rows
    pivot = pivot.reindex(sort_order)

    # Create hover text with detailed info
    hover_text = []
    for idx in pivot.index:
        row_hover = []
        for col in pivot.columns:
            prompt_data = prompt_metrics[
                (prompt_metrics['prompt_hash'] == idx) &
                (prompt_metrics['iteration'] == col)
            ]

            if len(prompt_data) > 0:
                row = prompt_data.iloc[0]
                text = (
                    f"Prompt: {idx}<br>"
                    f"Iteration: {col}<br>"
                    f"Score: {row['mean_score']:.3f} ± {row['std_score']:.3f}<br>"
                    f"V_i: {row['mean_V_i']:.3f}<br>"
                    f"S_i: {row['mean_S_i']:.3f}<br>"
                    f"Prompt Acc: {row['prompt_strict_acc']:.0%}<br>"
                    f"Instructions: {row['instruction_types'][:50]}..."
                )
            else:
                text = f"Prompt: {idx}<br>Iteration: {col}<br>No data"

            row_hover.append(text)
        hover_text.append(row_hover)

    # Choose colorscale based on metric
    if metric == 'prompt_strict_acc':
        colorscale = [[0, 'rgb(239,85,59)'], [1, 'rgb(99,190,123)']]
        zmin, zmax = 0, 1
    elif metric in ['mean_score', 'mean_V_i']:
        colorscale = 'RdYlGn'
        zmin, zmax = -0.5, 2.0 if metric == 'mean_score' else 1.0
    else:  # mean_S_i
        colorscale = 'Viridis'
        zmin, zmax = 0, 1

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Iter {i}" for i in pivot.columns],
        y=[f"{i[:8]}..." for i in pivot.index],  # Truncate hash for display
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=metric.replace('_', ' ').title()),
    ))

    # Update layout
    title_map = {
        'mean_score': 'F_i (Final Reward)',
        'mean_V_i': 'V_i (Instruction Accuracy)',
        'mean_S_i': 'S_i (Preference Score)',
        'prompt_strict_acc': 'Prompt Strict Accuracy'
    }

    fig.update_layout(
        title=f"Prompt Training Dynamics: {title_map[metric]}<br><sub>Sorted by {sort_by.replace('_', ' ').title()}</sub>",
        xaxis_title="Iteration",
        yaxis_title="Prompt (hash)",
        height=max(600, len(pivot) * 20),  # Dynamic height based on number of prompts
        width=1200,
        font=dict(size=10),
    )

    return fig
