"""Instruction type success heatmap."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_instruction_type_heatmap(inst_type_metrics: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing instruction type success rates over iterations.

    Args:
        inst_type_metrics: DataFrame from compute_instruction_type_metrics()

    Returns:
        Plotly Figure object
    """
    # Pivot data to create matrix (instruction types x iterations)
    pivot = inst_type_metrics.pivot(
        index='instruction_type',
        columns='iteration',
        values='success_rate'
    )

    # Sort instruction types by average success rate (hardest first)
    avg_success = pivot.mean(axis=1).sort_values()
    pivot = pivot.reindex(avg_success.index)

    # Compute difficulty tier for each instruction type
    difficulty_tiers = pd.cut(
        avg_success,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Hard', 'Medium', 'Easy']
    )

    # Create hover text
    hover_text = []
    for idx in pivot.index:
        row_hover = []
        for col in pivot.columns:
            inst_data = inst_type_metrics[
                (inst_type_metrics['instruction_type'] == idx) &
                (inst_type_metrics['iteration'] == col)
            ]

            if len(inst_data) > 0:
                row = inst_data.iloc[0]
                text = (
                    f"Instruction: {idx}<br>"
                    f"Iteration: {col}<br>"
                    f"Success Rate: {row['success_rate']:.1%}<br>"
                    f"Mean V_i: {row['mean_V_i']:.3f}<br>"
                    f"Std V_i: {row['std_V_i']:.3f}<br>"
                    f"Samples: {row['num_samples']}<br>"
                    f"Difficulty: {difficulty_tiers[idx]}"
                )
            else:
                text = f"Instruction: {idx}<br>Iteration: {col}<br>No data"

            row_hover.append(text)
        hover_text.append(row_hover)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Iter {i}" for i in pivot.columns],
        y=pivot.index,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        colorbar=dict(title='Success Rate'),
    ))

    # Add annotations for difficulty tiers
    annotations = []
    for i, (inst_type, tier) in enumerate(difficulty_tiers.items()):
        annotations.append(
            dict(
                x=-0.15,
                y=i,
                xref='x',
                yref='y',
                text=f"[{tier}]",
                showarrow=False,
                font=dict(size=8, color='gray'),
                xanchor='right',
            )
        )

    fig.update_layout(
        title="Instruction Type Learning Dynamics<br><sub>Sorted by average success rate (hardest first)</sub>",
        xaxis_title="Iteration",
        yaxis_title="Instruction Type",
        height=max(600, len(pivot) * 25),
        width=1200,
        annotations=annotations,
    )

    return fig
