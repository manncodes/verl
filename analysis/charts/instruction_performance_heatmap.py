"""Simple instruction-level performance heatmap over iterations."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional


def create_instruction_performance_heatmap(
    inst_type_metrics: pd.DataFrame,
    raw_df: pd.DataFrame,
    sort_by: str = 'category'
) -> go.Figure:
    """Create a simple heatmap showing instruction performance over iterations.

    Args:
        inst_type_metrics: DataFrame from compute_instruction_type_metrics()
        raw_df: Raw rollout DataFrame from loader.to_dataframe()
        sort_by: 'category', 'difficulty', or 'alphabetical'

    Returns:
        Plotly Figure
    """
    # Pivot to create matrix (instructions x iterations)
    pivot = inst_type_metrics.pivot(
        index='instruction_type',
        columns='iteration',
        values='success_rate'
    )

    # Sort instructions
    if sort_by == 'category':
        # Sort by category prefix (keywords:, format:, etc.)
        pivot = pivot.reindex(sorted(pivot.index, key=lambda x: (x.split(':')[0], x)))
    elif sort_by == 'difficulty':
        # Sort by average success rate (hardest first)
        avg_success = pivot.mean(axis=1).sort_values()
        pivot = pivot.reindex(avg_success.index)
    else:  # alphabetical
        pivot = pivot.sort_index()

    # Create hover text with detailed info
    hover_text = []
    for inst_type in pivot.index:
        row_hover = []
        for itr in pivot.columns:
            # Get metrics for this instruction at this iteration
            inst_data = inst_type_metrics[
                (inst_type_metrics['instruction_type'] == inst_type) &
                (inst_type_metrics['iteration'] == itr)
            ]

            if len(inst_data) > 0:
                row = inst_data.iloc[0]

                # Get sample rollouts for this instruction at this iteration
                rollout_samples = raw_df[
                    (raw_df['iteration'] == itr) &
                    (raw_df['instruction_types'].str.contains(inst_type, na=False))
                ].head(3)  # Get up to 3 examples

                # Build hover text
                text_parts = [
                    f"<b>{inst_type}</b>",
                    f"Iteration: {itr}",
                    f"Success Rate: {row['success_rate']:.1%}",
                    f"Mean V_i: {row['mean_V_i']:.3f}",
                    f"Samples: {row['num_samples']}",
                    "",
                ]

                # Add example rollouts
                if len(rollout_samples) > 0:
                    text_parts.append("<b>Example Rollouts:</b>")
                    for i, (_, sample) in enumerate(rollout_samples.iterrows(), 1):
                        text_parts.append(
                            f"  {i}. Score: {sample['score']:.2f}, "
                            f"V_i: {sample['V_i']:.2f}, "
                            f"S_i: {sample['S_i']:.2f}"
                        )
                        text_parts.append(f"     Case: {sample['reward_case']}")
                        # Truncate input/output for hover
                        input_preview = sample['input'][:100].replace('\n', ' ')
                        output_preview = sample['output'][:100].replace('\n', ' ')
                        text_parts.append(f"     Input: {input_preview}...")
                        text_parts.append(f"     Output: {output_preview}...")
                        text_parts.append("")

                text = "<br>".join(text_parts)
            else:
                text = f"<b>{inst_type}</b><br>Iteration: {itr}<br>No data"

            row_hover.append(text)
        hover_text.append(row_hover)

    # Determine category boundaries for visual separation
    categories = []
    prev_cat = None
    for inst_type in pivot.index:
        cat = inst_type.split(':')[0] if ':' in inst_type else inst_type
        if cat != prev_cat:
            categories.append(cat)
            prev_cat = cat

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
        colorbar=dict(
            title='Success<br>Rate',
            tickformat='.0%',
        ),
    ))

    # Add horizontal lines to separate categories
    if sort_by == 'category':
        shapes = []
        current_cat = None
        for i, inst_type in enumerate(pivot.index):
            cat = inst_type.split(':')[0] if ':' in inst_type else inst_type
            if cat != current_cat and current_cat is not None:
                shapes.append(
                    dict(
                        type='line',
                        x0=-0.5,
                        x1=len(pivot.columns) - 0.5,
                        y0=i - 0.5,
                        y1=i - 0.5,
                        line=dict(color='black', width=2),
                    )
                )
            current_cat = cat

        fig.update_layout(shapes=shapes)

    # Add category labels on the left
    if sort_by == 'category':
        annotations = []
        current_cat = None
        cat_start = 0
        for i, inst_type in enumerate(pivot.index):
            cat = inst_type.split(':')[0] if ':' in inst_type else inst_type
            if cat != current_cat:
                if current_cat is not None:
                    # Add label for previous category
                    cat_mid = (cat_start + i - 1) / 2
                    annotations.append(
                        dict(
                            x=-0.15,
                            y=cat_mid,
                            xref='x',
                            yref='y',
                            text=f"<b>{current_cat}:</b>",
                            showarrow=False,
                            font=dict(size=10, color='black'),
                            xanchor='right',
                        )
                    )
                current_cat = cat
                cat_start = i

        # Add last category label
        if current_cat is not None:
            cat_mid = (cat_start + len(pivot) - 1) / 2
            annotations.append(
                dict(
                    x=-0.15,
                    y=cat_mid,
                    xref='x',
                    yref='y',
                    text=f"<b>{current_cat}:</b>",
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    xanchor='right',
                )
            )

        fig.update_layout(annotations=annotations)

    # Compute overall statistics
    total_instructions = len(pivot)
    avg_initial = pivot.iloc[:, 0].mean()
    avg_final = pivot.iloc[:, -1].mean()
    improvement = avg_final - avg_initial

    fig.update_layout(
        title=dict(
            text=(
                f"Instruction-Level Performance Over Training<br>"
                f"<sub>{total_instructions} instructions | "
                f"Initial: {avg_initial:.1%} → Final: {avg_final:.1%} "
                f"({improvement:+.1%} improvement)</sub>"
            ),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title="Iteration",
        yaxis_title="Instruction Type",
        height=max(800, len(pivot) * 15),
        width=1400,
        font=dict(size=10),
    )

    return fig
