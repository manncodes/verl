"""Forgetting analysis chart."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_forgetting_analysis(
    prompt_metrics: pd.DataFrame,
    learning_dynamics: pd.DataFrame
) -> go.Figure:
    """Create a heatmap highlighting forgetting events.

    Args:
        prompt_metrics: DataFrame from compute_prompt_level_metrics()
        learning_dynamics: DataFrame from compute_learning_dynamics()

    Returns:
        Plotly Figure object
    """
    # Compute iteration-to-iteration delta for each prompt
    deltas = []

    for prompt_hash in prompt_metrics['prompt_hash'].unique():
        prompt_data = prompt_metrics[prompt_metrics['prompt_hash'] == prompt_hash].sort_values('iteration')

        if len(prompt_data) < 2:
            continue

        # Compute delta in prompt_strict_acc
        for i in range(1, len(prompt_data)):
            prev_row = prompt_data.iloc[i-1]
            curr_row = prompt_data.iloc[i]

            delta = curr_row['prompt_strict_acc'] - prev_row['prompt_strict_acc']

            deltas.append({
                'prompt_hash': prompt_hash,
                'iteration': curr_row['iteration'],
                'delta': delta,
                'prev_acc': prev_row['prompt_strict_acc'],
                'curr_acc': curr_row['prompt_strict_acc'],
                'is_forgetting': delta < -0.5,  # Went from success to failure
            })

    if not deltas:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="Forgetting Analysis<br><sub>No forgetting events detected</sub>")
        return fig

    delta_df = pd.DataFrame(deltas)

    # Pivot to create heatmap
    pivot = delta_df.pivot(
        index='prompt_hash',
        columns='iteration',
        values='delta'
    )

    # Sort by number of forgetting events (most forgotten first)
    forgetting_counts = delta_df[delta_df['is_forgetting']].groupby('prompt_hash').size()
    sort_order = forgetting_counts.sort_values(ascending=False).index.tolist()
    # Add prompts with no forgetting
    remaining = [p for p in pivot.index if p not in sort_order]
    sort_order.extend(remaining)
    pivot = pivot.reindex(sort_order)

    # Create hover text
    hover_text = []
    for idx in pivot.index:
        row_hover = []
        for col in pivot.columns:
            delta_data = delta_df[
                (delta_df['prompt_hash'] == idx) &
                (delta_df['iteration'] == col)
            ]

            if len(delta_data) > 0:
                row = delta_data.iloc[0]
                text = (
                    f"Prompt: {idx}<br>"
                    f"Iteration: {col}<br>"
                    f"Accuracy Change: {row['delta']:.0%}<br>"
                    f"Previous: {row['prev_acc']:.0%} → Current: {row['curr_acc']:.0%}<br>"
                    f"<b>{'FORGETTING EVENT' if row['is_forgetting'] else 'Normal'}</b>"
                )
            else:
                text = f"Prompt: {idx}<br>Iteration: {col}<br>No data"

            row_hover.append(text)
        hover_text.append(row_hover)

    # Create heatmap with diverging colorscale (red = forgetting, green = learning)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Iter {i}" for i in pivot.columns],
        y=[f"{i[:8]}..." for i in pivot.index],
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Accuracy<br>Change'),
    ))

    # Count total forgetting events
    total_forgetting = delta_df['is_forgetting'].sum()

    fig.update_layout(
        title=f"Catastrophic Forgetting Tracker<br><sub>{total_forgetting} forgetting events detected (red cells)</sub>",
        xaxis_title="Iteration",
        yaxis_title="Prompt (hash, sorted by forgetting frequency)",
        height=max(600, len(pivot) * 20),
        width=1200,
    )

    return fig
