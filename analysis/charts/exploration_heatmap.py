"""Exploration quality heatmap."""

import plotly.graph_objects as go
import pandas as pd


def create_exploration_heatmap(
    exploration_metrics: pd.DataFrame,
    learning_dynamics: pd.DataFrame
) -> go.Figure:
    """Create a heatmap showing exploration variance across rollouts.

    Args:
        exploration_metrics: DataFrame from compute_exploration_metrics()
        learning_dynamics: DataFrame from compute_learning_dynamics()

    Returns:
        Plotly Figure object
    """
    # Pivot data to create matrix (prompts x iterations)
    pivot = exploration_metrics.pivot(
        index='prompt_hash',
        columns='iteration',
        values='std_score'
    )

    # Sort by first success (same as prompt dynamics)
    sort_order = learning_dynamics.sort_values('first_success_iter', na_position='last')['prompt_hash']
    pivot = pivot.reindex(sort_order)

    # Create hover text
    hover_text = []
    for idx in pivot.index:
        row_hover = []
        for col in pivot.columns:
            exp_data = exploration_metrics[
                (exploration_metrics['prompt_hash'] == idx) &
                (exploration_metrics['iteration'] == col)
            ]

            if len(exp_data) > 0:
                row = exp_data.iloc[0]
                text = (
                    f"Prompt: {idx}<br>"
                    f"Iteration: {col}<br>"
                    f"Score Std Dev: {row['std_score']:.3f}<br>"
                    f"Best-Mean Gap: {row['best_mean_gap']:.3f}<br>"
                    f"Consistency: {row['consistency_score']:.3f}"
                )
            else:
                text = f"Prompt: {idx}<br>Iteration: {col}<br>No data"

            row_hover.append(text)
        hover_text.append(row_hover)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Iter {i}" for i in pivot.columns],
        y=[f"{i[:8]}..." for i in pivot.index],
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale='YlOrRd',  # Low variance = yellow (exploiting), high = red (exploring)
        colorbar=dict(title='Score<br>Std Dev'),
    ))

    fig.update_layout(
        title="Exploration Heatmap<br><sub>Variance across N rollouts - high variance = more exploration</sub>",
        xaxis_title="Iteration",
        yaxis_title="Prompt (hash)",
        height=max(600, len(pivot) * 20),
        width=1200,
    )

    return fig
