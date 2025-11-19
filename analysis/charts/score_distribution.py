"""Score distribution evolution chart."""

import plotly.graph_objects as go
from typing import Dict, List


def create_score_distribution_evolution(score_distributions: Dict[int, List[float]]) -> go.Figure:
    """Create violin plots showing score distribution evolution over iterations.

    Args:
        score_distributions: Dict from compute_score_distributions()

    Returns:
        Plotly Figure object
    """
    # Create violin plot for each iteration
    fig = go.Figure()

    for iteration in sorted(score_distributions.keys()):
        scores = score_distributions[iteration]

        fig.add_trace(go.Violin(
            y=scores,
            x=[f"Iter {iteration}"] * len(scores),
            name=f"Iter {iteration}",
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightblue',
            opacity=0.6,
            line_color='blue',
            hoverinfo='y',
        ))

    fig.update_layout(
        title="Score Distribution Evolution<br><sub>Shows convergence - tighter distributions = more consistent training</sub>",
        xaxis_title="Iteration",
        yaxis_title="Score (F_i)",
        height=600,
        width=1200,
        showlegend=False,
        violinmode='group',
    )

    # Add zero line for reference
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='gray',
        opacity=0.5,
    )

    return fig
