"""Reward case evolution over iterations."""

import plotly.graph_objects as go
import pandas as pd


def create_reward_case_evolution(reward_dist: pd.DataFrame) -> go.Figure:
    """Create a stacked area chart showing reward case distribution over time.

    Args:
        reward_dist: DataFrame from compute_reward_case_distribution()

    Returns:
        Plotly Figure object
    """
    # Pivot to get percentages for each case per iteration
    pivot = reward_dist.pivot(
        index='iteration',
        columns='reward_case',
        values='percentage'
    ).fillna(0)

    # Define colors for each reward case
    case_colors = {
        'V_i > 0 and S_i > α (bonus)': 'rgb(99,190,123)',  # Green - good
        'V_i > 0 and S_i ≤ α (penalty)': 'rgb(255,180,50)',  # Orange - warning
        'V_i ≤ 0 (no bonus/penalty)': 'rgb(239,85,59)',  # Red - bad
        'V_i > 0 but judge failed (penalty)': 'rgb(150,150,150)',  # Gray - error
    }

    # Create stacked area chart
    fig = go.Figure()

    for case in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[case],
            name=case,
            mode='lines',
            stackgroup='one',
            fillcolor=case_colors.get(case, 'gray'),
            line=dict(width=0.5, color=case_colors.get(case, 'gray')),
            hovertemplate=(
                f'<b>{case}</b><br>'
                'Iteration: %{x}<br>'
                'Percentage: %{y:.1f}%<br>'
                '<extra></extra>'
            ),
        ))

    fig.update_layout(
        title="Reward Case Distribution Evolution<br><sub>Shows shift from penalties/failures to bonuses during training</sub>",
        xaxis_title="Iteration",
        yaxis_title="Percentage of Rollouts (%)",
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        height=500,
        width=1200,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )

    return fig
