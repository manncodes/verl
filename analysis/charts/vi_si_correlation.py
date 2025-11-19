"""V_i vs S_i correlation scatter plot."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_vi_si_correlation(vi_si_data: pd.DataFrame) -> go.Figure:
    """Create a scatter plot showing correlation between V_i and S_i over iterations.

    Args:
        vi_si_data: DataFrame from compute_vi_si_correlation()

    Returns:
        Plotly Figure object
    """
    # Color by iteration to show temporal evolution
    df = vi_si_data.copy()

    # Create scatter plot
    fig = px.scatter(
        df,
        x='V_i',
        y='S_i',
        color='iteration',
        color_continuous_scale='Viridis',
        hover_data=['prompt_hash', 'reward_case'],
        labels={
            'V_i': 'V_i (Instruction Accuracy)',
            'S_i': 'S_i (Preference Score)',
            'iteration': 'Iteration',
        },
    )

    # Add threshold line for S_i (alpha = 0.5)
    alpha_threshold = df['iteration'].map(lambda x: 0.5)  # Assuming constant threshold
    if len(df) > 0:
        alpha = 0.5  # Default from your code

        fig.add_hline(
            y=alpha,
            line_dash='dash',
            line_color='red',
            opacity=0.7,
            annotation_text=f'α threshold = {alpha}',
            annotation_position='right',
        )

    # Add quadrant lines
    fig.add_vline(
        x=0.5,
        line_dash='dot',
        line_color='gray',
        opacity=0.3,
    )

    # Compute correlation coefficient per iteration
    correlations = df.groupby('iteration').apply(
        lambda g: g['V_i'].corr(g['S_i']) if len(g) > 1 else 0
    )

    # Add annotation with correlation info
    latest_iter = df['iteration'].max()
    latest_corr = correlations.get(latest_iter, 0)

    annotations = [
        dict(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=(
                f'<b>Latest Correlation (Iter {latest_iter}):</b> {latest_corr:.3f}<br>'
                f'<b>Interpretation:</b><br>'
                f'• High V_i + High S_i → Bonus (green quadrant)<br>'
                f'• High V_i + Low S_i → Penalty (orange quadrant)'
            ),
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
            align='left',
            xanchor='left',
            yanchor='top',
        )
    ]

    fig.update_layout(
        title="V_i vs S_i Correlation Analysis<br><sub>Shows if instruction following and quality are aligned or conflicting</sub>",
        xaxis_title="V_i (Instruction Accuracy)",
        yaxis_title="S_i (Preference Score)",
        height=700,
        width=1200,
        annotations=annotations,
    )

    return fig
