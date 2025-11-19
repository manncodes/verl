"""Plateau detection chart."""

import plotly.graph_objects as go
import pandas as pd


def create_plateau_detection_chart(learning_dynamics: pd.DataFrame) -> go.Figure:
    """Create a chart identifying prompts that have plateaued.

    Args:
        learning_dynamics: DataFrame from compute_learning_dynamics()

    Returns:
        Plotly Figure object
    """
    # Filter to prompts with plateau indicator
    df = learning_dynamics.copy()

    # Sort by recent variance (lowest first - most plateaued)
    df = df.sort_values('recent_variance')

    # Create color based on plateau status
    colors = ['red' if is_plat else 'green' for is_plat in df['is_plateau']]

    # Create hover text
    hover_text = [
        f"Prompt: {row['prompt_hash']}<br>"
        f"Recent Variance: {row['recent_variance']:.4f}<br>"
        f"Final Score: {row['final_score']:.3f}<br>"
        f"Learning Rate: {row['learning_rate']:.4f}<br>"
        f"Plateau: {'Yes' if row['is_plateau'] else 'No'}<br>"
        f"Instructions: {row['instruction_types'][:60]}..."
        for _, row in df.iterrows()
    ]

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['recent_variance'],
        marker=dict(
            color=colors,
            line=dict(width=1, color='white'),
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
    ))

    # Add threshold line
    plateau_threshold = 0.01
    fig.add_hline(
        y=plateau_threshold,
        line_dash='dash',
        line_color='orange',
        annotation_text=f'Plateau Threshold: {plateau_threshold}',
        annotation_position='right',
    )

    # Count plateaued prompts
    num_plateaued = df['is_plateau'].sum()
    total_prompts = len(df)

    fig.update_layout(
        title=f"Plateau Detection<br><sub>{num_plateaued}/{total_prompts} prompts have plateaued (variance < {plateau_threshold})</sub>",
        xaxis_title="Prompt Index (sorted by variance)",
        yaxis_title="Recent Variance (last 3 iterations)",
        height=600,
        width=1200,
        showlegend=False,
    )

    # Add annotation
    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref='paper',
        yref='paper',
        text=(
            f'<b>Plateaued Prompts ({num_plateaued}):</b><br>'
            f'• Low variance in recent iterations<br>'
            f'• Consider removing or upsampling<br>'
            f'• May have reached max performance'
        ),
        showarrow=False,
        font=dict(size=10),
        bgcolor='rgba(255,200,200,0.8)',
        bordercolor='red',
        borderwidth=1,
        align='left',
        xanchor='right',
        yanchor='top',
    )

    return fig
