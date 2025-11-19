"""Sample efficiency scatter chart."""

import plotly.graph_objects as go
import pandas as pd


def create_sample_efficiency_chart(learning_dynamics: pd.DataFrame) -> go.Figure:
    """Create a scatter plot showing sample efficiency (iterations to learn vs final score).

    Args:
        learning_dynamics: DataFrame from compute_learning_dynamics()

    Returns:
        Plotly Figure object
    """
    # Color by number of instructions (complexity)
    df = learning_dynamics.copy()

    # Create hover text
    hover_text = [
        f"Prompt: {row['prompt_hash']}<br>"
        f"Iterations to Learn: {row['iterations_to_learn']}<br>"
        f"Final Score: {row['final_score']:.3f}<br>"
        f"Learning Rate: {row['learning_rate']:.4f}<br>"
        f"Score Improvement: {row['score_improvement']:.3f}<br>"
        f"Forgetting Events: {row['num_forgetting_events']}<br>"
        f"Instructions: {row['instruction_types'][:60]}..."
        for _, row in df.iterrows()
    ]

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['iterations_to_learn'],
        y=df['final_score'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['num_instructions'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Num<br>Instructions'),
            line=dict(width=1, color='white'),
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
    ))

    # Add quadrant lines
    median_iters = df['iterations_to_learn'].median()
    median_score = df['final_score'].median()

    fig.add_hline(
        y=median_score,
        line_dash='dash',
        line_color='gray',
        opacity=0.5,
        annotation_text=f'Median Score: {median_score:.2f}',
        annotation_position='right',
    )

    fig.add_vline(
        x=median_iters,
        line_dash='dash',
        line_color='gray',
        opacity=0.5,
        annotation_text=f'Median Iters: {median_iters:.0f}',
        annotation_position='top',
    )

    # Add quadrant labels
    annotations = [
        dict(
            x=median_iters * 0.3,
            y=df['final_score'].max() * 0.95,
            text='<b>Fast Learners</b><br>(Keep in dataset)',
            showarrow=False,
            font=dict(color='green', size=12),
            bgcolor='rgba(99,190,123,0.1)',
        ),
        dict(
            x=df['iterations_to_learn'].max() * 0.9,
            y=df['final_score'].max() * 0.95,
            text='<b>Slow Learners</b><br>(Upsample or curriculum)',
            showarrow=False,
            font=dict(color='orange', size=12),
            bgcolor='rgba(255,180,50,0.1)',
        ),
        dict(
            x=df['iterations_to_learn'].max() * 0.9,
            y=df['final_score'].min() * 1.2,
            text='<b>Hard/Failed</b><br>(Upsample or remove)',
            showarrow=False,
            font=dict(color='red', size=12),
            bgcolor='rgba(239,85,59,0.1)',
        ),
        dict(
            x=median_iters * 0.3,
            y=df['final_score'].min() * 1.2,
            text='<b>Quick Failures</b><br>(Check data quality)',
            showarrow=False,
            font=dict(color='purple', size=12),
            bgcolor='rgba(150,100,200,0.1)',
        ),
    ]

    fig.update_layout(
        title="Sample Efficiency Analysis<br><sub>Iterations to learn vs final score - guide for upsampling/curriculum decisions</sub>",
        xaxis_title="Iterations to First Success",
        yaxis_title="Final Score",
        height=700,
        width=1200,
        annotations=annotations,
    )

    return fig
