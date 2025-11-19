"""Upsampling candidates table."""

import plotly.graph_objects as go
import pandas as pd


def create_upsampling_candidates_table(upsampling_data: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Create a table showing top candidates for upsampling.

    Args:
        upsampling_data: DataFrame from compute_upsampling_candidates()
        top_n: Number of top candidates to show

    Returns:
        Plotly Figure object
    """
    # Take top N candidates
    df = upsampling_data.head(top_n).copy()

    # Format values for display
    df['upsampling_score'] = df['upsampling_score'].round(3)
    df['final_score'] = df['final_score'].round(3)
    df['learning_rate'] = df['learning_rate'].round(4)

    # Truncate instruction types for display
    df['instruction_types_short'] = df['instruction_types'].apply(
        lambda x: x[:60] + '...' if len(x) > 60 else x
    )

    # Create color coding based on upsampling score
    colors = []
    for score in df['upsampling_score']:
        if score > 2.0:
            colors.append('rgba(239,85,59,0.3)')  # Red - high priority
        elif score > 1.0:
            colors.append('rgba(255,180,50,0.3)')  # Orange - medium priority
        else:
            colors.append('rgba(255,255,200,0.3)')  # Yellow - low priority

    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[
                '<b>Rank</b>',
                '<b>Prompt Hash</b>',
                '<b>Upsampling<br>Score</b>',
                '<b>Final<br>Score</b>',
                '<b>Learning<br>Rate</b>',
                '<b>Iterations<br>to Learn</b>',
                '<b>Forgetting<br>Events</b>',
                '<b>Instruction Types</b>',
            ],
            fill_color='lightgray',
            align='left',
            font=dict(size=11, color='black'),
        ),
        cells=dict(
            values=[
                list(range(1, len(df) + 1)),
                df['prompt_hash'],
                df['upsampling_score'],
                df['final_score'],
                df['learning_rate'],
                df['iterations_to_learn'],
                df['num_forgetting_events'],
                df['instruction_types_short'],
            ],
            fill_color=[colors],
            align='left',
            font=dict(size=10),
            height=30,
        )
    )])

    fig.update_layout(
        title=f"Top {top_n} Upsampling Candidates<br><sub>Prompts ranked by need for more training samples (red = high priority)</sub>",
        height=max(400, len(df) * 35 + 100),
        width=1400,
    )

    return fig
