"""Instruction interference matrix."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_instruction_interference_matrix(interference_data: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing which instruction pairs interfere with each other.

    Args:
        interference_data: DataFrame from compute_instruction_interference()

    Returns:
        Plotly Figure object
    """
    if len(interference_data) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="Instruction Interference Matrix<br><sub>No multi-instruction prompts found</sub>")
        return fig

    # Get all unique instruction types
    all_types = set()
    all_types.update(interference_data['inst_type_1'])
    all_types.update(interference_data['inst_type_2'])
    all_types = sorted(all_types)

    # Create symmetric matrix
    matrix = pd.DataFrame(1.0, index=all_types, columns=all_types)  # Default to 1.0 (no interference)

    # Fill in the matrix with joint success rates
    for _, row in interference_data.iterrows():
        inst1, inst2 = row['inst_type_1'], row['inst_type_2']
        rate = row['joint_success_rate']
        matrix.loc[inst1, inst2] = rate
        matrix.loc[inst2, inst1] = rate  # Symmetric

    # Diagonal should be 1.0 (no self-interference)
    for inst in all_types:
        matrix.loc[inst, inst] = 1.0

    # Create hover text
    hover_text = []
    for i, inst1 in enumerate(all_types):
        row_hover = []
        for j, inst2 in enumerate(all_types):
            if i == j:
                text = f"{inst1}<br>(diagonal - no co-occurrence)"
            else:
                # Find co-occurrence data
                pair_data = interference_data[
                    ((interference_data['inst_type_1'] == inst1) & (interference_data['inst_type_2'] == inst2)) |
                    ((interference_data['inst_type_1'] == inst2) & (interference_data['inst_type_2'] == inst1))
                ]

                if len(pair_data) > 0:
                    row = pair_data.iloc[0]
                    text = (
                        f"{inst1} + {inst2}<br>"
                        f"Joint Success Rate: {row['joint_success_rate']:.1%}<br>"
                        f"Co-occurrences: {row['co_occurrence_count']}"
                    )
                else:
                    text = f"{inst1} + {inst2}<br>No co-occurrence"

            row_hover.append(text)
        hover_text.append(row_hover)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=all_types,
        y=all_types,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        colorbar=dict(title='Joint<br>Success<br>Rate'),
    ))

    fig.update_layout(
        title="Instruction Interference Matrix<br><sub>Low values (red) indicate instruction pairs that conflict</sub>",
        xaxis_title="Instruction Type",
        yaxis_title="Instruction Type",
        height=800,
        width=900,
        xaxis=dict(tickangle=-45),
    )

    return fig
