"""Instruction hierarchy tree with animated difficulty evolution."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def _parse_instruction_hierarchy(inst_type_metrics: pd.DataFrame) -> pd.DataFrame:
    """Parse instruction types into hierarchical structure.

    Args:
        inst_type_metrics: DataFrame with instruction_type, iteration, success_rate

    Returns:
        DataFrame with columns: id, parent, label, iteration, success_rate, count
    """
    rows = []

    # Get all iterations
    iterations = sorted(inst_type_metrics['iteration'].unique())

    for itr in iterations:
        itr_data = inst_type_metrics[inst_type_metrics['iteration'] == itr]

        # Root node
        rows.append({
            'id': 'All Instructions',
            'parent': '',
            'label': 'All Instructions',
            'iteration': itr,
            'success_rate': itr_data['success_rate'].mean(),
            'count': len(itr_data),
            'level': 0,
        })

        # Track categories
        categories = {}

        for _, row in itr_data.iterrows():
            inst_type = row['instruction_type']

            # Parse category (e.g., "keywords:existence" -> "keywords")
            if ':' in inst_type:
                category, specific = inst_type.split(':', 1)
                category_id = f"{category}:"

                # Add category node if not exists
                if category_id not in categories:
                    categories[category_id] = {
                        'success_rates': [],
                        'counts': []
                    }

                categories[category_id]['success_rates'].append(row['success_rate'])
                categories[category_id]['counts'].append(row.get('num_samples', 1))

                # Add specific instruction node (leaf)
                rows.append({
                    'id': inst_type,
                    'parent': category_id,
                    'label': specific,
                    'iteration': itr,
                    'success_rate': row['success_rate'],
                    'count': row.get('num_samples', 1),
                    'level': 2,
                })
            else:
                # No category, attach directly to root
                rows.append({
                    'id': inst_type,
                    'parent': 'All Instructions',
                    'label': inst_type,
                    'iteration': itr,
                    'success_rate': row['success_rate'],
                    'count': row.get('num_samples', 1),
                    'level': 1,
                })

        # Add category nodes with aggregated difficulty
        for cat_id, cat_data in categories.items():
            # Weighted average by sample count
            weights = np.array(cat_data['counts'])
            rates = np.array(cat_data['success_rates'])
            avg_rate = np.average(rates, weights=weights) if weights.sum() > 0 else rates.mean()

            rows.append({
                'id': cat_id,
                'parent': 'All Instructions',
                'label': cat_id.rstrip(':'),
                'iteration': itr,
                'success_rate': avg_rate,
                'count': len(cat_data['success_rates']),
                'level': 1,
            })

    return pd.DataFrame(rows)


def create_instruction_hierarchy_tree(inst_type_metrics: pd.DataFrame,
                                       chart_type: str = 'sunburst') -> go.Figure:
    """Create animated instruction hierarchy tree showing difficulty evolution.

    Args:
        inst_type_metrics: DataFrame from compute_instruction_type_metrics()
        chart_type: 'sunburst' or 'treemap'

    Returns:
        Plotly Figure with animation slider
    """
    # Parse hierarchy
    hierarchy = _parse_instruction_hierarchy(inst_type_metrics)

    # Get iterations for animation frames
    iterations = sorted(hierarchy['iteration'].unique())

    # Create frames for each iteration
    frames = []
    for itr in iterations:
        frame_data = hierarchy[hierarchy['iteration'] == itr]

        if chart_type == 'sunburst':
            trace = go.Sunburst(
                ids=frame_data['id'],
                labels=frame_data['label'],
                parents=frame_data['parent'],
                values=frame_data['count'],
                marker=dict(
                    colors=frame_data['success_rate'],
                    colorscale='RdYlGn',
                    cmid=0.5,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title='Success<br>Rate',
                        tickformat='.0%',
                    ),
                ),
                text=[
                    f"{row['label']}<br>"
                    f"Success: {row['success_rate']:.1%}<br>"
                    f"Samples: {row['count']}"
                    for _, row in frame_data.iterrows()
                ],
                hovertemplate='<b>%{text}</b><extra></extra>',
                branchvalues='total',
            )
        else:  # treemap
            trace = go.Treemap(
                ids=frame_data['id'],
                labels=frame_data['label'],
                parents=frame_data['parent'],
                values=frame_data['count'],
                marker=dict(
                    colors=frame_data['success_rate'],
                    colorscale='RdYlGn',
                    cmid=0.5,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title='Success<br>Rate',
                        tickformat='.0%',
                    ),
                ),
                text=[
                    f"{row['label']}<br>{row['success_rate']:.1%}"
                    for _, row in frame_data.iterrows()
                ],
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Success Rate: %{color:.1%}<br>'
                    'Samples: %{value}<br>'
                    '<extra></extra>'
                ),
                branchvalues='total',
            )

        frames.append(go.Frame(
            data=[trace],
            name=str(itr),
            layout=go.Layout(
                title_text=f"Instruction Hierarchy - Iteration {itr}"
            )
        ))

    # Create initial figure (first iteration)
    initial_data = hierarchy[hierarchy['iteration'] == iterations[0]]

    if chart_type == 'sunburst':
        fig = go.Figure(
            data=go.Sunburst(
                ids=initial_data['id'],
                labels=initial_data['label'],
                parents=initial_data['parent'],
                values=initial_data['count'],
                marker=dict(
                    colors=initial_data['success_rate'],
                    colorscale='RdYlGn',
                    cmid=0.5,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title='Success<br>Rate',
                        tickformat='.0%',
                    ),
                ),
                text=[
                    f"{row['label']}<br>"
                    f"Success: {row['success_rate']:.1%}<br>"
                    f"Samples: {row['count']}"
                    for _, row in initial_data.iterrows()
                ],
                hovertemplate='<b>%{text}</b><extra></extra>',
                branchvalues='total',
            ),
            frames=frames
        )
    else:  # treemap
        fig = go.Figure(
            data=go.Treemap(
                ids=initial_data['id'],
                labels=initial_data['label'],
                parents=initial_data['parent'],
                values=initial_data['count'],
                marker=dict(
                    colors=initial_data['success_rate'],
                    colorscale='RdYlGn',
                    cmid=0.5,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(
                        title='Success<br>Rate',
                        tickformat='.0%',
                    ),
                ),
                text=[
                    f"{row['label']}<br>{row['success_rate']:.1%}"
                    for _, row in initial_data.iterrows()
                ],
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Success Rate: %{color:.1%}<br>'
                    'Samples: %{value}<br>'
                    '<extra></extra>'
                ),
                branchvalues='total',
            ),
            frames=frames
        )

    # Add animation controls
    fig.update_layout(
        title=dict(
            text=f"Instruction Hierarchy Difficulty Evolution<br><sub>Color: Success Rate (Red=Hard, Green=Easy) | Size: Sample Count</sub>",
            x=0.5,
            xanchor='center',
        ),
        height=800,
        width=1000,
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=300)
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ],
                x=0.1,
                y=-0.05,
                xanchor='left',
                yanchor='top',
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [str(itr)],
                            dict(
                                frame=dict(duration=300, redraw=True),
                                mode='immediate',
                                transition=dict(duration=300)
                            )
                        ],
                        label=f"Iter {itr}",
                        method='animate',
                    )
                    for itr in iterations
                ],
                x=0.1,
                y=-0.15,
                len=0.8,
                xanchor='left',
                yanchor='top',
                currentvalue=dict(
                    font=dict(size=16),
                    prefix='Iteration: ',
                    visible=True,
                    xanchor='right'
                ),
                transition=dict(duration=300),
            )
        ]
    )

    return fig


def create_instruction_hierarchy_sunburst(inst_type_metrics: pd.DataFrame) -> go.Figure:
    """Create animated sunburst chart of instruction hierarchy.

    Args:
        inst_type_metrics: DataFrame from compute_instruction_type_metrics()

    Returns:
        Plotly Figure
    """
    return create_instruction_hierarchy_tree(inst_type_metrics, chart_type='sunburst')


def create_instruction_hierarchy_treemap(inst_type_metrics: pd.DataFrame) -> go.Figure:
    """Create animated treemap chart of instruction hierarchy.

    Args:
        inst_type_metrics: DataFrame from compute_instruction_type_metrics()

    Returns:
        Plotly Figure
    """
    return create_instruction_hierarchy_tree(inst_type_metrics, chart_type='treemap')
