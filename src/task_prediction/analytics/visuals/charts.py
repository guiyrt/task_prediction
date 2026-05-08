import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_duration_boxplot(df: pd.DataFrame) -> go.Figure:
    if df.empty: return go.Figure()
    fig = px.box(
        df, x="task_type", y="duration_sec", color="asa_support_mode",
        category_orders={"asa_support_mode": ["NONE", "ADVISORY", "EXECUTION"]},
        points="all", hover_data=["participant_id", "scenario_id", "callsigns"],
        title="Task Duration Distribution by Automation Mode"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Duration (Seconds)", xaxis_title="Task Type")
    return fig

def plot_1d_run_timeline(df_run: pd.DataFrame, color_map: dict) -> go.Figure:
    if df_run.empty: return go.Figure()
    df_run["timeline_row"] = "Run Timeline"

    fig = px.timeline(
        df_run, x_start="start_time", x_end="end_time", y="timeline_row", 
        color="task_type", color_discrete_map=color_map, hover_data=["duration_sec", "callsigns"],
        title="Continuous Run Timeline (Drag horizontally to pan/zoom)"
    )
    
    fig.update_layout(
        showlegend=False, height=150, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest", dragmode="pan"
    )
    fig.update_yaxes(visible=False, showticklabels=False, fixedrange=True)
    fig.update_xaxes(fixedrange=False)
    
    return fig