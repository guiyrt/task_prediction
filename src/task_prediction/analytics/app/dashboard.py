import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from pathlib import Path
import logging
import json

from ..core.processing import load_and_prep_data, get_duration_frequency_stats, format_seconds
from ..core.callsign_lifecycle import get_callsign_lifecycles, get_sequence_consensus, CallsignLifecycle
from ..visuals.charts import plot_duration_boxplot, plot_1d_run_timeline

logger = logging.getLogger(__name__)

def build_lifecycle_ui(lifecycles: list[CallsignLifecycle], color_map: dict) -> html.Div:
    """Dynamically builds a visual step-matrix using Dash HTML components."""
    if not lifecycles:
        return html.Div("No callsign data available for this run.", className="text-muted")

    rows = []
    for lc in lifecycles:
        steps = []
        
        # 1. Render Lead Time
        if lc.lead_time_sec > 0:
            steps.append(html.Div(
                f"|--- {format_seconds(lc.lead_time_sec)} ---", 
                style={"color": "#adb5bd", "fontSize": "0.85em", "margin": "0 10px", "whiteSpace": "nowrap"}
            ))
            
        # 2. Render Tasks and Inter-Task Gaps
        for i, task in enumerate(lc.tasks):
            # The Task Box
            border_color = color_map.get(task.task.name, "#000") # Extract .name from TaskType enum
            others_text = f" (+{', '.join(task.other_callsigns)})" if task.other_callsigns else ""
            
            steps.append(html.Div([
                html.Strong(task.task.name),
                html.Span(f" {format_seconds(task.duration_sec)}{others_text}", 
                          style={"fontSize": "0.85em", "color": "#6c757d"})
            ], style={
                "border": f"2px solid {border_color}", "borderRadius": "6px", 
                "padding": "4px 8px", "backgroundColor": "#fff", "whiteSpace": "nowrap", 
                "boxShadow": "1px 1px 3px rgba(0,0,0,0.05)"
            }))
            
            # The Gap (if there is a next task)
            if i < len(lc.tasks) - 1:
                next_task = lc.tasks[i+1]
                gap_sec = (next_task.start_time - task.end_time).total_seconds()
                if gap_sec > 0:
                    steps.append(html.Div(
                        f"--- {format_seconds(gap_sec)} ---", 
                        style={"color": "#adb5bd", "fontSize": "0.85em", "margin": "0 10px", "whiteSpace": "nowrap"}
                    ))
                    
        # 3. Render Tail Time
        if lc.tail_time_sec > 0:
            steps.append(html.Div(
                f"--- {format_seconds(lc.tail_time_sec)} ---|", 
                style={"color": "#adb5bd", "fontSize": "0.85em", "margin": "0 10px", "whiteSpace": "nowrap"}
            ))
            
        # 4. Assemble the row for this callsign
        rows.append(dbc.Row([
            dbc.Col(html.H6(lc.callsign, className="mb-0"), width=1, className="d-flex align-items-center text-end pe-3"),
            dbc.Col(html.Div(steps, style={
                "display": "flex",
                "alignItems": "center",
                "overflowX": "auto",
                "padding": "10px 20px",
                "maskImage": "linear-gradient(90deg, transparent, black 5%, black 95%, transparent)",
                "WebkitMaskImage": "linear-gradient(90deg, transparent, black 5%, black 95%, transparent)",
                "scrollbarWidth": "none", 
                "msOverflowStyle": "none",
                }), width=11)
        ], className="border-bottom py-1 flex-nowrap"))
        
    return html.Div(rows, style={"backgroundColor": "#f8f9fa", "padding": "15px", "borderRadius": "8px", "maxHeight": "500px", "overflowY": "auto"})

def create_app(dataset_folder: Path) -> dash.Dash:
    parquet_path = dataset_folder / "labels.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Could not find {parquet_path.name} in {dataset_folder}")
    
    df_master = load_and_prep_data(parquet_path)
    task_types_all = sorted(df_master["task_type"].unique())
    modes_all = sorted(df_master["asa_support_mode"].unique())
    
    # Global Colors
    palette = px.colors.qualitative.Light24 
    GLOBAL_COLOR_MAP = {t: palette[i % len(palette)] for i, t in enumerate(task_types_all)}
    if "IDLE" in GLOBAL_COLOR_MAP:
        GLOBAL_COLOR_MAP["IDLE"] = "#e9ecef" 
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.title = "ATC Analytics"

    # BUILD CUSTOM TASK FILTER WITH "ONLY" BUTTONS
    task_filter_items = [
        dbc.Row([
            dbc.Col(dbc.Button("Select All", id="btn-select-all", size="sm", color="secondary", outline=True, className="w-100")),
        ], className="mb-2")
    ]
    for t in task_types_all:
        task_filter_items.append(
            dbc.Row([
                dbc.Col(
                    dbc.Checkbox(
                        id={'type': 'task-cb', 'task': t}, 
                        label=html.Span([html.Span("⬤", style={"color": GLOBAL_COLOR_MAP[t], "marginRight": "8px"}), t]), 
                        value=True
                    ), width=9, className="d-flex align-items-center"
                ),
                dbc.Col(
                    dbc.Button("Only", id={'type': 'task-only-btn', 'task': t}, size="sm", color="light", className="py-0 border"), 
                    width=3, className="d-flex justify-content-end"
                )
            ], className="mb-1")
        )

    # SIDEBAR
    sidebar = html.Div([
        html.H4("Filters", className="display-6"), html.Hr(),
        
        html.Label("ASA Support Mode:", className="fw-bold"),
        dcc.Checklist(
            id="mode-filter", options=[{"label": f" {m}", "value": m} for m in modes_all],
            value=modes_all, inputClassName="me-2", className="mb-3"
        ),
        
        html.Label("Select Scenarios:", className="fw-bold"),
        dcc.Dropdown(
            id="scenario-filter", options=[{"label": f"Scenario {i}", "value": i} for i in sorted(df_master["scenario_id"].unique())],
            value=df_master["scenario_id"].unique().tolist(), multi=True, className="mb-3"
        ),
        
        html.Label("Select Participants:", className="fw-bold"),
        dcc.Dropdown(
            id="participant-filter", options=[{"label": f"Participant {i}", "value": i} for i in sorted(df_master["participant_id"].unique())],
            value=df_master["participant_id"].unique().tolist(), multi=True, className="mb-3"
        ),
        
        html.Label("Task Types:", className="fw-bold mt-2"),
        html.Div(task_filter_items, style={"maxHeight": "40vh", "overflowY": "auto", "padding": "5px", "overflowX": "hidden"})
    ], style={"padding": "2rem 1rem", "backgroundColor": "#f8f9fa", "height": "100vh", "overflowY": "auto"})

    # MAIN CONTENT
    content = html.Div([
        html.H2("ATC Task Prediction Analytics", className="mb-4 mt-4"),
        dbc.Tabs([
            dbc.Tab(label="Global Statistics", tab_id="tab-stats"),
            dbc.Tab(label="Single Run Deep Dive", tab_id="tab-deep-dive"),
        ], id="tabs", active_tab="tab-deep-dive"),
        html.Div(id="tab-content", className="mt-4"),
    ], style={"padding": "2rem 2rem"})

    app.layout = dbc.Container([dbc.Row([dbc.Col(sidebar, width=3), dbc.Col(content, width=9)])], fluid=True)

    # --- CALLBACK 1: EXCLUSIVE SELECT ("ONLY") LOGIC ---
    @app.callback(
        Output({'type': 'task-cb', 'task': ALL}, 'value'),
        [Input({'type': 'task-only-btn', 'task': ALL}, 'n_clicks'),
         Input("btn-select-all", "n_clicks")],
        State({'type': 'task-cb', 'task': ALL}, 'id'),
        prevent_initial_call=True
    )
    def handle_task_selection(only_clicks, select_all_clicks, checkbox_ids):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "btn-select-all":
            return [True] * len(checkbox_ids) # Set all to True
            
        # An "Only" button was pressed. trigger_id is a JSON string of the dictionary ID
        trigger_dict = json.loads(trigger_id)
        target_task = trigger_dict['task']
        
        # Return True ONLY for the task that matches the button clicked
        return [d['task'] == target_task for d in checkbox_ids]

    # --- CALLBACK 2: MAIN RENDER LOGIC ---
    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "active_tab"), 
         Input("mode-filter", "value"),
         Input("scenario-filter", "value"), 
         Input("participant-filter", "value"),
         Input({'type': 'task-cb', 'task': ALL}, 'value')],
        State({'type': 'task-cb', 'task': ALL}, 'id')
    )
    def render_tab_content(active_tab, selected_modes, selected_scenarios, selected_participants, cb_values, cb_ids):
        # Extract which tasks are checked
        selected_tasks = [cb_ids[i]['task'] for i, is_checked in enumerate(cb_values) if is_checked]
        
        if not selected_scenarios or not selected_participants or not selected_tasks or not selected_modes:
            return html.Div("Please select at least one filter in all categories.", className="alert alert-warning")
            
        dff = df_master[
            (df_master["asa_support_mode"].isin(selected_modes)) &
            (df_master["scenario_id"].isin(selected_scenarios)) & 
            (df_master["participant_id"].isin(selected_participants)) &
            (df_master["task_type"].isin(selected_tasks))
        ]
        
        if dff.empty: return html.Div("No data available for the selected filters.", className="alert alert-danger")

        # ================= TAB 1: GLOBAL STATS =================
        if active_tab == "tab-stats":
            stats_df = get_duration_frequency_stats(dff, ["task_type"])
            target_scenario = selected_scenarios[0]

            if "IDLE" in stats_df["task_type"].values:
                idle_row = stats_df[stats_df["task_type"] == "IDLE"]
                other_rows = stats_df[stats_df["task_type"] != "IDLE"]
                stats_df = pd.concat([idle_row, other_rows]).reset_index(drop=True)

            scen_dff = dff[dff["scenario_id"] == target_scenario]
            consensus_df = get_sequence_consensus(scen_dff)
            consensus_ui = html.Div("No sequence data for this scenario.", className="text-muted")

            if not consensus_df.empty:
                fmt_p = lambda x: ", ".join(f"{p:03d}" for p in x) if x else "None"
                
                consensus_df["task_sequence_str"] = consensus_df["task_sequence"].apply(lambda x: " ➔ ".join(x))
                consensus_df["p_all"] = consensus_df["participants"].apply(fmt_p)
                consensus_df["p_none"] = consensus_df["participants_none"].apply(fmt_p)
                consensus_df["p_adv"] = consensus_df["participants_advisory"].apply(fmt_p)
                consensus_df["p_exe"] = consensus_df["participants_execution"].apply(fmt_p)
                
                display_callsigns = []
                prev_callsign = None
                for c in consensus_df["callsigns"]:
                    if c == prev_callsign:
                        display_callsigns.append("") # Blank for duplicates
                    else:
                        display_callsigns.append(c)
                        prev_callsign = c
                consensus_df["display_callsign"] = display_callsigns
                
                # 3. Build tooltips (Hover on sequence or count to see who did it)
                def format_participants_hover(participants_str: str):
                    return {"value": "**Participants:**\n" + participants_str, "type": "markdown"}


                tooltips = [
                    {
                        "total_count": format_participants_hover(row['p_all']),
                        "count_none": format_participants_hover(row['p_none']),
                        "count_advisory": format_participants_hover(row['p_adv']),
                        "count_execution": format_participants_hover(row['p_exe']),
                    }
                    for _, row in consensus_df.iterrows()
                ]

                final_df = consensus_df[["display_callsign", "total_count", "count_none", "count_advisory", "count_execution", "task_sequence_str"]]

                consensus_ui = dash_table.DataTable(
                    data=final_df.to_dict('records'),
                    columns=[
                        {"name": "Callsign", "id": "display_callsign"},
                        {"name": "Total", "id": "total_count"},
                        {"name": "None", "id": "count_none"},
                        {"name": "Advisory", "id": "count_advisory"},
                        {"name": "Execution", "id": "count_execution"},
                        {"name": "Sequence", "id": "task_sequence_str"},
                    ],
                    tooltip_data=tooltips,
                    tooltip_delay=0,
                    tooltip_duration=None,
                    sort_action="none", # Disabled sorting so the blanking trick doesn't scramble
                    style_table={'height': '500px', 'overflowY': 'auto', 'overflowX': 'auto'}, 
                    style_cell={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid #e9ecef'},
                    style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold', 'borderBottom': '2px solid #dee2e6'},
                    style_data_conditional=[
                        # Bold the callsign column
                        {'if': {'column_id': 'display_callsign'}, 'fontWeight': 'bold', 'backgroundColor': '#fcfcfc'},
                        
                        # Subtle "Hues" for the Mode Columns headers/cells
                        {'if': {'column_id': 'total_count'}, 'textAlign': 'center', 'fontWeight': 'bold'},
                        {'if': {'column_id': 'count_none'}, 'backgroundColor': '#fdfdfd', 'textAlign': 'center'},
                        {'if': {'column_id': 'count_advisory'}, 'backgroundColor': '#f0f7ff', 'textAlign': 'center'},
                        {'if': {'column_id': 'count_execution'}, 'backgroundColor': '#f2faf4', 'textAlign': 'center'},                        
                        
                        # Highlight non-zero counts in mode columns to make them pop
                        {'if': {'column_id': 'count_none', 'filter_query': '{count_none} > 0'}, 'color': '#6c757d', 'fontWeight': 'bold'},
                        {'if': {'column_id': 'count_advisory', 'filter_query': '{count_advisory} > 0'}, 'color': '#004085', 'fontWeight': 'bold'},
                        {'if': {'column_id': 'count_execution', 'filter_query': '{count_execution} > 0'}, 'color': '#155724', 'fontWeight': 'bold'},
                    ]
                )

            return html.Div([
                dcc.Graph(figure=plot_duration_boxplot(dff)),
                html.H5("Aggregated Task Statistics", className="mt-4"),
                dash_table.DataTable(
                    data=stats_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in stats_df.columns],
                    sort_action="native", filter_action="native",
                    style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
                    style_data_conditional=[{'if': {'filter_query': '{task_type} = "IDLE"'}, 'backgroundColor': '#f8f9fa', 'color': '#6c757d'}]
                ),
                html.Hr(className="my-5"),
                html.H4(f"Scenario {target_scenario} Consensus (Callsign Task Sequences)"),
                html.P("Hover over the Sequence or Count columns to see which ATCOs performed that specific sequence.", className="text-muted"),
                consensus_ui
            ])
            
        # ================= TAB 2: DEEP DIVE =================
        elif active_tab == "tab-deep-dive":
            target_scenario = selected_scenarios[0]
            target_participant = selected_participants[0]
            run_df = dff[(dff["scenario_id"] == target_scenario) & (dff["participant_id"] == target_participant)].copy()
            
            if run_df.empty:
                return html.Div("No data found for this specific run.", className="alert alert-danger")

            total_idle_sec = run_df[run_df["task_type"] == "IDLE"]["duration_sec"].sum()
            total_active_sec = run_df[run_df["task_type"] != "IDLE"]["duration_sec"].sum()
            total_time = total_active_sec + total_idle_sec
            active_pct = (total_active_sec / total_time * 100) if total_time > 0 else 0
            
            # Use iloc[0] to get the mode of this specific run
            run_mode = run_df["asa_support_mode"].iloc[0]
            
            kpi_cards = dbc.Row([
                dbc.Col(dbc.Card([html.H6("Participant", className="text-muted"), html.H4(target_participant)], body=True)),
                dbc.Col(dbc.Card([html.H6("Scenario", className="text-muted"), html.H4(target_scenario)], body=True)),
                dbc.Col(dbc.Card([html.H6("Mode", className="text-muted"), html.H4(run_mode)], body=True)),
                dbc.Col(dbc.Card([html.H6("Active Time", className="text-muted"), html.H4(f"{format_seconds(total_active_sec)} ({active_pct:.1f}%)")], body=True)),
                dbc.Col(dbc.Card([html.H6("Idle Time", className="text-muted"), html.H4(f"{format_seconds(total_idle_sec)} ({100-active_pct:.1f}%)")], body=True)),
            ], className="mb-4")

            scen_bounds = df_master[df_master["scenario_id"] == target_scenario]
            lifecycles = get_callsign_lifecycles(run_df, run_df["start_time"].min(), run_df["end_time"].max())
            lifecycle_ui = build_lifecycle_ui(lifecycles, GLOBAL_COLOR_MAP)

            # Stats Table formatting (force IDLE to top)
            run_stats = get_duration_frequency_stats(run_df, ["task_type"])
            if "IDLE" in run_stats["task_type"].values:
                idle_row = run_stats[run_stats["task_type"] == "IDLE"]
                other_rows = run_stats[run_stats["task_type"] != "IDLE"]
                run_stats = pd.concat([idle_row, other_rows]).reset_index(drop=True)

            # Sequence Table formatting
            timeline_fig = plot_1d_run_timeline(run_df, GLOBAL_COLOR_MAP)

            run_df.sort_values("start_time", inplace=True)
            run_start_time = run_df["start_time"].min()
            run_df["rel_start"] = (run_df["start_time"] - run_start_time).dt.total_seconds().apply(format_seconds)
            
            run_df["start_time"] = run_df["start_time"].dt.strftime("%H:%M:%S.%f").str[:-3]
            run_df["end_time"] = run_df["end_time"].dt.strftime("%H:%M:%S.%f").str[:-3]
            run_df["duration_sec"] = run_df["duration_sec"].round(2)
            
            def format_callsigns(c):
                try: return ", ".join(list(c))
                except TypeError: return ""
            run_df["callsigns"] = run_df["callsigns"].apply(format_callsigns)
            run_df = run_df[["rel_start", "start_time", "end_time", "duration_sec", "task_type", "callsigns"]]

            return html.Div([
                kpi_cards,

                html.H5("Entity Lifecycle View (Per Callsign)", className="mt-4 mb-3"),
                lifecycle_ui,
                
                html.Hr(className="my-4"),

                html.H5("Run Statistics"),
                dash_table.DataTable(
                    data=run_stats.to_dict('records'), columns=[{"name": i, "id": i} for i in run_stats.columns],
                    sort_action="native", style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_data_conditional=[{'if': {'filter_query': '{task_type} = "IDLE"'}, 'backgroundColor': '#f8f9fa', 'color': '#6c757d'}]
                ),
                html.Div(dcc.Graph(figure=timeline_fig), className="mt-4 mb-4"),
                html.H5("Detailed Task Sequence"),
                dash_table.DataTable(
                    data=run_df.to_dict('records'),
                    columns=[{"name": c, "id": c} for c in run_df.columns],
                    filter_action="native", sort_action="native", fixed_rows={'headers': True},
                    style_table={'height': '1000px', 'overflowY': 'auto', 'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_data_conditional=[{'if': {'filter_query': '{task_type} = "IDLE"'}, 'backgroundColor': '#f8f9fa', 'color': '#6c757d'}]
                )
            ])

    return app