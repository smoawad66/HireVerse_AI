from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

def calculate_technical_scores(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if not data:
            return {}

        skill_scores = []
        for entry in data:
            skill = entry.get('question_metadata', {}).get('skill')
            overall_score = entry.get('scores', {}).get('overall')
            if skill and overall_score is not None:
                skill_scores.append({'skill': skill, 'overall_score': overall_score})
        
        if not skill_scores:
            return {}

        df = pd.DataFrame(skill_scores)
        technical_scores = (df.groupby('skill')['overall_score'].mean()).round(1).to_dict()
        return technical_scores

    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while calculating technical scores: {e}")
        return {}

def gaze_section(results, df):
    figs = []
    try:
        gaze_counts_dict = results['metrics']['gaze'].get('counts')
        if gaze_counts_dict:
            gaze_df = pd.DataFrame(list(gaze_counts_dict.items()), columns=['Direction', 'Percentage'])
            gaze_df['Percentage'] = pd.to_numeric(gaze_df['Percentage'], errors='coerce').fillna(0)
            fig_pie = px.pie(gaze_df, values='Percentage', names='Direction', title='Gaze Direction Distribution')
            figs.append(dcc.Graph(figure=fig_pie, style={'display': 'inline-block', 'width': '49%'}))
        
        if 'Gaze_Direction' in df.columns:
            fig_gaze_ts = px.scatter(df, y='Gaze_Direction', title='Gaze Direction Over Time')
            fig_gaze_ts.update_layout(yaxis_title='Direction')
            figs.append(dcc.Graph(figure=fig_gaze_ts, style={'display': 'inline-block', 'width': '49%'}))

        cheating_analysis = results.get('gaze_cheating_analysis')
        if cheating_analysis:
            indicator_score = cheating_analysis.get('potential_cheating_indicator', 0)
            assessment = cheating_analysis.get('assessment', 'N/A')

            if indicator_score < 20:
                msg_color, msg = "#28a745", "Very Low: Gaze patterns appear natural and consistent."
            elif indicator_score < 40:
                msg_color, msg = "#4CAF50", "Low: Gaze patterns are mostly natural."
            elif indicator_score < 60:
                msg_color, msg = "#ffc107", "Moderate: Some inconsistencies noted."
            elif indicator_score < 80:
                msg_color, msg = "#fd7e14", "High: Significant deviations observed."
            else:
                msg_color, msg = "#dc3545", "Very High: Strong indicators of unusual behavior."

            simplified_details = html.Div([
                html.H3("Gaze Focus Assessment", style={'textAlign': 'left', 'color': '#333', 'marginBottom': '10px'}),
                html.P([
                    "Your gaze focus during the interview was assessed as: ",
                    html.Span(f"{msg}", style={'fontWeight': 'bold', 'color': msg_color, 'fontSize': '1.2em'})
                ], style={'textAlign': 'left', 'marginBottom': '15px'}),
                html.P("Based on factors like:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Ul([
                    html.Li(f"Time away from screen: {cheating_analysis.get('away_gaze_percentage', 0):.1f}%"),
                    html.Li(f"Gaze shifts per minute: {cheating_analysis.get('gaze_shifts_per_minute', 0):.1f}"),
                    html.Li(f"Average glance duration: {cheating_analysis.get('average_away_glance_duration_seconds', 0):.2f}s")
                ], style={'listStyleType': 'none', 'paddingLeft': '0', 'fontSize': '0.95em', 'color': '#555'})
            ], style={
                'marginTop': '20px',
                'padding': '20px',
                'border': '1px solid #e0e0e0',
                'borderRadius': '8px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                'fontFamily': 'Arial, sans-serif'
            })
            figs.append(simplified_details)

        return html.Div(figs) if figs else html.Div("No gaze data available.")
    except Exception as e:
        print(f"Error rendering gaze section: {e}")
        return html.Div(f"Error: {str(e)}", style={'color': 'red'})

def distance_section(results, df):
    figs = []
    try:
        y_col = 'Distance_cm_Rolling' if 'Distance_cm_Rolling' in df.columns else 'Distance_cm'
        y_label = 'Distance (Rolling, cm)' if 'Distance_cm_Rolling' in df.columns else 'Distance (Raw, cm)'
        if y_col in df.columns:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df[y_col], errors='coerce').dropna(), name=y_label, line=dict(color='purple')))
            if 'Analysis_State' in df.columns:
                unstable_states = ['DISTANCE_UNSTABLE', 'TRACKING_LOST']
                unstable_df_plot = df[df['Analysis_State'].isin(unstable_states)].copy()
                unstable_df_plot[y_col] = pd.to_numeric(unstable_df_plot[y_col], errors='coerce').dropna()
                if not unstable_df_plot.empty:
                    fig_dist.add_trace(go.Scatter(x=unstable_df_plot.index, y=unstable_df_plot[y_col], mode='markers', name='Unstable/Lost', marker=dict(color='red', size=5)))
            fig_dist.update_layout(title=f'{y_label} Over Time', yaxis_title='Distance (cm)')
            figs.append(dcc.Graph(figure=fig_dist, style={'display': 'inline-block', 'width': '65%'}))

        unstable_pct = results.get('metrics', {}).get('distance', {}).get('unstable_pct', 0)
        fig_stability = px.pie(values=[100 - unstable_pct, unstable_pct], names=['Stable', 'Unstable/Lost'], 
                                 title='Tracking Stability', color_discrete_sequence=['#2ca02c', '#d62728'])
        figs.append(dcc.Graph(figure=fig_stability, style={'display': 'inline-block', 'width': '34%'}))
        return html.Div(figs) if figs else html.Div("No distance data available.")
    except Exception as e:
        print(f"Error rendering distance section: {e}")
        return html.Div(f"Error: {str(e)}", style={'color': 'red'})

def headpose_section(results, df):
    figs = []
    try:
        head_pose_counts_dict = results.get('metrics', {}).get('head_pose', {}).get('counts')
        if head_pose_counts_dict:
            head_pose_df = pd.DataFrame(list(head_pose_counts_dict.items()), columns=['Pose', 'Percentage'])
            head_pose_df['Percentage'] = pd.to_numeric(head_pose_df['Percentage'], errors='coerce').fillna(0)
            head_pose_df = head_pose_df.sort_values(by='Percentage', ascending=True)

            fig_bar = px.bar(
                head_pose_df, 
                x='Percentage', 
                y='Pose', 
                orientation='h',
                title='Head Pose Distribution',
                color='Pose',
                color_discrete_map={
                    'Center': 'rgb(75, 192, 192)',   # Greenish
                    'Back': 'rgb(255, 159, 64)',    # Orange
                    'Forward': 'rgb(54, 162, 235)', # Blue
                    'Tilted Right': 'rgb(255, 205, 86)', # Yellow
                    'Tilted Left': 'rgb(153, 102, 255)' # Purple
                }
            )
            fig_bar.update_layout(
                xaxis=dict(
                    title='Percentage',
                    ticksuffix='%',
                    range=[0, 100],
                    gridcolor='#e0e0e0',
                    showgrid=True,
                    showline=False
                ),
                yaxis=dict(
                    title='',
                    showgrid=True,
                    gridcolor='#e0e0e0',
                    showline=False
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            fig_bar.update_traces(marker_line_width=0)
            figs.append(dcc.Graph(figure=fig_bar, style={'width': '100%'}))

        return html.Div(figs) if figs else html.Div("No head pose data available.")
    except Exception as e:
        print(f"Error rendering headpose section: {e}")
        return html.Div(f"Error: {str(e)}", style={'color': 'red'})

def quality_section(results, df):
    figs = []
    try:
        if 'FPS' in df.columns:
            fig_fps = px.line(df, y=pd.to_numeric(df['FPS'], errors='coerce').dropna(), title='Frames Per Second (FPS) Over Time')
            fig_fps.add_hline(y=15, line_dash="dot", annotation_text="Low FPS Threshold", line_color="red")
            fig_fps.update_layout(yaxis_title='FPS')
            figs.append(dcc.Graph(figure=fig_fps))
        return html.Div(figs) if figs else html.Div("No data quality metrics available.")
    except Exception as e:
        print(f"Error rendering quality section: {e}")
        return html.Div(f"Error: {str(e)}", style={'color': 'red'})
