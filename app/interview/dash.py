import dash
from dash import dcc, html, Input, Output
import pandas as pd
import os, json
from .soft_skills.results_analysis import analyze_interview_performance
from .board import gaze_section, calculate_technical_scores, distance_section, headpose_section, quality_section

BASE_DIR = os.path.dirname(__file__)



def get_file_paths(interview_id):
    """Generate file paths based on interview_id"""
    csv_path = os.path.join(BASE_DIR, f'soft_skills/analysis_metrics/interview-{interview_id}.csv')
    json_path = os.path.join(BASE_DIR, f'technical_skills/analysis_metrics/interview-{interview_id}.json')
    return csv_path, json_path


def create_dash_app(flask_app):
    dash_app = dash.Dash(__name__, server=flask_app, url_base_pathname='/api/interview/dashboard/')
    dash_app.title = "Interview Performance Dashboard"
    
    # Add interview_id store to the layout
    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),  # Add URL component
        dcc.Store(id='interview-id-store'),     # Store for interview_id
        html.H1("Interview Performance Dashboard", 
                style={'textAlign': 'center', 'color': '#2e86ab', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '20px'}),
        
        dcc.Loading(
            id="loading-indicator",
            type="circle",
            children=[
                html.Div(id='main-content-area', children=[
                    html.P("Loading analysis...", 
                            style={'textAlign': 'center', 'marginTop': '30px', 'fontFamily': 'Arial, sans-serif'})
                ])
            ]
        ),
        
        dcc.Store(id='analysis-results-store'),
        dcc.Store(id='processed-dataframe-store'),
        dcc.Store(id='technical-scores-store'),
        dcc.Store(id='dummy-trigger', data=0)
    ])
    
    # Add callback to extract interview_id from URL
    from urllib.parse import urlparse, parse_qs

    @dash_app.callback(
        Output('interview-id-store', 'data'),
        Input('url', 'href')
    )
    def extract_interview_id(href):
        try:
            if href:
                parsed_url = urlparse(href)
                query_params = parse_qs(parsed_url.query)
                interview_id = query_params.get('id', [None])[0]
                if interview_id and interview_id.isdigit():
                    return interview_id
        except:
            pass
        return None
    
    # Update the main callback to use interview_id
    @dash_app.callback(
        Output('main-content-area', 'children'),
        Output('analysis-results-store', 'data'),
        Output('processed-dataframe-store', 'data'),
        Output('technical-scores-store', 'data'),
        Input('dummy-trigger', 'data'),
        Input('interview-id-store', 'data')
    )
    def update_analysis_and_render_page(trigger_value, interview_id):
        if not interview_id:
            error_message = html.Div("Error: No interview ID provided in URL", 
                                   style={'color': 'red', 'textAlign': 'center'})
            return error_message, None, None, None
            
        # Get file paths based on interview_id
        CSV_FILE_PATH, JSON_FILE_PATH = get_file_paths(interview_id)
        
        try:
            # Read CSV directly from path
            try:
                processed_df = pd.read_csv(CSV_FILE_PATH)
                if 'Timestamp' in processed_df.columns:
                    processed_df['Timestamp'] = pd.to_datetime(processed_df['Timestamp'], errors='coerce')
                    processed_df = processed_df.set_index('Timestamp')
                file_name = CSV_FILE_PATH.split('/')[-1]
            except FileNotFoundError:
                error_message = html.Div(f"Error: CSV file not found for interview {interview_id}",
                                       style={'color': 'red', 'textAlign': 'center'})
                return error_message, None, None, None
            except Exception as e:
                error_message = html.Div(f"Error reading CSV file for interview {interview_id}: {str(e)}",
                                       style={'color': 'red', 'textAlign': 'center'})
                return error_message, None, None, None

            print(f"Analyzing interview {interview_id} from path: {CSV_FILE_PATH}")
            
            with open(CSV_FILE_PATH, 'r') as csv_file_obj:
                results, processed_df_from_analysis = analyze_interview_performance(csv_file_obj, rolling_window=5)
            
            if processed_df_from_analysis is not None:
                processed_df = processed_df_from_analysis

            if results is None or processed_df is None:
                error_message = html.Div(f"Error analyzing interview {interview_id}. Please check the CSV format and required columns.",
                                       style={'color': 'red', 'textAlign': 'center'})
                return error_message, None, None, None

            results_json = json.dumps(results)
            processed_df_reset = processed_df.reset_index()
            processed_df_reset['Timestamp'] = processed_df_reset['Timestamp'].astype(str)
            dataframe_json = processed_df_reset.to_json(orient='split', date_format='iso')

            # Calculate technical scores from the JSON file
            technical_scores = calculate_technical_scores(JSON_FILE_PATH) 
            technical_scores_json = json.dumps(technical_scores)

            component_scores = results.get('component_scores', {})
            cheating_analysis = results.get('gaze_cheating_analysis')
            
            # Create a list of skill score HTML elements
            technical_score_items = []
            if technical_scores:
                for skill, score in technical_scores.items():
                    technical_score_items.append(html.Li(f"{skill.title()}: {score:.1f}/100"))
            else:
                technical_score_items.append(html.Li("No technical skill scores available."))

            overall_summary = html.Div([
                html.H2(f"Overall Performance Score: {results.get('final_score', 0):.1f}/100", 
                                style={'color': '#2e86ab', 'textAlign': 'center'}),
                html.P(f"Interview ID: {interview_id} | Analyzed File: {file_name}", 
                       style={'textAlign': 'center', 'fontSize': 'small', 'color': '#666'}),
                html.Hr(),
                html.Div([
                    html.Div([
                        html.H4("Behavioral Component Scores", style={'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                        html.Ul([html.Li(f"{name.replace('_', ' ').title()}: {score:.1f}/100") 
                                     for name, score in component_scores.items() if name != 'gaze'], 
                                     style={'listStyleType': 'none', 'paddingLeft': 0})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),
                    html.Div([
                        html.H4("Technical Skill Scores", style={'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                        html.Ul(technical_score_items, style={'listStyleType': 'none', 'paddingLeft': 0})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),
                ], style={'marginTop': '10px'}),
                html.Div([
                    html.H4("Potential Gaze Cheating Indicator", style={'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                    html.P(f"Indicator percentage: {cheating_analysis.get('potential_cheating_indicator', 0):.1f}/100" if cheating_analysis else "N/A"),
                ], style={'width': '98%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '15px', 'border': '1px dashed #f0ad4e', 'borderRadius': '5px', 'backgroundColor': '#fff8e1', 'marginTop': '10px'})
            ], style={'margin': '20px auto', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9', 'maxWidth': '1000px'})

            full_page_layout = html.Div([
                overall_summary,
                html.Div([
                    html.H2("Detailed Analysis", style={'textAlign': 'center', 'marginTop': '40px'}),
                    html.H3("Gaze Analysis", style={'marginTop': '30px', 'borderBottom': '2px solid #ccc', 'paddingBottom': '5px'}),
                    gaze_section(results, processed_df),
                    html.H3("Distance & Stability", style={'marginTop': '30px', 'borderBottom': '2px solid #ccc', 'paddingBottom': '5px'}),
                    distance_section(results, processed_df),
                    html.H3("Head Pose", style={'marginTop': '30px', 'borderBottom': '2px solid #ccc', 'paddingBottom': '5px'}),
                    headpose_section(results, processed_df),
                    html.H3("Data Quality", style={'marginTop': '30px', 'borderBottom': '2px solid #ccc', 'paddingBottom': '5px'}),
                    quality_section(results, processed_df),
                ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': 'auto'})
            ])

            return full_page_layout, results_json, dataframe_json, technical_scores_json
            
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}")
            error_message = html.Div(f"An error occurred for interview {interview_id}: {str(e)}", 
                                   style={'color': 'red', 'textAlign': 'center', 'marginTop': '30px'})
            return error_message, None, None, None
    
    return dash_app