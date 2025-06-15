from .gaze import analyze_gaze, set_calibration_thresholds
from .head import analyze_head
from ..helpers import setup_logging, merge_head_gaze
from .results_analysis import analyze_interview_performance
import os


BASE_DIR = os.path.dirname(__file__)

def evaluate_soft_skills(videos_paths, interview_id):
    gaze_path, head_path = setup_logging(interview_id)

    for video in videos_paths:
        analyze_gaze(video, gaze_path, interview_id)
        analyze_head(video, head_path)

    all_metrics_path = merge_head_gaze(gaze_path, head_path, interview_id)
    
    #push file to s3
    results, _ = analyze_interview_performance(all_metrics_path, rolling_window=5)

    overall_score = results.get('final_score', 0)

    return overall_score



def store_calibration_thresholds(thresholds, interview_id): 
    set_calibration_thresholds(thresholds, interview_id)
