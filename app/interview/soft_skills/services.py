from .gaze import analyze_gaze, set_calibration_thresholds
from .head import analyze_head
from ..helpers import setup_logging
import os


BASE_DIR = os.path.dirname(__file__)

def evaluate_soft_skills(videos_paths, interview_id):
    gaze_path, head_path = setup_logging(interview_id)

    for video in videos_paths:
        analyze_gaze(video, gaze_path, interview_id)
        analyze_head(video, head_path)


def store_calibration_thresholds(thresholds, interview_id): 
    set_calibration_thresholds(thresholds, interview_id)
