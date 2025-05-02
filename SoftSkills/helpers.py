from collections import deque
import numpy as np, cv2, os, csv
from SoftSkills.constants import POSTURE_SMOOTHING
from globals import BASE_PATH

def bin2img(binary_image):
    np_arr = np.frombuffer(binary_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def init_interview(interviews, interview_id):
    interviews[interview_id] = deque(maxlen=POSTURE_SMOOTHING)
    csv_path = f"{BASE_PATH}/SoftSkills/logs/analysis_metrics_{interview_id}.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'posture_score', 'posture_confidence', 'posture_feedback',
                'head_pose', 'head_pose_confidence', 'overall_confidence', 'fps'
            ])