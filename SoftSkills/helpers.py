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

def img2bin(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # JPEG quality 90
    _, buffer = cv2.imencode('.jpg', img, encode_param)
    frame_binary = buffer.tobytes()
    return frame_binary

def get_default_interview_state():

    default_state = {
        'phase': 'distance_monitoring',
        'focal_length': None,
        'dynamic_gaze_thresholds': {
            'EAR_THRESHOLD_BLINK': 0.25,  # Replace with your DEFAULT_EAR_THRESHOLD_BLINK
            'PARTIAL_SHUT_EAR_DOWN_THRESHOLD': 0.3,
            'PARTIAL_SHUT_EAR_CENTER_THRESHOLD': 0.35,
            'GAZE_H_LEFT_THRESHOLD': 0.35,
            'GAZE_H_CENTER_FROM_LEFT_THRESHOLD': 0.45,
            'GAZE_H_RIGHT_THRESHOLD': 0.65,
            'GAZE_H_CENTER_FROM_RIGHT_THRESHOLD': 0.55,
            'GAZE_V_UP_THRESHOLD': 0.4
        },
        'reference_iod_store': {'value': None},
        'distance_buffer': None,
        'fps_filter': 30.0,
        'calibration_state': 'INSTRUCTIONS',
        'calibration_start_time': 0,
        'calibration_data': {"CENTER": {'ear': [], 'ratio_h': [], 'ratio_v': []}},
        'cal_ear_hist': None,
        'cal_ratio_h_hist': None,
        'cal_ratio_v_hist': None,
        'initial_calibration_iod': None,
        'user_proceed': False,
        'user_exit': False
    }
    
    return default_state
