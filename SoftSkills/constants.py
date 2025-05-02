import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


face_mesh_model = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
    
pose_model = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1,
    smooth_landmarks=True
)


REF_EYE_DISTANCE_CM = 6.3
MIN_DISTANCE_CM, MAX_DISTANCE_CM = 57, 65
FOCAL_LENGTH_CALIBRATION_DISTANCE_CM = 60 


POSTURE_SMOOTHING = 10
SHOULDER_DIFF_THRESHOLD = 0.05
HEAD_DISTANCE_THRESHOLD = 0.10 
POSTURE_WEIGHTS = np.array([0.5, 0.3, 0.2]) 
HEAD_POSE_FORWARD_THRESHOLD_Y = 1.5 
HEAD_POSE_FORWARD_THRESHOLD_X = 1.0 


POSTURE_CONFIDENCE_WEIGHT = 0.6
HEAD_POSE_CONFIDENCE_WEIGHT = 0.4

FRAME_WIDTH = 1000
FRAME_HEIGHT = 750

CAM_MATRIX = np.array([
    [FRAME_WIDTH, 0, FRAME_WIDTH / 2],
    [0, FRAME_WIDTH, FRAME_HEIGHT / 2],
    [0, 0, 1]
], dtype=np.float32)
DIST_MATRIX = np.zeros((4, 1), dtype=np.float32) 

LANDMARK_INDICES_HEAD_POSE = [1, 33, 263, 61, 291, 199]
LANDMARK_INDICES_HEAD_POSE = [1, 199, 33, 263, 61, 291]



COLORS = {
    'text': (255, 255, 255), 
    'good': (0, 255, 0),     
    'close': (0, 0, 255),    
    'far': (0, 165, 255),
    'none': (255, 0, 0),     
    'center': (0, 255, 0),   
    'not_center': (0, 0, 255)
}


BAR_SETTINGS = (50, 120, 350, 70) 
