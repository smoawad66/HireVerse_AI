import mediapipe as mp
import numpy as np
from globals import BASE_PATH


# General Options
SHOW_DEBUG_WINDOWS = False # Set to True to show separate eye ROI/threshold windows (adds overhead)
SAVE_METRICS_LOG = True    # Enable saving combined metrics to CSV
LOG_TO_CONSOLE = True     # Enable logging messages to console
# Processing Frame Size: Smaller frames run faster on less powerful hardware.
# The display window can still be resized independently.
FRAME_WIDTH = 800          # Processing frame width (e.g., 800, 640)
FRAME_HEIGHT = 600         # Processing frame height (e.g., 600, 480)
MAIN_WINDOW_NAME = "AI Interview Analysis" # Unified window name

# Gaze Tracking & Distance Options
# USE_REFINED_LANDMARKS:
# True: Uses landmarks 473/468 (inner eye corners from refined mesh) or 234/454 (outer refined approx) for IOD. Requires MediaPipe FaceMesh(refine_landmarks=True).
#       Potentially more stable points but higher computational cost.
# False: Uses outer eye corners 33/263 for IOD. Lower computational cost.
# This choice affects distance calculation, scaling, and stability checks.
USE_REFINED_LANDMARKS = True # Keep True for potentially better stability if hardware allows
# <<< MODIFIED: Set to True to enable separate gaze log file >>>
SAVE_GAZE_LOG_SEPARATE = False # Set True to save gaze direction to its own file (gaze_log.csv)

# Presence Analysis Options
# Pose model complexity (0=fastest, 1=balanced, 2=most accurate but slowest).
POSE_MODEL_COMPLEXITY = 1 # Defaulting to balanced

# --- Constants ---

# -- Landmark Indices --
# Select IOD landmarks based on configuration
if USE_REFINED_LANDMARKS:
    # Using outer refined approx landmarks as per original test.txt logic
    LEFT_EYE_IOD_LM = 454 # Left eye outer corner (approx) - Index from refined landmarks
    RIGHT_EYE_IOD_LM = 234 # Right eye outer corner (approx) - Index from refined landmarks
    IOD_LANDMARK_DESC = "Refined Outer Approx (454/234)"

else:
    LEFT_EYE_IOD_LM = 33  # Left eye outer corner (standard landmarks)
    RIGHT_EYE_IOD_LM = 263 # Right eye outer corner (standard landmarks)
    IOD_LANDMARK_DESC = "Outer Corners (33/263)"

# Landmarks for EAR calculation (standard)
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 387, 385, 263, 380, 373]
# Landmarks for Gaze Ratio calculation (standard)
LEFT_EYE_INNER_CORNER = 133
LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263
LEFT_EYE_UPPER_MIDPOINT = 159
LEFT_EYE_LOWER_MIDPOINT = 145
RIGHT_EYE_UPPER_MIDPOINT = 386
RIGHT_EYE_LOWER_MIDPOINT = 374
# Landmarks for Eye ROI Bounding Box (standard)
LEFT_EYE_OUTLINE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7, 246, 161, 159, 157, 173, 154, 155]
RIGHT_EYE_OUTLINE_INDICES = [263, 387, 385, 362, 380, 373, 390, 249, 466, 388, 386, 384, 398, 381, 382]
# Nose tip, Nose base (?), L-Eye Outer, R-Eye Outer, L-Mouth, R-Mouth
LANDMARK_INDICES_HEAD_POSE = [1, 199, 33, 263, 61, 291]

# -- Gaze Tracking Constants --
# Calibration
GAZE_CALIBRATION_DURATION_SEC = 7 # Duration for collecting calibration data
CALIBRATION_IOD_TOLERANCE = 0.10 # Allow +/- 10% change in IOD during data collection relative to initial IOD
MIN_CALIBRATION_SAMPLES = 20     # Minimum number of stable samples required for successful calibration
CALIBRATION_TEXT_COLOR = (255, 255, 255) # White
CALIBRATION_COUNTDOWN_COLOR = (0, 255, 255) # Yellow
CALIBRATION_SUCCESS_COLOR = (0, 255, 0) # Green
CALIBRATION_FAILURE_COLOR = (0, 0, 255) # Red
CALIBRATION_INSTRUCTION_COLOR = (255, 255, 0) # Cyan for instructions
CALIBRATION_POINT_COLOR = (0, 0, 255) # Red for attention point
CALIBRATION_CROSSHAIR_COLOR = (255, 255, 255) # White for crosshair
CALIBRATION_STABLE_COLOR = (0, 255, 0) # Green when distance is stable
CALIBRATION_UNSTABLE_COLOR = (0, 165, 255) # Orange when distance is unstable

# Default Thresholds (overwritten by calibration if successful)
DEFAULT_EAR_THRESHOLD_BLINK = 0.18
DEFAULT_PARTIAL_SHUT_EAR_DOWN_THRESHOLD = 0.25
DEFAULT_PARTIAL_SHUT_EAR_CENTER_THRESHOLD = 0.28
DEFAULT_GAZE_H_LEFT_THRESHOLD = 0.45
DEFAULT_GAZE_H_CENTER_FROM_LEFT_THRESHOLD = 0.48
DEFAULT_GAZE_H_RIGHT_THRESHOLD = 0.55
DEFAULT_GAZE_H_CENTER_FROM_RIGHT_THRESHOLD = 0.52
DEFAULT_GAZE_V_UP_THRESHOLD = 0.40
# Threshold Estimation Multipliers/Offsets (based on calibrated center values)
EAR_DOWN_MULTIPLIER = 0.8
EAR_CENTER_HYSTERESIS_MULTIPLIER = 0.9
HORIZONTAL_LEFT_OFFSET = 0.05
HORIZONTAL_RIGHT_OFFSET = 0.05
HORIZONTAL_HYSTERESIS_OFFSET = 0.02 # Smaller offset for center transition thresholds
VERTICAL_UP_OFFSET = 0.05
# Pupil Detection Parameters (Base values, dynamically scaled by IOD ratio)
BASE_ADAPTIVE_THRESH_BLOCK_SIZE = 25 # Base block size (must be odd >= 3)
BASE_MORPH_KERNEL_SIZE_TUPLE = (5, 5) # Base kernel size (must be odd >= 1)
BASE_ROI_PADDING = 5 # Base padding around eye ROI
# Fixed Pupil Detection Parameters
BASE_GAUSSIAN_BLUR_KERNEL_SIZE = (7, 7) # Fixed Gaussian blur size
FIXED_ADAPTIVE_THRESH_C = 10 # Constant C for adaptive threshold
MIN_PUPIL_AREA_ROI_RATIO = 0.005 # Min pupil area relative to ROI area
MAX_PUPIL_AREA_ROI_RATIO = 0.30 # Max pupil area relative to ROI area
MIN_PUPIL_ASPECT_RATIO = 0.3 # Minimum aspect ratio (minor/major axis) for detected ellipse
# Smoothing
GAZE_SMOOTHING_WINDOW_SIZE = 5 # Window size for smoothing gaze metrics (EAR, Ratios)
SMOOTHING_CAL = 5 # Smaller smoothing window during calibration phase itself
# Display
GAZE_DEBUG_WINDOW_SIZE = (200, 150) # For optional separate eye windows

# -- Presence Analysis Constants --
# Distance Monitoring
REF_EYE_DISTANCE_CM = 6.3 # Approx real-world distance between eyes (adjust if needed)
MIN_DISTANCE_CM, MAX_DISTANCE_CM = 50, 80 # Desired operational distance range
# NOTE: Focal length calibration assumes user is roughly this far during the *first*
# successful IOD detection. Accuracy depends on this initial estimate.
FOCAL_LENGTH_CALIBRATION_DISTANCE_CM = 60
DISTANCE_SMOOTHING = 5 # Window size for smoothing distance calculation
# Distance Unstable State Constants
# Threshold for triggering UNSTABLE state: Percentage change in IOD from REFERENCE IOD
DISTANCE_CHANGE_THRESHOLD = 0.10 # e.g., 10% change triggers unstable
# Threshold for returning to TRACKING state: Max relative deviation within history buffer
DISTANCE_STABLE_THRESHOLD = 0.02 # e.g., IOD must vary less than 2% within buffer to be stable
DISTANCE_UNSTABLE_COOLDOWN_SEC = 2.0 # Min time in UNSTABLE state before checking for stability
UNSTABLE_HISTORY_LEN = GAZE_SMOOTHING_WINDOW_SIZE * 2 # Buffer length for checking stability
# Scale Factor Clamping (Improvement)
MIN_SCALE_FACTOR = 0.7  # Minimum allowed scale factor for pupil parameters
MAX_SCALE_FACTOR = 1.3  # Maximum allowed scale factor for pupil parameters
# Posture and Head Pose Analysis
POSTURE_SMOOTHING = 10 # Window size for smoothing posture metrics
SHOULDER_DIFF_THRESHOLD = 0.05 # Normalized Y diff between shoulders for 'level' score
HEAD_DISTANCE_THRESHOLD = 0.10 # Normalized Y diff (Nose-ShoulderMid) for 'slump' score
POSTURE_WEIGHTS = np.array([0.5, 0.3, 0.2]) # Weights: Shoulder Level, Head Slump, Centering
# Head Pose Thresholds (using values from test.txt, applied with soft.txt logic)
HEAD_POSE_FORWARD_THRESHOLD_Y = 1.5 # Yaw threshold (degrees, using *360 scaling)
HEAD_POSE_FORWARD_THRESHOLD_X = 1.0 # Pitch threshold for looking down (degrees, using *360 scaling)
HEAD_POSE_UP_THRESHOLD_X_FACTOR = 6.5 # Multiplier for looking up threshold
# Confidence Weights (Ensure they sum to 1.0)
POSTURE_CONFIDENCE_WEIGHT = 0.3
HEAD_POSE_CONFIDENCE_WEIGHT = 0.4
GAZE_CONFIDENCE_WEIGHT = 0.3
# Confidence mapping for gaze direction
GAZE_CONFIDENCE_MAP = {
    "CENTER": 100, "LEFT": 70, "RIGHT": 70, "UP": 50, "DOWN": 40,
    "BLINK": 10, "N/A": 0, "UNSTABLE": 0
}
# Camera Matrix for Head Pose (Approximate - assumes standard webcam FoV)
CAM_MATRIX = np.array([
    [FRAME_WIDTH, 0, FRAME_WIDTH / 2],
    [0, FRAME_WIDTH, FRAME_HEIGHT / 2], # Assuming focal length approx FRAME_WIDTH
    [0, 0, 1]
], dtype=np.float32)
DIST_MATRIX = np.zeros((4, 1), dtype=np.float32) # Assuming no lens distortion
# UI Colors
COLORS = {
    'text': (255, 255, 255), 'good': (0, 255, 0), 'close': (0, 0, 255),
    'far': (0, 165, 255), 'none': (255, 0, 0), 'center': (0, 255, 0),
    'not_center': (0, 0, 255), 'info': (255, 255, 0), 'warn': (0, 255, 255),
    'error': (0, 0, 255), 'gaze_l': (255, 0, 0), 'gaze_r': (0, 0, 255),
    'gaze_u': (255, 255, 0), 'gaze_d': (0, 165, 255), 'gaze_c': (0, 255, 0),
    'gaze_b': (255, 0, 255), 'unstable': (128, 128, 128) # Gray for unstable state
}
# UI Element Settings
DISTANCE_BAR_SETTINGS = (50, FRAME_HEIGHT - 100, 350, 20) # x, y, width, height
CENTERING_TOLERANCE_PX = 50 # Pixel tolerance for horizontal centering


mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose



face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=USE_REFINED_LANDMARKS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
    
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=POSE_MODEL_COMPLEXITY,
    smooth_landmarks=True
)


