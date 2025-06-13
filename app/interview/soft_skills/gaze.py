import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv # Import the csv module for writing to CSV files
from datetime import datetime # Import datetime for timestamps
from typing import Optional, Tuple # Import Optional and Tuple for type hinting
from scipy.spatial import distance as dist # Import dist for Euclidean distance
from collections import deque # For implementing moving average
import os # For path operations
import argparse # For command-line arguments
import sys # To check if running in a notebook-like environment
import traceback # For detailed error printing
import json

# --- Display/Processing Configuration ---
FRAME_WIDTH = 800
FRAME_HEIGHT = 600
DEBUG_WINDOW_SIZE = (200, 150)
DEBUG_INFO_WIDTH = 350
DEBUG_INFO_HEIGHT = 650

# --- Calibration Parameters ---
CALIBRATION_TEXT_COLOR = (255, 255, 255) # White
CALIBRATION_SUCCESS_COLOR = (0, 255, 0) # Green
CALIBRATION_FAILURE_COLOR = (0, 0, 255) # Red
CALIBRATION_DOT_COLOR = (0, 0, 255) # Red dot for calibration target
CALIBRATION_DOT_RADIUS = 10
CALIBRATION_DURATION_SEC = 5 # Duration to collect data for center calibration
CALIBRATION_INSTRUCTIONS_DURATION_SEC = 2 # Duration to show instructions before data collection

# --- Constants for Tuning (Default/Fallback values) ---
# These thresholds are initial values and will be overwritten by calibration if performed.
# With only CENTER calibration, other thresholds are based on offsets from calibrated center.
# These offsets/multipliers are crucial for performance and might need tuning.
# DEFAULT_EAR_THRESHOLD_BLINK = 0.18 # Will be calculated from calibrated CENTER EAR
# DEFAULT_PARTIAL_SHUT_EAR_DOWN_THRESHOLD = 0.25 # Will be estimated from calibrated CENTER EAR
# DEFAULT_PARTIAL_SHUT_EAR_CENTER_THRESHOLD = 0.28 # Will be estimated from calibrated CENTER EAR
DEFAULT_GAZE_H_LEFT_THRESHOLD = 0.45
DEFAULT_GAZE_H_CENTER_FROM_LEFT_THRESHOLD = 0.48 # Hysteresis threshold
DEFAULT_GAZE_H_RIGHT_THRESHOLD = 0.55
DEFAULT_GAZE_H_CENTER_FROM_RIGHT_THRESHOLD = 0.52 # Hysteresis threshold
DEFAULT_GAZE_V_UP_THRESHOLD = 0.40 # Default threshold for vertical UP gaze ratio

# --- Multipliers/Offsets for Threshold Calculation based on CENTER Calibration ---
# These are used to calculate the final classification thresholds based on CENTER calibration data
# IMPORTANT: These values will heavily influence classification accuracy and might need tuning.
# These offsets are relative to the CALIBRATED average EAR/Ratio for the CENTER state.
EAR_BLINK_OFFSET_FROM_CENTER = 0.10 # Blink EAR is CENTER EAR minus this offset
# Offset for estimating DOWN EAR from CENTER EAR. This might need tuning.
EAR_DOWN_ESTIMATE_OFFSET_FROM_CENTER = 0.07 # Example offset: Estimated Down EAR is CENTER EAR minus this offset
# Multiplier for estimating CENTER hysteresis threshold from estimated DOWN EAR.
EAR_CENTER_FROM_DOWN_ESTIMATE_MULTIPLIER = 1.05 # Example multiplier

GAZE_H_THRESHOLD_OFFSET = 0.07 # Offset from calibrated center H-Ratio for horizontal thresholds (might need tuning)
GAZE_V_THRESHOLD_OFFSET = 0.07 # Offset from calibrated center V-Ratio for vertical thresholds (might need tuning)
GAZE_H_HYSTERESIS_OFFSET = 0.03 # Smaller offset for horizontal hysteresis thresholds (might need tuning)
GAZE_V_HYSTERESIS_OFFSET = 0.03 # Smaller offset for vertical hysteresis thresholds (might need tuning)


FIXED_ADAPTIVE_THRESH_C = 10 # Constant C for adaptive thresholding

# --- Smoothing Parameters ---
SMOOTHING_WINDOW_SIZE = 5 # Size of the moving average window

# --- Base Image Processing Parameters ---
BASE_ADAPTIVE_THRESH_BLOCK_SIZE = 25 # Base block size for adaptive thresholding
BASE_MORPH_KERNEL_SIZE_TUPLE = (5, 5) # Base kernel size for morphological operations
BASE_GAUSSIAN_BLUR_KERNEL_SIZE = (7, 7) # Kernel size for Gaussian blur
BASE_ROI_PADDING = 5 # Padding around the eye ROI

# --- Smile Detection Parameters ---
SMILE_MOUTH_WIDTH_IOD_RATIO_THRESHOLD = 0.4 # Threshold for smile detection based on mouth width to IOD ratio

# --- Other constants ---
MIN_PUPIL_AREA_RATIO = 0.005 # Minimum pupil area relative to eye ROI area
MAX_PUPIL_AREA_RATIO = 0.30 # Maximum pupil area relative to eye ROI area

# --- Landmark Indices ---
# Indices for key facial landmarks from MediaPipe Face Mesh
LEFT_EYE_INNER_CORNER = 133
LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144] # Indices for calculating Left Eye EAR
RIGHT_EYE_EAR_INDICES = [362, 387, 385, 263, 380, 373] # Indices for calculating Right Eye EAR
LEFT_EYE_OUTLINE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7, 246, 161, 159, 157, 173, 154, 155] # Indices for Left Eye outline
RIGHT_EYE_OUTLINE_INDICES = [263, 387, 385, 362, 380, 373, 390, 249, 466, 388, 386, 384, 398, 381, 382] # Indices for Right Eye outline
LEFT_EYE_UPPER_MIDPOINT = 159
LEFT_EYE_LOWER_MIDPOINT = 145
RIGHT_EYE_UPPER_MIDPOINT = 386
RIGHT_EYE_LOWER_MIDPOINT = 374
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# --- Helper Functions ---

default_thresholds = {
    'EAR_THRESHOLD_BLINK': 0.18, # Fallback default
    'PARTIAL_SHUT_EAR_DOWN_THRESHOLD': 0.25, # Fallback default
    'PARTIAL_SHUT_EAR_CENTER_THRESHOLD': 0.28, # Fallback default
    'GAZE_H_LEFT_THRESHOLD': DEFAULT_GAZE_H_LEFT_THRESHOLD,
    'GAZE_H_CENTER_FROM_LEFT_THRESHOLD': DEFAULT_GAZE_H_CENTER_FROM_LEFT_THRESHOLD,
    'GAZE_H_RIGHT_THRESHOLD': DEFAULT_GAZE_H_RIGHT_THRESHOLD,
    'GAZE_H_CENTER_FROM_RIGHT_THRESHOLD': DEFAULT_GAZE_H_CENTER_FROM_RIGHT_THRESHOLD,
    'GAZE_V_UP_THRESHOLD': DEFAULT_GAZE_V_UP_THRESHOLD
}




# --- Logging Setup ---


def get_lm_coord(landmarks, index: int, img_w: int, img_h: int) -> Optional[Tuple[int, int]]:
    """Safely gets landmark coordinates in pixels."""
    if not landmarks or not (0 <= index < len(landmarks)):
        # print(f"Warning: Landmark index {index} out of bounds or landmarks not available.")
        return None
    try:
        lm = landmarks[index]
        # Use image dimensions for coordinate calculation
        return (int(lm.x * img_w), int(lm.y * img_h))
    except (AttributeError, TypeError):
        # print(f"Warning: Could not get coordinates for landmark index {index}.")
        return None

def calculate_distance(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> float:
    """Calculates the Euclidean distance between two points."""
    if p1 is None or p2 is None: return 0.0
    return dist.euclidean(p1, p2)

def calculate_ear(landmarks, eye_indices: list[int], img_w: int, img_h: int) -> Optional[float]:
    """Calculates the Eye Aspect Ratio (EAR)."""
    if not landmarks or not eye_indices or len(eye_indices) != 6: return None
    try:
        # Get pixel coordinates for the 6 eye landmarks
        p1 = get_lm_coord(landmarks, eye_indices[0], img_w, img_h)
        p2 = get_lm_coord(landmarks, eye_indices[1], img_w, img_h)
        p3 = get_lm_coord(landmarks, eye_indices[2], img_w, img_h)
        p4 = get_lm_coord(landmarks, eye_indices[3], img_w, img_h)
        p5 = get_lm_coord(landmarks, eye_indices[4], img_w, img_h)
        p6 = get_lm_coord(landmarks, eye_indices[5], img_w, img_h)

        # Ensure all points are available
        if None in [p1, p2, p3, p4, p5, p6]: return None

        # Calculate vertical distances
        vertical_dist1 = calculate_distance(p2, p6)
        vertical_dist2 = calculate_distance(p3, p5)

        # Calculate horizontal distance
        horizontal_dist = calculate_distance(p1, p4)

        # Avoid division by zero
        if horizontal_dist == 0: return None

        # Calculate EAR
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    except Exception as e:
        # print(f"Error calculating EAR: {e}")
        return None

def detect_pupil(eye_roi_gray, eye_roi_color, current_block_size, current_kernel_tuple,
                 threshold_c, min_area_ratio, max_area_ratio,
                 eye_name="?"):
    """
    Detects the pupil using dynamically scaled parameters and ellipse fitting.
    Processes the full eye ROI.
    Returns pupil center (x, y) relative to ROI, or None, plus the threshold image for debug.
    """
    if eye_roi_gray is None or eye_roi_gray.size == 0:
        # Return None for pupil center and a blank image for threshold display
        return None, None, np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0]), dtype=np.uint8) if eye_roi_gray is not None else np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0]), dtype=np.uint8)

    rows, cols = eye_roi_gray.shape
    roi_area = rows * cols
    if roi_area == 0:
        return None, None, np.zeros_like(eye_roi_gray)

    # Apply Gaussian blur to reduce noise
    blurred_roi = cv2.GaussianBlur(eye_roi_gray, BASE_GAUSSIAN_BLUR_KERNEL_SIZE, 0)

    # Ensure block size is odd and at least 3
    safe_block_size = max(3, current_block_size if current_block_size % 2 == 1 else current_block_size + 1)

    # Apply adaptive thresholding
    try:
        # Use THRESH_BINARY_INV to get dark areas (pupil) as white
        threshold = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, safe_block_size, threshold_c)
    except cv2.error as e:
        # print(f"Adaptive threshold error ({eye_name}): {e}")
        # Return a black image for threshold display on error
        return None, None, np.zeros_like(eye_roi_gray)
    except Exception as e:
         # Catch other potential errors during thresholding
         # print(f"Unexpected error during adaptive threshold ({eye_name}): {e}")
         return None, None, np.zeros_like(eye_roi_gray)


    # Ensure kernel size is odd and at least 1
    safe_kernel_size = max(1, current_kernel_tuple[0] if current_kernel_tuple[0] % 2 == 1 else current_kernel_tuple[0] + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_kernel_size, safe_kernel_size))

    # Apply morphological operations to refine the thresholded image
    # CLOSE helps close small gaps in the pupil
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
    # OPEN helps remove small white noise spots outside the pupil
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    if roi_area == 0: roi_area = 1 # Avoid division by zero later

    # Filter contours to find potential pupil candidates
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by area ratio relative to the eye ROI area
        if not (min_area_ratio < area / roi_area < max_area_ratio):
            continue

        # Fit an ellipse to the contour (pupils are roughly elliptical)
        if len(cnt) >= 5: # Need at least 5 points to fit an ellipse
            try:
                (cx, cy), (minor_axis, major_axis), angle = cv2.fitEllipse(cnt)
                if major_axis > 0:
                    aspect_ratio = minor_axis / major_axis
                    # Filter by aspect ratio (pupils are not extremely elongated)
                    if aspect_ratio > 0.3: # This threshold might need tuning
                         valid_contours.append({
                            'contour': cnt,
                            'ellipse': ((cx, cy), (minor_axis, major_axis), angle),
                            'area': area,
                            'center': (int(cx), int(cy)) # Store integer center coordinates
                        })
            except cv2.error:
                # print(f"Error fitting ellipse for a contour in {eye_name} eye.")
                pass # Skip this contour if ellipse fitting fails
            except Exception as e:
                 # Catch other potential errors during ellipse fitting
                 # print(f"Unexpected error fitting ellipse for a contour in {eye_name} eye: {e}")
                 pass # Skip this contour

    best_pupil_center_roi = None
    # best_threshold_display = threshold # Use the processed threshold image for display

    # Select the contour with the largest area as the most likely pupil
    if valid_contours:
        best_contour_data = max(valid_contours, key=lambda x: x['area'])
        best_ellipse = best_contour_data['ellipse']
        best_pupil_center_roi = best_contour_data['center'] # Use the stored integer center

        # Draw the selected pupil ellipse and center on the color ROI (for debug visualization)
        # The eye_roi_color is passed by reference, so drawing here affects the original ROI image
        cv2.ellipse(eye_roi_color, best_ellipse, (0, 255, 0), 1) # Green ellipse
        cv2.circle(eye_roi_color, best_pupil_center_roi, 3, (0, 255, 255), -1) # Yellow dot

    # Return pupil center coordinates relative to the ROI origin and the threshold image
    return best_pupil_center_roi[0] if best_pupil_center_roi else None, \
           best_pupil_center_roi[1] if best_pupil_center_roi else None, \
           threshold # Return the final threshold image

def get_horizontal_gaze_direction_ratio(pupil_x: Optional[int], inner_corner_x: Optional[int], outer_corner_x: Optional[int], is_left_eye: bool) -> Optional[float]:
    """Calculates horizontal pupil position relative to eye corners (0-1 ratio)."""
    if pupil_x is None or inner_corner_x is None or outer_corner_x is None: return None

    # Determine the left and right bounds of the eye based on whether it's the left or right eye
    if is_left_eye:
        # For the left eye, the outer corner is to the left, inner corner to the right
        left_bound_x, right_bound_x = outer_corner_x, inner_corner_x
    else:
        # For the right eye, the inner corner is to the left, outer corner to the right
        left_bound_x, right_bound_x = inner_corner_x, outer_corner_x

    eye_width = right_bound_x - left_bound_x

    # Avoid division by zero or negative width
    if eye_width <= 0: return None

    # Calculate the pupil's position relative to the left bound, normalized by eye width
    relative_pos = (pupil_x - left_bound_x) / eye_width

    # Clip the ratio to be within the 0-1 range
    return np.clip(relative_pos, 0.0, 1.0)

def get_vertical_gaze_direction_ratio(pupil_y: Optional[int], upper_midpoint_y: Optional[int], lower_midpoint_y: Optional[int]) -> Optional[float]:
    """Calculates vertical pupil position relative to upper and lower eyelid midpoints (0-1 ratio)."""
    if pupil_y is None or upper_midpoint_y is None or lower_midpoint_y is None: return None

    # Ensure upper midpoint is above lower midpoint (Y-coordinates increase downwards)
    if upper_midpoint_y >= lower_midpoint_y: return None

    eye_height = lower_midpoint_y - upper_midpoint_y

    # Avoid division by zero or negative height
    if eye_height <= 0: return None

    # Calculate the pupil's position relative to the upper bound, normalized by eye height
    relative_pos = (pupil_y - upper_midpoint_y) / eye_height

    # Clip the ratio to be within the 0-1 range
    return np.clip(relative_pos, 0.0, 1.0)


def smooth_value(history: deque, current_value: Optional[float]) -> Optional[float]:
    """Adds the current value to history and returns the smoothed average."""
    # Only append if the current value is not None
    if current_value is not None:
        history.append(current_value)

    # If history is empty, return None
    if not history: return None

    # Calculate and return the average of values in the history
    return sum(history) / len(history)







# --- Main Processing Function ---
def process_video(input_video_path, log_file_path, frame_skip=1, save_log=True, use_refined_landmarks=False, show_debug=False, calibrated_thresholds=None):


    if not os.path.exists(input_video_path):
        print(f"Error: Input video file not found at {input_video_path}")
        return

    
    # Construct log file path relative to output directory or current directory
    # log_file_name = os.path.splitext(os.path.basename(output_video_path))[0] + '_gaze_log.csv'
    # log_file_path = os.path.join(output_dir if output_dir else '.', log_file_name)


    # --- Initialize MediaPipe ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = None # Initialize to None
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=use_refined_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"Error initializing MediaPipe Face Mesh: {e}")
        return # Exit if MediaPipe initialization fails


    # --- Video Capture ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        if face_mesh: face_mesh.close()
        return

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # Handle case where FPS might be 0 or invalid
    if original_fps <= 0:
        print("Warning: Could not determine video FPS. Using default 30.")
        original_fps = 30.0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {original_width}x{original_height} @ {original_fps:.2f} FPS, Total Frames: {total_frames if total_frames > 0 else 'Unknown'}")

    
    # --- Thresholds ---
    # Use calibrated thresholds if provided, otherwise use defaults
    if calibrated_thresholds:
        dynamic_thresholds = calibrated_thresholds
        print("Using calibrated thresholds.")
    else:
        # Define default thresholds if calibration is skipped or fails
        dynamic_thresholds = default_thresholds
        print("Using default thresholds (calibration skipped or failed).")

    # Smoothing and Hysteresis states
    left_ear_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    right_ear_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    left_ratio_h_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    right_ratio_h_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    left_ratio_v_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    right_ratio_v_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    mouth_width_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)

    # Initialize previous gaze states for hysteresis
    prev_vertical_gaze_left = "CENTER"
    prev_vertical_gaze_right = "CENTER"
    prev_horizontal_gaze_left = "CENTER"
    prev_horizontal_gaze_right = "CENTER"
    # Initialize blink state for hysteresis
    is_currently_blinking = False


    # Distance related (less critical, but kept for parameter scaling)
    # Initialize reference_iod and scale_factor for the main processing loop
    reference_iod = None
    scale_factor = 1.0

    # CSV Logging Setup
    csv_file = None
    csv_writer = None
    if save_log:
        try:
            # Open in 'w' mode to overwrite or create anew for each run
            csv_file = open(log_file_path, 'a', newline='')
            csv_writer = csv.writer(csv_file)
            # Write header row including the new columns
            
            print(f"CSV log file will be saved to: {log_file_path}")
        except Exception as e:
            print(f"Error opening CSV file '{log_file_path}' for writing: {e}")
            save_log = False # Disable logging if error

    # --- Processing Loop ---
    try:
        # Initialize frame counters and time for FPS calculation
        frame_count_read = 0
        frame_count_processed = 0
        start_time = time.time() # Start time for FPS calculation

        while cap.isOpened(): # Continue as long as the video capture is open
            ret, frame = cap.read()
            if not ret:
                # If ret is False, it means the end of the video is reached or there's a read error
                if frame_count_read == 0:
                    print("Error: Could not read the first frame.")
                else:
                    print("End of video file reached.")
                break # Exit loop

            frame_count_read += 1

            # Resize the frame FIRST to the target processing/output dimensions
            # Flipping horizontally might be needed depending on camera source, adjust if necessary
            # frame = cv2.flip(frame, 1) # Optional: Flip if needed
            try:
                # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging (or consistent INTER_LINEAR)
                interpolation = cv2.INTER_AREA if (frame.shape[1] > FRAME_WIDTH or frame.shape[0] > FRAME_HEIGHT) else cv2.INTER_LINEAR
                resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=interpolation)
            except cv2.error as e:
                print(f"Error resizing frame {frame_count_read}: {e}. Skipping frame.")
                # Write a blank frame to maintain video duration/sync
                blank_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

                # Log data even for skipped frames, indicating the state
                if save_log and csv_writer:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    # Calculate current FPS for logging
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count_read / elapsed_time if elapsed_time > 0 else 0
                    # Log N/A for metrics not available for skipped/failed frames
                    try:
                        csv_writer.writerow([
                            timestamp,
                            frame_count_read,
                            f"{current_fps:.1f}", # FPS
                            "N/A", # Distance_cm
                            "SKIPPED", # Gaze Direction
                            "N/A", # Smoothed_EAR_Avg
                            "N/A", # Smoothed_Ratio_H_Avg
                            "N/A", # Smoothed_Ratio_V_Avg
                            "FRAME_ERROR_OR_SKIPPED", # Analysis_State
                            "N/A", # Reference_IOD_px
                            "N/A", # Scale_Factor
                            "N/A" # Smiling
                        ])
                    except Exception as e:
                         print(f"Error writing skipped frame data to CSV file: {e}")

                continue # Skip to the next frame

            # --- Frame Skipping Logic ---
            # Process frame only if it's the first frame or matches the skip interval
            process_this_frame = (frame_count_read == 1) or (frame_skip <= 1) or (frame_count_read % frame_skip == 0)

            results = None # Initialize results to None for each frame
            landmarks_available = False # Track if landmarks were found in the current frame

            if process_this_frame:
                frame_count_processed += 1
                # Make frame non-writeable for MediaPipe processing (performance hint)
                resized_frame.flags.writeable = False
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                # Process the frame with MediaPipe
                try:
                    results = face_mesh.process(rgb_frame)
                    if results and results.multi_face_landmarks:
                         landmarks_available = True
                except Exception as e:
                    print(f"Error processing frame {frame_count_read} with MediaPipe: {e}. Skipping analysis for this frame.")
                    results = None # Ensure results is None if processing fails
                    landmarks_available = False # No landmarks if processing failed

                # Make writeable again for drawing overlays
                resized_frame.flags.writeable = True


            # Initialize variables for the current frame analysis results
            # These will hold the results from the *last successfully processed frame* if the current one is skipped
            # Use getattr with a default value to handle the first frame case
            final_gaze = getattr(process_video, 'last_final_gaze', "N/A")
            smiling_status = getattr(process_video, 'last_smiling_status', "N/A")
            # Initialize metrics for logging for the current frame
            current_fps = 0.0
            current_iod = None
            current_scale_factor = 1.0
            smoothed_ear_avg = None
            smoothed_ratio_h_avg = None
            smoothed_ratio_v_avg = None
            analysis_state = "NO_FACE" # Default state if no landmarks found

            # Initialize debug display images (even if not shown, avoids errors if logic tries to use them)
            # Use last known debug images or create blank ones
            left_roi_display = getattr(process_video, 'last_left_roi_display', np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8))
            right_roi_display = getattr(process_video, 'last_right_roi_display', np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8))
            left_thresh_display = getattr(process_video, 'last_left_thresh_display', np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8))
            right_thresh_display = getattr(process_video, 'last_right_thresh_display', np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8))


            # --- Perform Analysis ONLY if process_this_frame is True AND landmarks were found ---
            if process_this_frame and landmarks_available:
                # Access landmarks for the first detected face
                landmarks = results.multi_face_landmarks[0].landmark
                img_h, img_w = resized_frame.shape[:2] # Use resized dimensions

                # --- Feature Extraction ---
                # Calculate Eye Aspect Ratio (EAR) for the current frame
                current_left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES, img_w, img_h)
                current_right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES, img_w, img_h)

                # Smooth the EAR values
                smoothed_left_ear = smooth_value(left_ear_history, current_left_ear)
                smoothed_right_ear = smooth_value(right_ear_history, current_right_ear)
                # Calculate average smoothed EAR for logging
                if smoothed_left_ear is not None and smoothed_right_ear is not None:
                    smoothed_ear_avg = (smoothed_left_ear + smoothed_right_ear) / 2.0
                elif smoothed_left_ear is not None:
                     smoothed_ear_avg = smoothed_left_ear
                elif smoothed_right_ear is not None:
                     smoothed_ear_avg = smoothed_right_ear
                # else: smoothed_ear_avg remains None


                # Get key points for IOD, smile, and gaze ratios
                left_outer_corner_pt = get_lm_coord(landmarks, LEFT_EYE_OUTER_CORNER, img_w, img_h)
                right_outer_corner_pt = get_lm_coord(landmarks, RIGHT_EYE_OUTER_CORNER, img_w, img_h)
                left_mouth_corner_pt = get_lm_coord(landmarks, LEFT_MOUTH_CORNER, img_w, img_h)
                right_mouth_corner_pt = get_lm_coord(landmarks, RIGHT_MOUTH_CORNER, img_w, img_h)

                # Calculate Inter-Ocular Distance (IOD) for scaling
                current_iod = None # Initialize for current frame
                current_scale_factor = 1.0 # Initialize for current frame
                if left_outer_corner_pt and right_outer_corner_pt:
                    current_iod = calculate_distance(left_outer_corner_pt, right_outer_corner_pt)
                    if current_iod is not None and current_iod > 0:
                        if reference_iod is None:
                            reference_iod = current_iod # Set reference on first valid IOD
                            # print(f"Reference IOD set to: {reference_iod:.2f}")
                        # Calculate scale factor relative to the reference IOD
                        current_scale_factor = current_iod / reference_iod if reference_iod > 0 else 1.0
                    else:
                         current_scale_factor = 1.0 # Default scale if IOD is zero or None
                else:
                    current_scale_factor = 1.0 # Default scale if corners not found

                # Scale image processing parameters based on distance (IOD)
                # Clamp scale factor to reasonable bounds to prevent extreme values
                current_scale_factor_for_params = max(0.5, min(current_scale_factor, 1.5))
                scaled_block_size = round(BASE_ADAPTIVE_THRESH_BLOCK_SIZE * current_scale_factor_for_params)
                scaled_kernel_val = round(BASE_MORPH_KERNEL_SIZE_TUPLE[0] * current_scale_factor_for_params)
                scaled_roi_padding = round(BASE_ROI_PADDING * current_scale_factor_for_params)

                # Ensure parameters meet minimum requirements and are odd where necessary
                scaled_block_size = max(3, scaled_block_size if scaled_block_size % 2 == 1 else scaled_block_size + 1)
                scaled_kernel_val = max(1, scaled_kernel_val if scaled_kernel_val % 2 == 1 else scaled_kernel_val + 1)
                scaled_kernel_tuple = (scaled_kernel_val, scaled_kernel_val)
                scaled_roi_padding = max(1, scaled_roi_padding)

                # --- Smile Detection ---
                current_mouth_width = None
                smiling_status = "NOT SMILING" # Default status
                if left_mouth_corner_pt and right_mouth_corner_pt:
                     current_mouth_width = calculate_distance(left_mouth_corner_pt, right_mouth_corner_pt)
                     smoothed_mouth_width = smooth_value(mouth_width_history, current_mouth_width)

                     # Check smile status based on smoothed width relative to IOD
                     if smoothed_mouth_width is not None and current_iod is not None and current_iod > 0:
                          mouth_width_iod_ratio = smoothed_mouth_width / current_iod
                          smiling_status = "SMILING" if mouth_width_iod_ratio > SMILE_MOUTH_WIDTH_IOD_RATIO_THRESHOLD else "NOT SMILING"
                     # else: smiling_status remains "NOT SMILING" if IOD is missing or zero
                # else: smiling_status remains "NOT SMILING" if mouth corners are missing


                # --- Pupil Detection and Gaze Ratio Calculation ---
                # Note: Pupil detection is performed within the eye ROI.
                # If the eye is closed (blink), pupil detection is expected to fail,
                # resulting in None for pupil coordinates and gaze ratios.
                # Blink detection relies primarily on the Eye Aspect Ratio (EAR),
                # which drops significantly when the eye is closed.

                left_inner_corner_pt = get_lm_coord(landmarks, LEFT_EYE_INNER_CORNER, img_w, img_h)
                right_inner_corner_pt = get_lm_coord(landmarks, RIGHT_EYE_INNER_CORNER, img_w, img_h)
                left_upper_midpoint_pt = get_lm_coord(landmarks, LEFT_EYE_UPPER_MIDPOINT, img_w, img_h)
                left_lower_midpoint_pt = get_lm_coord(landmarks, LEFT_EYE_LOWER_MIDPOINT, img_w, img_h)
                right_upper_midpoint_pt = get_lm_coord(landmarks, RIGHT_EYE_UPPER_MIDPOINT, img_w, img_h)
                right_lower_midpoint_pt = get_lm_coord(landmarks, RIGHT_EYE_LOWER_MIDPOINT, img_w, img_h)

                # Get eye region bounding boxes based on outline landmarks
                left_eye_pixels = [get_lm_coord(landmarks, i, img_w, img_h) for i in LEFT_EYE_OUTLINE_INDICES]
                left_eye_pixels = [pt for pt in left_eye_pixels if pt is not None] # Filter out None values

                right_eye_pixels = [get_lm_coord(landmarks, i, img_w, img_h) for i in RIGHT_EYE_OUTLINE_INDICES]
                right_eye_pixels = [pt for pt in right_eye_pixels if pt is not None] # Filter out None values

                # Process Left Eye
                left_raw_ratio_h, left_raw_ratio_v = None, None
                left_roi_color = None # Initialize ROI color image for debug
                left_thresh_display = np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8) # Initialize threshold display

                # Track if pupil was successfully detected in the current frame for robustness check
                left_pupil_detected_current_frame = False
                right_pupil_detected_current_frame = False


                if left_eye_pixels: # Ensure we have landmarks for bounding box
                    try:
                        # Calculate bounding box for the eye region
                        lx, ly, lw, lh = cv2.boundingRect(np.array(left_eye_pixels))
                        # Add padding, ensuring bounds are within the frame dimensions
                        lx1, ly1 = max(0, lx - scaled_roi_padding), max(0, ly - scaled_roi_padding)
                        lx2, ly2 = min(img_w, lx + lw + scaled_roi_padding), min(img_h, ly + lh + scaled_roi_padding)

                        # Extract ROI if valid dimensions
                        if lx2 > lx1 and ly2 > ly1:
                            left_roi_color = resized_frame[ly1:ly2, lx1:lx2].copy() # Copy for independent modification
                            left_roi_gray = cv2.cvtColor(left_roi_color, cv2.COLOR_BGR2GRAY)

                            # Detect pupil within the ROI
                            px_roi, py_roi, thresh_l = detect_pupil(
                                left_roi_gray, left_roi_color, scaled_block_size, scaled_kernel_tuple,
                                FIXED_ADAPTIVE_THRESH_C, MIN_PUPIL_AREA_RATIO, MAX_PUPIL_AREA_RATIO, "L"
                            )
                            left_thresh_display = thresh_l # Store threshold image for debug

                            # If pupil detected, calculate absolute coordinates and ratios
                            if px_roi is not None and py_roi is not None:
                                left_pupil_detected_current_frame = True # Pupil detected
                                current_left_pupil_abs = (lx1 + px_roi, ly1 + py_roi)
                                # Calculate horizontal gaze ratio
                                left_raw_ratio_h = get_horizontal_gaze_direction_ratio(
                                    current_left_pupil_abs[0],
                                    left_inner_corner_pt[0] if left_inner_corner_pt else None,
                                    left_outer_corner_pt[0] if left_outer_corner_pt else None,
                                    True # is_left_eye
                                )
                                # Calculate vertical gaze ratio
                                left_raw_ratio_v = get_vertical_gaze_direction_ratio(
                                    current_left_pupil_abs[1],
                                    left_upper_midpoint_pt[1] if left_upper_midpoint_pt else None,
                                    left_lower_midpoint_pt[1] if left_lower_midpoint_pt else None
                                )

                        if show_debug and left_roi_color is not None:
                             left_roi_display = cv2.resize(left_roi_color, DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
                             # Ensure threshold image is BGR for display stacking
                             if left_thresh_display is not None:
                                 if len(left_thresh_display.shape) == 2:
                                     left_thresh_display = cv2.cvtColor(cv2.resize(left_thresh_display, DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                                 else:
                                     left_thresh_display = cv2.resize(left_thresh_display, DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_NEAREST)
                             else: # If thresholding failed, show a blank image
                                 left_thresh_display = np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8)

                    except Exception as e:
                        # print(f"Error processing left eye: {e}") # Optional detailed logging
                        pass # Continue processing if one eye fails

                # Process Right Eye (similar logic)
                right_raw_ratio_h, right_raw_ratio_v = None, None
                right_roi_color = None # Initialize ROI color image for debug
                right_thresh_display = np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8) # Initialize threshold display

                if right_eye_pixels:
                    try:
                        rx, ry, rw, rh = cv2.boundingRect(np.array(right_eye_pixels))
                        rx1, ry1 = max(0, rx - scaled_roi_padding), max(0, ry - scaled_roi_padding)
                        rx2, ry2 = min(img_w, rx + rw + scaled_roi_padding), min(img_h, ry + lh + scaled_roi_padding)

                        if rx2 > rx1 and ry2 > ry1:
                            right_roi_color = resized_frame[ry1:ry2, rx1:rx2].copy()
                            right_roi_gray = cv2.cvtColor(right_roi_color, cv2.COLOR_BGR2GRAY)

                            px_roi, py_roi, thresh_r = detect_pupil(
                                right_roi_gray, right_roi_color, scaled_block_size, scaled_kernel_tuple,
                                FIXED_ADAPTIVE_THRESH_C, MIN_PUPIL_AREA_RATIO, MAX_PUPIL_AREA_RATIO, "R"
                            )
                            right_thresh_display = thresh_r # Store threshold image for debug

                            if px_roi is not None and py_roi is not None:
                                right_pupil_detected_current_frame = True # Pupil detected
                                current_right_pupil_abs = (rx1 + px_roi, ry1 + py_roi)
                                right_raw_ratio_h = get_horizontal_gaze_direction_ratio(current_right_pupil_abs[0], right_inner_corner_pt[0] if right_inner_corner_pt else None, right_outer_corner_pt[0] if right_outer_corner_pt else None, False)
                                right_raw_ratio_v = get_vertical_gaze_direction_ratio(current_right_pupil_abs[1], right_upper_midpoint_pt[1] if right_upper_midpoint_pt else None, right_lower_midpoint_pt[1] if right_lower_midpoint_pt else None)

                        if show_debug and right_roi_color is not None:
                             right_roi_display = cv2.resize(right_roi_color, DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
                             if right_thresh_display is not None:
                                 if len(right_thresh_display.shape) == 2:
                                     right_thresh_display = cv2.cvtColor(cv2.resize(right_thresh_display, DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                                 else:
                                     right_thresh_display = cv2.resize(right_thresh_display, DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_NEAREST)
                             else:
                                 right_thresh_display = np.zeros((DEBUG_WINDOW_SIZE[1], DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8)

                    except Exception as e:
                        # print(f"Error processing right eye: {e}")
                        pass

                # Smooth the calculated ratios
                smoothed_left_ratio_h = smooth_value(left_ratio_h_history, left_raw_ratio_h)
                smoothed_right_ratio_h = smooth_value(right_ratio_h_history, right_raw_ratio_h)
                smoothed_left_ratio_v = smooth_value(left_ratio_v_history, left_raw_ratio_v)
                smoothed_right_ratio_v = smooth_value(right_ratio_v_history, right_raw_ratio_v)

                # Calculate average smoothed ratios for logging
                if smoothed_left_ratio_h is not None and smoothed_right_ratio_h is not None:
                    smoothed_ratio_h_avg = (smoothed_left_ratio_h + smoothed_right_ratio_h) / 2.0
                elif smoothed_left_ratio_h is not None:
                    smoothed_ratio_h_avg = smoothed_left_ratio_h
                elif smoothed_right_ratio_h is not None:
                    smoothed_ratio_h_avg = smoothed_right_ratio_h
                # else: smoothed_ratio_h_avg remains None

                if smoothed_left_ratio_v is not None and smoothed_right_ratio_v is not None:
                    smoothed_ratio_v_avg = (smoothed_left_ratio_v + smoothed_right_ratio_v) / 2.0
                elif smoothed_left_ratio_v is not None:
                    smoothed_ratio_v_avg = smoothed_left_ratio_v
                elif smoothed_right_ratio_v is not None:
                    smoothed_ratio_v_avg = smoothed_right_ratio_v
                # else: smoothed_ratio_v_avg remains None


                # --- Gaze Determination (using smoothed values and dynamic thresholds) ---
                # Determine vertical gaze state (BLINK > EAR Down > Ratio Up > Center)

                # Check if current EAR suggests a low EAR state (either eye below center hysteresis threshold)
                # This is a broader check than just blink threshold
                is_low_ear = (smoothed_left_ear is not None and smoothed_left_ear < dynamic_thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD']) or \
                             (smoothed_right_ear is not None and smoothed_right_ear < dynamic_thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD'])

                # Check if pupil detection failed for *both* eyes in the current frame
                pupil_detection_failed_both_eyes = not left_pupil_detected_current_frame and not right_pupil_detected_current_frame

                # Blink Detection with Hysteresis and Pupil Detection Check
                # Enter BLINK state if EAR is below BLINK threshold AND pupil detection failed for both eyes
                if (smoothed_left_ear is not None and smoothed_left_ear < dynamic_thresholds['EAR_THRESHOLD_BLINK']) and \
                   (smoothed_right_ear is not None and smoothed_right_ear < dynamic_thresholds['EAR_THRESHOLD_BLINK']) and \
                   pupil_detection_failed_both_eyes:
                    is_currently_blinking = True
                # Exit BLINK state if EAR is above the CENTER Hysteresis threshold
                elif is_currently_blinking and \
                     (smoothed_left_ear is not None and smoothed_left_ear > dynamic_thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD']) and \
                     (smoothed_right_ear is not None and smoothed_right_ear > dynamic_thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD']):
                    is_currently_blinking = False

                # Determine vertical gaze state based on blink state and EAR thresholds
                if is_currently_blinking:
                    current_vertical_gaze = "BLINK"
                else:
                    # If not blinking, check for DOWN gaze using EAR thresholds and hysteresis
                    # Apply hysteresis: If previously DOWN, stay DOWN unless clearly back to CENTER by EAR
                    was_prev_down = prev_vertical_gaze_left == "DOWN" or prev_vertical_gaze_right == "DOWN"

                    # Check if current EAR suggests a DOWN gaze (either eye below down threshold)
                    is_currently_down_ear = (smoothed_left_ear is not None and smoothed_left_ear < dynamic_thresholds['PARTIAL_SHUT_EAR_DOWN_THRESHOLD']) or \
                                            (smoothed_right_ear is not None and smoothed_right_ear < dynamic_thresholds['PARTIAL_SHUT_EAR_DOWN_THRESHOLD'])

                    # Check if current EAR suggests a CENTER gaze (both eyes above center hysteresis threshold)
                    is_currently_center_ear = (smoothed_left_ear is not None and smoothed_left_ear > dynamic_thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD']) and \
                                              (smoothed_right_ear is not None and smoothed_right_ear > dynamic_thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD'])


                    if was_prev_down and not is_currently_center_ear:
                         # If was DOWN and not clearly CENTER by EAR, stay DOWN (hysteresis)
                         current_vertical_gaze = "DOWN"
                    elif is_currently_down_ear:
                         # If not in hysteresis and current EAR suggests DOWN, transition to DOWN
                         current_vertical_gaze = "DOWN"
                    else:
                         # If not BLINK or DOWN, check for UP gaze using vertical ratio
                         # UP gaze is detected if both smoothed vertical ratios are below the UP threshold.
                         # This check is only performed if pupil detection was successful and ratios are available.
                         if smoothed_left_ratio_v is not None and smoothed_right_ratio_v is not None:
                             avg_vertical_ratio = (smoothed_left_ratio_v + smoothed_right_ratio_v) / 2.0
                             current_vertical_gaze = "UP" if avg_vertical_ratio < dynamic_thresholds['GAZE_V_UP_THRESHOLD'] else "CENTER"
                         else:
                             # If ratios are unavailable (e.g., pupil not detected, but not a blink),
                             # assume CENTER vertically or maintain previous state if needed for robustness
                             # For now, defaulting to CENTER if ratios are missing and not blinking/down by EAR
                             current_vertical_gaze = "CENTER"


                # Update previous vertical gaze state for next frame's hysteresis
                prev_vertical_gaze_left = current_vertical_gaze
                prev_vertical_gaze_right = current_vertical_gaze

                # Determine horizontal gaze state (Left/Right/Center using horizontal ratio + hysteresis)
                # Left Eye Horizontal State
                stable_horizontal_gaze_left = "CENTER" # Default
                if smoothed_left_ratio_h is not None:
                    if prev_horizontal_gaze_left == "LEFT": # Was LEFT
                        # Stay LEFT unless ratio is above the CENTER_FROM_LEFT threshold
                        stable_horizontal_gaze_left = "CENTER" if smoothed_left_ratio_h > dynamic_thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'] else "LEFT"
                    elif prev_horizontal_gaze_left == "RIGHT": # Was RIGHT
                         # Stay RIGHT unless ratio is below the CENTER_FROM_RIGHT threshold
                        stable_horizontal_gaze_left = "CENTER" if smoothed_left_ratio_h < dynamic_thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD'] else "RIGHT"
                    else: # Was CENTER
                        # Transition to LEFT if ratio is below LEFT threshold
                        if smoothed_left_ratio_h < dynamic_thresholds['GAZE_H_LEFT_THRESHOLD']: stable_horizontal_gaze_left = "LEFT"
                        # Transition to RIGHT if ratio is above RIGHT threshold
                        elif smoothed_left_ratio_h > dynamic_thresholds['GAZE_H_RIGHT_THRESHOLD']: stable_horizontal_gaze_left = "RIGHT"
                        # else: remains CENTER if within CENTER thresholds
                # else: stable_horizontal_gaze_left remains "CENTER" if ratio unavailable

                # Right Eye Horizontal State (similar logic)
                stable_horizontal_gaze_right = "CENTER" # Default
                if smoothed_right_ratio_h is not None:
                    if prev_horizontal_gaze_right == "LEFT": # Was LEFT
                        stable_horizontal_gaze_right = "CENTER" if smoothed_right_ratio_h > dynamic_thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'] else "LEFT"
                    elif prev_horizontal_gaze_right == "RIGHT": # Was RIGHT
                        stable_horizontal_gaze_right = "CENTER" if smoothed_right_ratio_h < dynamic_thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD'] else "RIGHT"
                    else: # Was CENTER
                        if smoothed_right_ratio_h < dynamic_thresholds['GAZE_H_LEFT_THRESHOLD']: stable_horizontal_gaze_right = "LEFT"
                        elif smoothed_right_ratio_h > dynamic_thresholds['GAZE_H_RIGHT_THRESHOLD']: stable_horizontal_gaze_right = "RIGHT"
                        # else: remains CENTER
                # else: stable_horizontal_gaze_right remains "CENTER" if ratio unavailable

                # Update previous horizontal states for next frame's hysteresis
                prev_horizontal_gaze_left = stable_horizontal_gaze_left
                prev_horizontal_gaze_right = stable_horizontal_gaze_right

                # Combine Horizontal Gaze
                # Prioritize agreement between eyes. If they disagree, or one is N/A,
                # fall back to the state of the eye that is not CENTER, if any.
                # If both are CENTER or disagree, the combined state is CENTER.
                current_horizontal_gaze = "CENTER" # Default combined horizontal gaze
                if stable_horizontal_gaze_left == stable_horizontal_gaze_right:
                    current_horizontal_gaze = stable_horizontal_gaze_left # Both eyes agree
                elif stable_horizontal_gaze_left != "CENTER" and stable_horizontal_gaze_right == "CENTER":
                    current_horizontal_gaze = stable_horizontal_gaze_left # Left eye is decisive
                elif stable_horizontal_gaze_left == "CENTER" and stable_horizontal_gaze_right != "CENTER":
                    current_horizontal_gaze = stable_horizontal_gaze_right # Right eye is decisive
                # else: Disagreement (e.g., LEFT vs RIGHT) or both CENTER, defaults to CENTER


                # --- Final Gaze Combination ---
                # Combine the determined vertical state with the combined horizontal state.
                # Blink has the highest priority.
                # If not blinking, the final gaze is determined by the vertical state (UP/DOWN)
                # If vertical state is CENTER, the final gaze is the combined horizontal state.
                if is_currently_blinking:
                    final_gaze = "BLINK"
                elif current_vertical_gaze == "UP":
                    final_gaze = "UP"
                elif current_vertical_gaze == "DOWN":
                    final_gaze = "DOWN"
                else: # Vertical Gaze is CENTER
                    final_gaze = current_horizontal_gaze # Final gaze is determined by horizontal state

                # Determine Analysis State
                if landmarks_available:
                    # Check for distance instability if IOD is available and changes significantly
                    # This is a simple check; a more robust check might involve a moving average of IOD
                    if current_iod is not None and reference_iod is not None and abs(current_iod - reference_iod) / reference_iod > 0.15: # Example threshold 15% change
                         analysis_state = "DISTANCE_UNSTABLE"
                    else:
                         analysis_state = "TRACKING"
                else:
                     analysis_state = "NO_FACE"


                # Store the results from this processed frame to be available for skipped frames
                process_video.last_final_gaze = final_gaze
                process_video.last_smiling_status = smiling_status
                if show_debug:
                    process_video.last_left_roi_display = left_roi_display
                    process_video.last_right_roi_display = right_roi_display
                    process_video.last_left_thresh_display = left_thresh_display
                    process_video.last_right_thresh_display = right_thresh_display

            # Calculate current FPS based on overall read frames
            elapsed_time = time.time() - start_time
            current_fps = frame_count_read / elapsed_time if elapsed_time > 0 else 0


            # --- Draw Overlays on the resized_frame ---
            # This section draws based on the `final_gaze` and `smiling_status` determined
            # in the *last processed frame* if the current one was skipped, or the current
            # frame's results if it was processed.

            # Draw pupil location (only if found in the *current* processed frame)
            if process_this_frame and landmarks_available: # Only draw current pupils if processed this frame and landmarks found
                # Need to re-calculate pupil absolute positions if drawing on the current frame
                # This requires re-extracting ROI and detecting pupil for drawing purposes
                # A more efficient approach would be to store the absolute pupil positions from the processing step
                # For simplicity here, I'll assume we can access the raw ratios and recalculate approximate pupil positions for drawing
                # A better implementation would pass pupil coordinates from the processing step
                # For now, skipping drawing pupils if not processed this frame to avoid errors.
                pass # Skipping drawing pupil for now to avoid complexity with skipped frames drawing

            # Define colors for text display
            color_green = (0, 255, 0); color_blue = (255, 0, 0); color_yellow = (0, 255, 255)
            color_orange = (0, 165, 255); color_cyan = (255, 255, 0); color_gray = (128, 128, 128)
            color_red = (0, 0, 255); color_white = (255, 255, 255)

            # Determine text color based on the final gaze state
            text_color = color_gray # Default N/A or skipped
            if final_gaze == "CENTER": text_color = color_green
            elif final_gaze == "DOWN": text_color = color_blue
            elif final_gaze in ["LEFT", "RIGHT"]: text_color = color_yellow
            elif final_gaze == "BLINK": text_color = color_orange
            elif final_gaze == "UP": text_color = color_cyan
            # Draw the final gaze direction text
            cv2.putText(resized_frame, f"GAZE: {final_gaze}", (20, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

            # Determine text color based on the smiling status
            smile_display_color = color_gray # Default N/A or skipped
            if smiling_status == "SMILING": smile_display_color = color_green
            elif smiling_status == "NOT SMILING": smile_display_color = color_red
            # Draw the smiling status text
            cv2.putText(resized_frame, f"Smiling: {smiling_status}", (20, FRAME_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, smile_display_color, 2, cv2.LINE_AA)


            # --- Log Data ---
            # Log data for *every* frame, using the most recent processed data if skipped
            if save_log and csv_writer:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                # Use the values from the current processed frame if available, otherwise use the last known values
                log_gaze_direction = final_gaze
                log_smoothed_ear_avg = f"{smoothed_ear_avg:.4f}" if smoothed_ear_avg is not None else "N/A"
                log_smoothed_ratio_h_avg = f"{smoothed_ratio_h_avg:.4f}" if smoothed_ratio_h_avg is not None else "N/A"
                log_smoothed_ratio_v_avg = f"{smoothed_ratio_v_avg:.4f}" if smoothed_ratio_v_avg is not None else "N/A"
                log_analysis_state = analysis_state
                log_reference_iod_px = f"{reference_iod:.2f}" if reference_iod is not None else "N/A"
                log_scale_factor = f"{current_scale_factor:.4f}" if current_scale_factor is not None else "N/A"
                log_smiling = smiling_status

                # For Distance_cm, we don't have a calibrated value, so log N/A
                log_distance_cm = "N/A"

                # Log the data row
                try:
                    csv_writer.writerow([timestamp, frame_count_read, f"{current_fps:.1f}",  log_distance_cm, log_gaze_direction, log_smoothed_ear_avg, log_smoothed_ratio_h_avg, log_smoothed_ratio_v_avg, log_analysis_state, log_reference_iod_px, log_scale_factor, log_smiling])
                except Exception as e:
                    print(f"Error writing to CSV file: {e}")
                    # Optionally disable logging if repeated errors occur


            # --- Display Debug Windows (if enabled) ---
            if show_debug:
                # Create Debug Info Display dynamically based on current frame status
                debug_info_frame = np.zeros((DEBUG_INFO_HEIGHT, DEBUG_INFO_WIDTH, 3), dtype=np.uint8)
                y_pos = 20
                font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.45; thickness = 1
                def put_text(current_y, text, color=color_white):
                    """Helper to put text on the debug frame."""
                    # Ensure text is a string and handle potential None values gracefully
                    display_text = str(text) if text is not None else "N/A"
                    cv2.putText(debug_info_frame, display_text, (10, current_y), font, font_scale, color, thickness, cv2.LINE_AA)
                    return current_y + 18 # Adjust spacing

                y_pos = put_text(y_pos, f"Frame: {frame_count_read}/{total_frames if total_frames > 0 else '?'}")
                y_pos = put_text(y_pos, f"Processed: {frame_count_processed}")
                y_pos = put_text(y_pos, f"Skipping: {frame_skip-1} frames" if frame_skip > 1 else "Processing all frames")
                y_pos = put_text(y_pos, "--- Status ---")
                y_pos = put_text(y_pos, f"Processing Frame: {'Yes' if process_this_frame else 'No'}")
                y_pos = put_text(y_pos, f"Landmarks Found: {'Yes' if landmarks_available else 'No'}")
                y_pos = put_text(y_pos, f"Analysis State: {analysis_state}") # Display Analysis State
                y_pos = put_text(y_pos, f"Final Gaze: {final_gaze}", color=text_color) # Use determined color
                y_pos = put_text(y_pos, f"Smiling: {smiling_status}", color=smile_display_color) # Use determined color

                # Add more detailed debug info (only show if frame was processed)
                if process_this_frame and landmarks_available:
                    y_pos = put_text(y_pos, "--- Details (Smoothed) ---")
                    ear_text_l = f"L EAR: {smoothed_left_ear:.4f}" if smoothed_left_ear is not None else "L EAR: N/A"
                    ear_text_r = f"R EAR: {smoothed_right_ear:.4f}" if smoothed_right_ear is not None else "R EAR: N/A"
                    y_pos = put_text(y_pos, ear_text_l)
                    y_pos = put_text(y_pos, ear_text_r)
                    ratio_text_l_h = f"L H-Ratio: {smoothed_left_ratio_h:.4f}" if smoothed_left_ratio_h is not None else "L H-Ratio: N/A"
                    ratio_text_r_h = f"R H-Ratio: {smoothed_right_ratio_h:.4f}" if smoothed_right_ratio_h is not None else "R H-Ratio: N/A"
                    y_pos = put_text(y_pos, ratio_text_l_h)
                    y_pos = put_text(y_pos, ratio_text_r_h)
                    ratio_text_l_v = f"L V-Ratio: {smoothed_left_ratio_v:.4f}" if smoothed_left_ratio_v is not None else "L V-Ratio: N/A"
                    ratio_text_r_v = f"R V-Ratio: {smoothed_right_ratio_v:.4f}" if smoothed_right_ratio_v is not None else "R V-Ratio: N/A"
                    y_pos = put_text(y_pos, ratio_text_l_v)
                    y_pos = put_text(y_pos, ratio_text_r_v)
                    y_pos = put_text(y_pos, f"IOD: {current_iod:.2f}px" if current_iod is not None else "IOD: N/A")
                    y_pos = put_text(y_pos, f"Scale: {current_scale_factor:.4f}")
                    y_pos = put_text(y_pos, "--- Thresholds ---")
                    for key, value in dynamic_thresholds.items():
                         y_pos = put_text(y_pos, f"{key}: {value:.4f}")

                else:
                     y_pos = put_text(y_pos, "--- Details (Last Processed) ---", color=color_gray)
                     # Display last known smoothed values if current frame wasn't processed
                     y_pos = put_text(y_pos, f"Avg EAR: {log_smoothed_ear_avg}")
                     y_pos = put_text(y_pos, f"Avg H-Ratio: {log_smoothed_ratio_h_avg}")
                     y_pos = put_text(y_pos, f"Avg V-Ratio: {log_smoothed_ratio_v_avg}")
                     y_pos = put_text(y_pos, f"Ref IOD: {log_reference_iod_px}")
                     y_pos = put_text(y_pos, f"Scale: {log_scale_factor}")


                # Display the windows
                try:
                    cv2.imshow('Processed Frame', resized_frame) # Show the frame being written
                    cv2.imshow('Debug Info', debug_info_frame)
                    # Stack and show ROI/Threshold windows
                    combined_left = np.hstack((left_roi_display, left_thresh_display))
                    combined_right = np.hstack((right_roi_display, right_thresh_display))
                    cv2.imshow('Left Eye Debug (ROI | Threshold)', combined_left)
                    cv2.imshow('Right Eye Debug (ROI | Threshold)', combined_right)

                    # Allow minimal delay to see frames, press ESC to stop early
                    if cv2.waitKey(1) & 0xFF == 27: # 27 is the ESC key
                        print("ESC pressed, stopping processing early.")
                        break
                except cv2.error as e:
                    print(f"Error displaying debug windows: {e}")
                    # Optionally disable show_debug if display errors occur repeatedly


            # Print progress update periodically to the console
            if frame_count_read % 100 == 0:
                 elapsed_time = time.time() - start_time
                 # Calculate effective FPS based on read frames
                 fps_overall = frame_count_read / elapsed_time if elapsed_time > 0 else 0
                 # Estimate remaining time
                 if total_frames > 0 and fps_overall > 0:
                     remaining_frames = total_frames - frame_count_read
                     time_remaining_sec = remaining_frames / fps_overall
                     time_remaining_str = time.strftime("%H:%M:%S", time.gmtime(time_remaining_sec))
                     progress_percent = (frame_count_read / total_frames) * 100
                     print(f"Processed frame {frame_count_read}/{total_frames} ({progress_percent:.1f}%) - Overall FPS: {fps_overall:.1f} - Est. Time Left: {time_remaining_str}")
                 else: # Handle unknown total frames
                     print(f"Processed frame {frame_count_read} - Overall FPS: {fps_overall:.1f}")


    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # --- Cleanup ---
        print("\nReleasing resources...")
        # Release video capture and writer
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("Video capture released.")
        # if 'video_writer' in locals() and video_writer.isOpened():
            # video_writer.release()
            # print("Video writer released.")

        # Close MediaPipe Face Mesh
        if 'face_mesh' in locals() and face_mesh:
            face_mesh.close()
            print("MediaPipe Face Mesh closed.")

        # Close CSV file
        if csv_file:
            try:
                csv_file.close()
                print("CSV log file closed.")
            except Exception as e:
                print(f"Error closing CSV file: {e}")

        print("-" * 30)
        print(f"Processing finished.")
        print(f"Total frames read: {frame_count_read}")
        print(f"Total frames processed (analyzed): {frame_count_processed}")
        if save_log and os.path.exists(log_file_path): # Check if log file was actually created
             print(f"Gaze log saved to: {log_file_path}")
        elif save_log:
             print(f"CSV log file was requested but not successfully created/saved.")
        print("-" * 30)






BASE_DIR = os.path.dirname(__file__)


def get_calibration_file_path():
    cal_path = os.path.join(BASE_DIR, 'calibration')
    cal_file_path = os.path.join(cal_path, 'calibration.json')

    if not os.path.exists(cal_file_path):
        os.makedirs(cal_path, exist_ok=True)
        
        with open(cal_file_path, 'w') as f:
            json.dump({}, f)

    return cal_file_path

def get_calibrated_thresholds(interview_id):
    cal_file_path = get_calibration_file_path()
    with open(cal_file_path, 'r') as file:
        data = json.load(file)
    
    calibrated = data.get(str(interview_id), {})

    return calibrated if calibrated else default_thresholds


def set_calibration_thresholds(thresholds, interview_id):
    cal_file_path = get_calibration_file_path()
    with open(cal_file_path, 'r') as file:
        data = json.load(file)

    data[str(interview_id)] = thresholds

    with open(cal_file_path, 'w') as file:
        json.dump(data)
     

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$calibrate$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def analyze_gaze(video_path, csv_path, interview_id):

    # csv_log_path = setup_logging(interview_id)

    frames_to_skip = 1  # Process every frame (1), or set to 5 to process 1 in 5, etc.
    should_save_log = True # Set to True or False
    use_refined = False # Set to True for refined landmarks (slower), False otherwise
    show_debug_windows = False # Set to True to see debug windows, False to hide them
    perform_calibration = True # <<<=== SET THIS TO TRUE to run the simplified calibration
    # --- END OF PARAMETER DEFINITION ---

    # Validate frame skip value
    if frames_to_skip < 1:
        print("Warning: Frame skip value must be 1 or greater. Setting to 1.")
        frames_to_skip = 1

    calibrated_thresholds = None


    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = None
    cap = None
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=use_refined,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            if face_mesh: face_mesh.close()
            sys.exit(1) # Exit if video cannot be opened

    except Exception as e:
        print(f"Error initializing resources: {e}")
        if face_mesh: face_mesh.close()
        if cap: cap.release()
        sys.exit(1) # Exit on initialization error


    calibrated_thresholds = get_calibrated_thresholds(interview_id)


    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        # If cap was closed during calibration (e.g., by ESC), try reopening for processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot reopen video file {video_path} for processing after calibration.")
            if face_mesh: face_mesh.close()
            sys.exit(1)



    process_video(video_path, csv_path, frames_to_skip, should_save_log, use_refined, show_debug_windows, calibrated_thresholds)

    # Ensure resources are closed after processing in interactive mode
    if face_mesh: face_mesh.close()
    if cap and cap.isOpened(): cap.release()
    if show_debug_windows:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

