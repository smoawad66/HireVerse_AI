import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import logging
import sys
import os, json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial import distance as dist
from collections import deque
from SoftSkills.constants import *
from globals import BASE_PATH

# --- Logging Setup ---
def setup_logging(interview_id, log_to_file=True, log_to_console=True, separate_gaze_log=False): # Added separate_gaze_log flag
    """Configure logging system."""
    handlers = []
    log_file_path = None
    csv_metrics_path = None
    log_dir = f'{BASE_PATH}/SoftSkills/logs' # Define log directory

    if log_to_file:
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError as e:
                print(f"Error creating log directory '{log_dir}': {e}")
                log_to_file = False # Disable file logging if dir creation fails

        if log_to_file:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # log_file_path = os.path.join(log_dir, f'analysis_{timestamp}.log')
            csv_metrics_path = os.path.join(log_dir, f'metrics_{interview_id}.csv')
            
            # try:
            #     handlers.append(logging.FileHandler(log_file_path))
            # except IOError as e:
            #      print(f"Error creating log file handler '{log_file_path}': {e}")
            #      log_file_path = None # Disable file logging if handler fails

            # Setup CSV file for metrics
            try:
                # Check if file exists and is empty to write header
                write_header = not os.path.exists(csv_metrics_path) or os.path.getsize(csv_metrics_path) == 0
                # Use 'a' mode to append if file exists
                with open(csv_metrics_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        # <<< MODIFIED: Conditionally exclude 'Gaze_Direction' from metrics header >>>
                        header = [
                            'Timestamp', 'FPS', 'Distance_cm', 'Centering_Status',
                            'Posture_Score', 'Posture_Confidence', 'Posture_Feedback',
                            'Head_Pose', 'Head_Pose_Confidence',
                            # Conditionally add Gaze_Direction
                            'Smoothed_EAR_Avg', 'Smoothed_Ratio_H_Avg', 'Smoothed_Ratio_V_Avg',
                            'Overall_Confidence', 'Analysis_State', 'Reference_IOD_px', 'Scale_Factor'
                        ]
                        if not separate_gaze_log:
                            header.insert(9, 'Gaze_Direction')

                        writer.writerow(header)
                        
                logging.info(f"Metrics logging to: {csv_metrics_path}")
            except IOError as e:
                logging.error(f"Failed to open/write metrics CSV file '{csv_metrics_path}': {e}")
                csv_metrics_path = None # Disable CSV logging if file access fails

    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout)) # Explicitly use stdout

    if not handlers: # If both failed or were disabled
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.warning("Logging is disabled.")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # More detailed format
            handlers=handlers
        )

    return log_file_path, csv_metrics_path

# --- Helper Functions ---

def get_landmark_coords(landmarks: Optional[List[Any]], index: int, img_w: int, img_h: int) -> Optional[Tuple[int, int]]:
    """Safely gets pixel coordinates for a given landmark index."""
    if landmarks and 0 <= index < len(landmarks):
        lm = landmarks[index]
        # Check for valid attributes and finite values
        if lm and hasattr(lm, 'x') and hasattr(lm, 'y') and np.isfinite(lm.x) and np.isfinite(lm.y):
             # Allow slightly off-screen values (-0.1 to 1.1) to handle edge cases
             if -0.1 < lm.x < 1.1 and -0.1 < lm.y < 1.1:
                 # Convert normalized coordinates to pixel coordinates, clamping to image bounds
                 px = int(np.clip(lm.x * img_w, 0, img_w - 1))
                 py = int(np.clip(lm.y * img_h, 0, img_h - 1))
                 return (px, py)
             else:
                 # Log if landmarks are significantly outside the expected [0, 1] range
                 if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                     logging.debug(f"Landmark {index} normalized coordinates out of [0, 1] bounds: ({lm.x:.3f}, {lm.y:.3f})")
                 return None # Return None if outside the slightly wider accepted range
        else:
             logging.debug(f"Invalid or non-finite landmark {index} data: {lm}")
             return None
    return None

def calculate_distance(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> float:
    """Calculates the Euclidean distance between two points. Returns 0.0 if input is invalid."""
    if p1 is None or p2 is None: return 0.0
    # Basic check for tuple/list of length 2
    if not (isinstance(p1, (tuple, list)) and len(p1) == 2 and
            isinstance(p2, (tuple, list)) and len(p2) == 2):
        logging.warning(f"Invalid input format for calculate_distance: p1={p1}, p2={p2}")
        return 0.0
    try:
        # Ensure coordinates are numeric before calculating distance
        if all(isinstance(coord, (int, float)) for coord in p1 + p2):
            return dist.euclidean(p1, p2)
        else:
            logging.warning(f"Non-numeric coordinates in calculate_distance: p1={p1}, p2={p2}")
            return 0.0
    except Exception as e:
        logging.error(f"Error calculating Euclidean distance between {p1} and {p2}: {e}", exc_info=False) # Reduce log noise
        return 0.0

def smooth_value(history: deque, current_value: Optional[float]) -> Optional[float]:
    """Adds valid value to history deque and returns the smoothed average."""
    if current_value is not None and np.isfinite(current_value): # Check for None, NaN, inf
        history.append(current_value)
    # Deque automatically handles maxlen
    if not history: return None
    try:
        return sum(history) / len(history)
    except ZeroDivisionError: # Should not happen if history is not empty, but safe check
        return None

# -- Gaze Tracking Helpers --

def detect_pupil(eye_roi_gray: np.ndarray, eye_roi_color: np.ndarray,
                 current_block_size: int, current_kernel_tuple: Tuple[int, int],
                 threshold_c: int, min_area_ratio: float, max_area_ratio: float,
                 min_aspect_ratio: float, eye_name: str = "?") -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray]]:
    """
    Detects the pupil using dynamically scaled parameters and ellipse fitting.
    Returns pupil center coordinates within the ROI and the thresholded image.
    """
    pupil_center_roi = None
    threshold_display = None

    if eye_roi_gray is None or eye_roi_gray.size == 0:
        logging.debug(f"Pupil detection ({eye_name}): Input ROI is empty.")
        return None, None
    rows, cols = eye_roi_gray.shape
    if rows < 5 or cols < 5: # Need minimum size for processing
        logging.debug(f"Pupil detection ({eye_name}): Input ROI too small ({rows}x{cols}).")
        return None, np.zeros_like(eye_roi_gray) # Return black image

    roi_area = rows * cols
    if roi_area == 0: return None, np.zeros_like(eye_roi_gray)

    try:
        # 1. Gaussian Blur (Fixed Kernel Size)
        blurred_roi = cv2.GaussianBlur(eye_roi_gray, BASE_GAUSSIAN_BLUR_KERNEL_SIZE, 0)

        # 2. Adaptive Threshold (Dynamically Scaled Block Size)
        # Ensure block size is odd and >= 3
        safe_block_size = max(3, current_block_size if current_block_size % 2 == 1 else current_block_size + 1)
        threshold = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, safe_block_size, threshold_c)
        threshold_display = threshold.copy() # Keep a copy for display

        # 3. Morphology (Dynamically Scaled Kernel Size)
        # Ensure kernel size is odd and >= 1
        safe_kernel_size = max(1, current_kernel_tuple[0] if current_kernel_tuple[0] % 2 == 1 else current_kernel_tuple[0] + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_kernel_size, safe_kernel_size))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Contour Detection
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Contour Filtering and Ellipse Fitting
        valid_contours = []
        min_pupil_area_px = roi_area * min_area_ratio
        max_pupil_area_px = roi_area * max_area_ratio

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area
            if not (min_pupil_area_px < area < max_pupil_area_px): continue

            # Filter by minimum points for ellipse fitting
            if len(cnt) >= 5:
                try:
                    # Fit ellipse
                    (cx, cy), (minor_axis, major_axis), angle = cv2.fitEllipse(cnt)

                    # Filter by valid axes and aspect ratio
                    if major_axis > 0 and minor_axis > 0: # Ensure valid ellipse dimensions
                        aspect_ratio = minor_axis / major_axis
                        if aspect_ratio >= min_aspect_ratio:
                            valid_contours.append({
                                'contour': cnt,
                                'ellipse': ((cx, cy), (minor_axis, major_axis), angle),
                                'area': area,
                                'center': (int(cx), int(cy))
                            })
                    else:
                         logging.debug(f"Invalid ellipse axes detected in {eye_name}: minor={minor_axis}, major={major_axis}")

                except cv2.error as fit_error:
                     # fitEllipse can fail on degenerate contours
                     if "points.size()" in str(fit_error): # More specific check for common error
                         logging.debug(f"fitEllipse failed in {eye_name} due to insufficient points variation.")
                     else:
                         logging.debug(f"fitEllipse failed for a contour in {eye_name}: {fit_error}")
                     continue # Skip this contour

        # 6. Select Best Pupil Candidate (largest valid area)
        if valid_contours:
            best_contour_data = max(valid_contours, key=lambda x: x['area'])
            pupil_center_roi = best_contour_data['center']
            best_ellipse = best_contour_data['ellipse']

            # Draw ellipse and center on the color ROI for debugging/visualization
            cv2.ellipse(eye_roi_color, best_ellipse, (0, 255, 0), 1) # Green ellipse
            cv2.circle(eye_roi_color, pupil_center_roi, 2, (0, 255, 255), -1) # Yellow center

    except cv2.error as e:
        logging.debug(f"OpenCV error during pupil detection ({eye_name}): {e}")
        return None, threshold_display # Return None for center, but maybe threshold img
    except Exception as e:
        logging.error(f"Unexpected error during pupil detection ({eye_name}): {e}", exc_info=True)
        return None, threshold_display

    return pupil_center_roi, threshold_display

def get_horizontal_gaze_direction_ratio(pupil_x: Optional[int], inner_corner_x: Optional[int], outer_corner_x: Optional[int], is_left_eye: bool) -> Optional[float]:
    """Calculates horizontal pupil position relative to eye corners (0-1 ratio)."""
    if pupil_x is None or inner_corner_x is None or outer_corner_x is None: return None

    # Determine left/right bounds based on which eye it is
    if is_left_eye:
        left_bound_x, right_bound_x = outer_corner_x, inner_corner_x
    else: # Right eye
        left_bound_x, right_bound_x = inner_corner_x, outer_corner_x

    # Ensure corners are logically placed (e.g., left < right)
    if left_bound_x >= right_bound_x:
        logging.debug(f"Invalid horizontal eye corners: left={left_bound_x}, right={right_bound_x}")
        return None

    eye_width = right_bound_x - left_bound_x
    if eye_width <= 1: # Avoid division by zero or tiny width
        logging.debug(f"Invalid eye width for H ratio: {eye_width}")
        return None

    relative_pos = (pupil_x - left_bound_x) / eye_width
    return np.clip(relative_pos, 0.0, 1.0) # Clip to ensure valid range

def get_vertical_gaze_direction_ratio(pupil_y: Optional[int], upper_midpoint_y: Optional[int], lower_midpoint_y: Optional[int]) -> Optional[float]:
    """Calculates vertical pupil position relative to upper/lower eyelid midpoints (0-1 ratio)."""
    if pupil_y is None or upper_midpoint_y is None or lower_midpoint_y is None: return None
    # Ensure upper is above lower
    if upper_midpoint_y >= lower_midpoint_y:
        logging.debug(f"Invalid vertical landmarks: upper={upper_midpoint_y}, lower={lower_midpoint_y}")
        return None

    eye_height = lower_midpoint_y - upper_midpoint_y
    if eye_height <= 1: # Avoid division by zero or tiny height
        logging.debug(f"Invalid eye height for V ratio: {eye_height}")
        return None

    relative_pos = (pupil_y - upper_midpoint_y) / eye_height
    return np.clip(relative_pos, 0.0, 1.0) # Clip to ensure valid range

def calculate_ear(landmarks: Optional[List[Any]], eye_indices: List[int], img_w: int, img_h: int) -> Optional[float]:
    """Calculates the Eye Aspect Ratio (EAR)."""
    if not landmarks or not eye_indices or len(eye_indices) != 6: return None

    try:
        coords = [get_landmark_coords(landmarks, i, img_w, img_h) for i in eye_indices]
        # Check if all 6 points were successfully retrieved
        if any(c is None for c in coords):
            logging.debug(f"Could not get all 6 EAR landmarks for indices: {eye_indices}")
            return None

        p1, p2, p3, p4, p5, p6 = coords

        # Calculate vertical distances
        vertical_dist1 = calculate_distance(p2, p6)
        vertical_dist2 = calculate_distance(p3, p5)
        # Calculate horizontal distance
        horizontal_dist = calculate_distance(p1, p4)

        if horizontal_dist < 1: # Prevent division by zero or near-zero
            logging.debug(f"Invalid horizontal distance for EAR: {horizontal_dist}")
            return None

        # Calculate EAR
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    except Exception as e: # Catch potential errors during calculation
        logging.debug(f"Error calculating EAR: {e}", exc_info=False)
        return None

def calculate_dynamic_gaze_thresholds(cal_data: Dict[str, Dict[str, List[float]]], thresholds: Dict[str, float]) -> bool:
    """
    Calculates dynamic gaze thresholds from collected calibration data (CENTER only).
    Updates the `thresholds` dictionary in-place. Returns True on success, False on failure.
    """
    try:
        center_ears = cal_data.get("CENTER", {}).get('ear', [])
        center_ratios_h = cal_data.get("CENTER", {}).get('ratio_h', [])
        center_ratios_v = cal_data.get("CENTER", {}).get('ratio_v', [])

        # Filter out None/NaN/Inf values that might have slipped in
        center_ears = [e for e in center_ears if e is not None and np.isfinite(e)]
        center_ratios_h = [h for h in center_ratios_h if h is not None and np.isfinite(h)]
        center_ratios_v = [v for v in center_ratios_v if v is not None and np.isfinite(v)]

        # Check minimum sample size
        if len(center_ears) < MIN_CALIBRATION_SAMPLES or \
           len(center_ratios_h) < MIN_CALIBRATION_SAMPLES or \
           len(center_ratios_v) < MIN_CALIBRATION_SAMPLES:
            logging.warning(f"Gaze calibration failed: Insufficient stable samples collected (Need {MIN_CALIBRATION_SAMPLES}). "
                            f"EAR: {len(center_ears)}, H: {len(center_ratios_h)}, V: {len(center_ratios_v)}")
            return False

        # Calculate Averages
        avg_center_ear = sum(center_ears) / len(center_ears)
        avg_center_ratio_h = sum(center_ratios_h) / len(center_ratios_h)
        avg_center_ratio_v = sum(center_ratios_v) / len(center_ratios_v)
        logging.info(f"Calibration Averages - EAR: {avg_center_ear:.3f}, Ratio H: {avg_center_ratio_h:.3f}, Ratio V: {avg_center_ratio_v:.3f}")

        # --- Estimate Thresholds based on Averages ---

        # 1. Vertical EAR Thresholds (Down/Center Hysteresis)
        # Threshold to transition from Center -> Down
        thresholds['PARTIAL_SHUT_EAR_DOWN_THRESHOLD'] = avg_center_ear * EAR_DOWN_MULTIPLIER
        # Threshold to transition from Down -> Center (higher for hysteresis)
        thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD'] = avg_center_ear * EAR_CENTER_HYSTERESIS_MULTIPLIER
        # Ensure DOWN threshold is above BLINK threshold + small margin
        thresholds['PARTIAL_SHUT_EAR_DOWN_THRESHOLD'] = max(
            thresholds['PARTIAL_SHUT_EAR_DOWN_THRESHOLD'],
            thresholds['EAR_THRESHOLD_BLINK'] + 0.01
        )
        # Ensure CENTER threshold is above DOWN threshold + small margin
        thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD'] = max(
            thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD'],
            thresholds['PARTIAL_SHUT_EAR_DOWN_THRESHOLD'] + 0.01
        )

        # 2. Horizontal Ratio Thresholds (Left/Right/Center Hysteresis)
        # Threshold to transition from Center -> Left
        thresholds['GAZE_H_LEFT_THRESHOLD'] = avg_center_ratio_h - HORIZONTAL_LEFT_OFFSET
        # Threshold to transition from Left -> Center
        thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'] = avg_center_ratio_h - HORIZONTAL_HYSTERESIS_OFFSET
        # Threshold to transition from Center -> Right
        thresholds['GAZE_H_RIGHT_THRESHOLD'] = avg_center_ratio_h + HORIZONTAL_RIGHT_OFFSET
        # Threshold to transition from Right -> Center
        thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD'] = avg_center_ratio_h + HORIZONTAL_HYSTERESIS_OFFSET

        # Clip thresholds to valid 0-1 range and ensure logical order/gaps
        thresholds['GAZE_H_LEFT_THRESHOLD'] = np.clip(thresholds['GAZE_H_LEFT_THRESHOLD'], 0.0, 1.0)
        thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'] = np.clip(
            thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'],
            thresholds['GAZE_H_LEFT_THRESHOLD'] + 0.01, # Ensure gap from LEFT
            1.0
        )
        thresholds['GAZE_H_RIGHT_THRESHOLD'] = np.clip(thresholds['GAZE_H_RIGHT_THRESHOLD'], 0.0, 1.0)
        thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD'] = np.clip(
            thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD'],
            0.0,
            thresholds['GAZE_H_RIGHT_THRESHOLD'] - 0.01 # Ensure gap from RIGHT
        )
        # Final check: Ensure Center_from_Left is less than Center_from_Right
        if thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'] >= thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD']:
             # If they overlap or cross, set them symmetrically around the average center with a small gap
             mid_point = (thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'] + thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD']) / 2.0
             thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD'] = np.clip(mid_point - 0.01, thresholds['GAZE_H_LEFT_THRESHOLD'] + 0.01, 1.0)
             thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD'] = np.clip(mid_point + 0.01, 0.0, thresholds['GAZE_H_RIGHT_THRESHOLD'] - 0.01)
             logging.warning("Adjusted overlapping horizontal center thresholds.")

        # 3. Vertical UP Ratio Threshold
        thresholds['GAZE_V_UP_THRESHOLD'] = avg_center_ratio_v - VERTICAL_UP_OFFSET
        thresholds['GAZE_V_UP_THRESHOLD'] = np.clip(thresholds['GAZE_V_UP_THRESHOLD'], 0.01, 0.99) # Ensure it's within 0-1 and not exactly 0 or 1

        logging.info(f"Dynamic Gaze Thresholds Calculated: { {k: f'{v:.3f}' for k, v in thresholds.items()} }")
        return True

    except Exception as e:
        logging.error(f"Error during dynamic gaze threshold calculation: {e}", exc_info=True)
        return False


def analyze_head_pose(face_landmarks_list: Optional[List[Any]], frame_dims: Tuple[int, int],
                      cam_matrix: np.ndarray, dist_matrix: np.ndarray) -> Tuple[str, int, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Analyzes head pose from face landmarks using solvePnP, based on soft.txt logic.
    Returns: head pose string, confidence score, start point (nose), end point for direction line.
    """
    width, height = frame_dims
    head_pose = "N/A"
    head_pose_confidence = 0
    p1 = p2 = None # Points for drawing direction line
    nose_2d = None # Store nose tip coordinates

    if not face_landmarks_list:
        logging.debug("Head pose: No face landmarks provided.")
        return head_pose, head_pose_confidence, p1, p2

    try:
        # Extract 2D and 3D points for head pose estimation
        face_2d = []
        face_3d = []

        for idx in LANDMARK_INDICES_HEAD_POSE:
            # Check if index is valid for the provided landmark list
            if not (0 <= idx < len(face_landmarks_list)):
                 logging.debug(f"Head pose: Landmark index {idx} out of bounds.")
                 return "N/A", 0, None, None # Cannot estimate pose without all points

            lm = face_landmarks_list[idx]
            # Safely get 2D coordinates using the helper function
            lm_coords = get_landmark_coords(face_landmarks_list, idx, width, height)

            if lm_coords is None:
                logging.debug(f"Head pose: Missing or invalid coordinates for landmark {idx}.")
                return "N/A", 0, None, None # Cannot estimate pose without all points

            x, y = lm_coords
            # Check for valid Z coordinate
            if hasattr(lm, 'z') and np.isfinite(lm.z):
                face_2d.append([x, y])
                # Use Z coordinate directly from MediaPipe for 3D point (relative depth)
                # This matches the logic from soft.txt (_analyze_head_pose)
                face_3d.append([x, y, lm.z])

                if idx == 1: # Nose tip landmark index (as per LANDMARK_INDICES_HEAD_POSE[0])
                    nose_2d = (x, y)
            else:
                logging.debug(f"Head pose: Invalid Z coordinate for landmark {idx}.")
                return "N/A", 0, None, None

        # Ensure we have the correct number of points and the nose tip
        if len(face_2d) != len(LANDMARK_INDICES_HEAD_POSE) or nose_2d is None:
            logging.debug(f"Head pose: Incorrect number of valid points collected ({len(face_2d)}/{len(LANDMARK_INDICES_HEAD_POSE)}) or nose_2d missing.")
            return "N/A", 0, None, None

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        success_pnp, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_EPNP # EPNP is often robust, DLS also an option
            # flags=cv2.SOLVEPNP_ITERATIVE # Original test.txt flag
            # flags=cv2.SOLVEPNP_DLS # Flag from soft.txt
        )

        if success_pnp:
            # Get rotation matrix
            rmat, _ = cv2.Rodrigues(rot_vec)
            # Decompose rotation matrix to get Euler angles using RQDecomp3x3
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # This scaling is kept for consistency with the original thresholds.
            x_angle = angles[0] * 360  # Pitch (Looking Up/Down)
            y_angle = angles[1] * 360  # Yaw (Looking Left/Right)
            # z_angle = angles[2] * 360  # Roll (Tilting Head) - Not used in logic below

            # Determine head direction based on angles and thresholds (using global constants)
            # Logic matches soft.txt _analyze_head_pose
            if y_angle < -HEAD_POSE_FORWARD_THRESHOLD_Y * 2: # Threshold doubled for Left/Right
                head_pose = "Looking Left"
            elif y_angle > HEAD_POSE_FORWARD_THRESHOLD_Y * 2:
                head_pose = "Looking Right"
            elif x_angle < -HEAD_POSE_FORWARD_THRESHOLD_X: # Negative pitch = looking down
                head_pose = "Looking Down"
            # Using the factor for UP threshold from test.txt constants
            elif x_angle > HEAD_POSE_FORWARD_THRESHOLD_X * HEAD_POSE_UP_THRESHOLD_X_FACTOR:
                head_pose = "Looking Up"
            else:
                head_pose = "Forward"

            # Assign confidence based on direction (matches soft.txt logic)
            head_pose_confidence = (
                100 if head_pose == "Forward"
                else 30 if "Down" in head_pose or "Up" in head_pose
                else 50 # Left/Right
            )

            # Calculate endpoint for drawing the head direction line (matches soft.txt logic)
            p1 = nose_2d # Start at nose tip
            # Endpoint calculation uses scaled angles (arbitrary scaling factor 5 from original scripts)
            p2_x = int(nose_2d[0] + y_angle * 5)
            # Subtract pitch because positive angle means looking down (image Y increases downwards)
            p2_y = int(nose_2d[1] - x_angle * 5)
            # Clamp endpoint to frame boundaries
            p2 = (np.clip(p2_x, 0, width-1), np.clip(p2_y, 0, height-1))

        else:
             logging.debug("solvePnP failed for head pose estimation.")
             head_pose = "N/A"
             head_pose_confidence = 0

    except cv2.error as pnp_error:
         logging.error(f"OpenCV error during solvePnP: {pnp_error}", exc_info=False)
         head_pose = "Error"
         head_pose_confidence = 0
    except Exception as e:
        # Log other potential errors (e.g., issues with landmark access, array creation)
        logging.error(f"Head pose estimation error: {e}", exc_info=True)
        head_pose = "Error"
        head_pose_confidence = 0

    return head_pose, head_pose_confidence, p1, p2


def analyze_posture(pose_landmarks_obj: Optional[Any], frame_dims: Tuple[int, int],
                    posture_history: deque) -> Tuple[int, int, str, Optional[Tuple[Tuple[int, int], Tuple[int, int]]], Tuple[int, int, int], deque]:
    """
    Analyzes posture from pose landmarks.
    Returns: posture score (1-5), confidence (0-100), feedback string,
             shoulder line points, shoulder line color, updated history deque.
    """
    width, height = frame_dims
    posture_score = 1 # Default score (1=Poor, 5=Good)
    posture_confidence = 0 # Default confidence
    posture_feedback = "No pose detected"
    shoulder_line_pts = None
    shoulder_color = COLORS['error'] # Default red

    if not pose_landmarks_obj or not hasattr(pose_landmarks_obj, 'landmark'):
        posture_history.clear() # Clear history if no pose
        return posture_score, posture_confidence, posture_feedback, shoulder_line_pts, shoulder_color, posture_history

    try:
        landmarks = pose_landmarks_obj.landmark
        # Get required landmarks using MediaPipe's enum
        lm_indices = {
            'ls': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
            'rs': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
            'n': mp.solutions.pose.PoseLandmark.NOSE.value
        }
        # Check if all required landmarks exist
        if not all(0 <= idx < len(landmarks) for idx in lm_indices.values()):
             logging.warning("Pose landmarks missing required indices.")
             posture_history.clear()
             return 1, 0, "Pose landmarks incomplete", None, COLORS['error'], posture_history

        sl_lm = landmarks[lm_indices['ls']]
        sr_lm = landmarks[lm_indices['rs']]
        nose_lm = landmarks[lm_indices['n']]

        # Check visibility threshold (more reliable than just coordinates)
        VISIBILITY_THRESHOLD = 0.6 # Increased slightly
        if sl_lm.visibility > VISIBILITY_THRESHOLD and \
           sr_lm.visibility > VISIBILITY_THRESHOLD and \
           nose_lm.visibility > VISIBILITY_THRESHOLD:

            # Calculate metrics using normalized coordinates (0 to 1)
            shoulder_center_x = (sl_lm.x + sr_lm.x) * 0.5
            shoulder_avg_y = (sl_lm.y + sr_lm.y) * 0.5
            shoulder_diff_y = abs(sl_lm.y - sr_lm.y) # Vertical difference (levelness)
            # Vertical distance: Nose Y - Avg Shoulder Y. Negative is good (nose above shoulders).
            head_shoulder_distance_y = nose_lm.y - shoulder_avg_y

            # Store current measurement [levelness_diff, slump_diff, centering_pos]
            current_measurement = np.array([shoulder_diff_y, head_shoulder_distance_y, shoulder_center_x])
            posture_history.append(current_measurement)

            # Get pixel coordinates for drawing shoulder line
            sl_pt = get_landmark_coords(landmarks, lm_indices['ls'], width, height)
            sr_pt = get_landmark_coords(landmarks, lm_indices['rs'], width, height)
            if sl_pt and sr_pt:
                shoulder_line_pts = (sl_pt, sr_pt)

            # Analyze posture if enough history has accumulated
            if len(posture_history) >= POSTURE_SMOOTHING // 2: # Require at least half the window
                # Calculate smoothed metrics
                avg_metrics = np.mean(posture_history, axis=0)
                avg_shoulder_diff, avg_head_dist, avg_center_x = avg_metrics

                # Calculate individual scores (0 to 1, where 1 is best)
                # Shoulder Level Score: Decreases as vertical difference increases
                shoulder_level_score = 1.0 - np.clip(avg_shoulder_diff / SHOULDER_DIFF_THRESHOLD, 0, 1)
                # Head Slump Score: Score = 1 if nose is above shoulders (avg_head_dist <= 0),
                # decreases linearly to 0 as nose drops below shoulders up to the threshold.
                head_slump_score = np.clip(1.0 - (max(0, avg_head_dist) / HEAD_DISTANCE_THRESHOLD), 0, 1)
                # Centering Score: Decreases as horizontal position deviates from center (0.5)
                center_score = 1.0 - np.clip(abs(avg_center_x - 0.5) / 0.3, 0, 1) # 0.3 is arbitrary range for deviation

                # Combine scores using weights
                combined_posture_score_01 = np.dot(POSTURE_WEIGHTS, [shoulder_level_score, head_slump_score, center_score])
                # Scale score to 1-5 range
                posture_score = int(np.clip(1 + combined_posture_score_01 * 4, 1, 5))
                # Confidence is the combined score scaled to 0-100
                posture_confidence = int(np.clip(combined_posture_score_01 * 100, 0, 100))

                # Generate feedback string
                if posture_score >= 4: posture_feedback = "Good posture"
                elif posture_score >= 3: posture_feedback = "Fair posture"
                else: posture_feedback = "Posture needs improvement"

                # Add specific tips based on individual scores
                tips = []
                if shoulder_level_score < 0.6: tips.append("Level shoulders")
                if head_slump_score < 0.6: tips.append("Sit upright")
                if center_score < 0.6: tips.append("Center yourself")
                if tips: posture_feedback += f" ({', '.join(tips)})"

                # Determine shoulder line color based on levelness score
                shoulder_color = COLORS['good'] if shoulder_level_score > 0.7 else COLORS['warn'] if shoulder_level_score > 0.4 else COLORS['error']
            else:
                 # Not enough history yet
                 posture_feedback = "Analyzing posture..."
                 shoulder_color = COLORS['info'] # Use info color while gathering data

        else:
            # Landmarks not visible enough
            posture_feedback = "Shoulders/head not fully visible"
            posture_history.clear() # Clear history if visibility lost
            shoulder_color = COLORS['error']

    except IndexError:
         logging.warning("Pose landmarks index out of range (model might not have loaded fully?)")
         posture_feedback = "Pose model loading..."
         posture_history.clear()
    except Exception as e:
        logging.error(f"Error processing pose landmarks: {e}", exc_info=True)
        posture_feedback = "Pose analysis error"
        posture_history.clear()

    return posture_score, posture_confidence, posture_feedback, shoulder_line_pts, shoulder_color, posture_history

def calculate_overall_confidence(posture_conf: int, head_pose_conf: int, gaze_conf: int) -> int:
    """Calculates overall confidence score based on weighted average of components."""
    # Ensure confidences are within [0, 100]
    posture_conf = np.clip(posture_conf, 0, 100)
    head_pose_conf = np.clip(head_pose_conf, 0, 100)
    gaze_conf = np.clip(gaze_conf, 0, 100)

    overall = (
        (posture_conf * POSTURE_CONFIDENCE_WEIGHT) +
        (head_pose_conf * HEAD_POSE_CONFIDENCE_WEIGHT) +
        (gaze_conf * GAZE_CONFIDENCE_WEIGHT)
    )
    return int(np.clip(overall, 0, 100)) # Ensure final score is also clipped

def update_fps(loop_start_time: float, fps_filter: float) -> float:
    """Calculates current FPS and updates smoothed FPS using EMA."""
    loop_time = time.time() - loop_start_time
    current_fps = 1.0 / max(loop_time, 1e-9) # Avoid division by zero
    # Exponential Moving Average (EMA)
    fps_filter = fps_filter * 0.9 + current_fps * 0.1
    return fps_filter

def get_face_landmarks(face_results: Optional[Any]) -> Optional[List[Any]]:
    """Safely extracts landmarks list from FaceMesh results."""
    if face_results and face_results.multi_face_landmarks:
        # Check if the list is not empty and the first face has landmarks
        if face_results.multi_face_landmarks[0].landmark:
            return face_results.multi_face_landmarks[0].landmark
        else:
            logging.debug("Face detected, but landmark list is empty.")
            return None
    return None

def get_pose_landmarks_obj(pose_results: Optional[Any]) -> Optional[Any]:
    """Safely extracts the PoseLandmarkList object from Pose results."""
    if pose_results and pose_results.pose_landmarks:
        return pose_results.pose_landmarks
    return None

# -- Drawing Helper --
def draw_overlay(frame: np.ndarray, metrics: Dict[str, Any], colors: Dict[str, Tuple[int, int, int]],
                 phase: str = "Analysis", analysis_state: str = "TRACKING") -> np.ndarray:
    """Draws the information overlay on the frame, adapting for different phases and analysis states."""
    height, width = frame.shape[:2]
    overlay = frame.copy() # Work on a copy to draw transparent background
    info_area_height = 160 # Height of the top info area background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_large = 0.7
    font_scale_medium = 0.6
    font_scale_small = 0.5
    thickness_normal = 2
    thickness_thin = 1
    text_color = colors['text']

    # --- Top Info Area Background (Semi-Transparent) ---
    cv2.rectangle(overlay, (0, 0), (width, info_area_height), (0, 0, 0), -1)
    alpha = 0.6 # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # --- Phase Specific Drawing ---
    if phase == "Distance":
        # Display Distance
        dist_text = f"Distance: {metrics['distance_cm']:.1f} cm" if metrics['distance_cm'] is not None else "Distance: -- cm"
        cv2.putText(frame, dist_text, (10, 30), font, font_scale_medium, COLORS['text'], thickness_normal)
        # Display Status
        dist_color = colors.get(metrics['distance_status'], colors['none'])
        cv2.putText(frame, f"Status: {metrics['distance_status'].upper()}", (10, 60), font, font_scale_medium, dist_color, thickness_normal)
        # Display Instructions
        cv2.putText(frame, "Press 'C' for Gaze Calibration (ESC to Cancel)", (10, 90), font, font_scale_small, CALIBRATION_INSTRUCTION_COLOR, thickness_thin)
        # Draw Distance Bar
        if metrics['distance_cm'] is not None:
            x, y, w, h = DISTANCE_BAR_SETTINGS
            # Calculate position ratio within the desired range
            pos_ratio = 0.0
            if MAX_DISTANCE_CM > MIN_DISTANCE_CM: # Avoid division by zero
                 pos_ratio = np.clip((metrics['distance_cm'] - MIN_DISTANCE_CM) / (MAX_DISTANCE_CM - MIN_DISTANCE_CM), 0, 1)
            # Draw background bar
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), -1)
            # Draw indicator circle
            indicator_x = x + int(pos_ratio * w)
            cv2.circle(frame, (indicator_x, y + h // 2), 10, dist_color, -1)
            # Min/Max markers and labels
            cv2.line(frame, (x, y + h), (x, y + h + 10), COLORS['text'], 2)
            cv2.line(frame, (x + w, y + h), (x + w, y + h + 10), COLORS['text'], 2)
            cv2.putText(frame, str(MIN_DISTANCE_CM), (x, y + h + 30), font, 0.4, COLORS['text'], 1)
            max_text = str(MAX_DISTANCE_CM)
            t_size, _ = cv2.getTextSize(max_text, font, 0.4, 1)
            cv2.putText(frame, max_text, (x + w - t_size[0], y + h + 30), font, 0.4, COLORS['text'], 1)
        # Draw Center Alignment Guide
        center_color = colors['center'] if metrics['centering_status'] else colors['not_center']
        # Vertical line below info area
        cv2.line(frame, (width // 2, info_area_height), (width // 2, height), center_color, 1)
        # User position marker at the bottom
        if metrics.get('user_x') is not None:
            cv2.line(frame, (int(metrics['user_x']), height - 20), (int(metrics['user_x']), height), COLORS['text'], 3)

    elif phase == "Calibration":
        # Display Calibration Instructions / Countdown / Status
        instruction_text = metrics.get('instruction_text', "")
        countdown_text = metrics.get('countdown_text', "")
        status_text = metrics.get('status_text', "") # e.g., "Hold Still", "Collecting Data"
        status_color = metrics.get('status_color', CALIBRATION_TEXT_COLOR)

        # Position text centrally
        def draw_centered_text(y_offset, text, scale, color, thickness):
             text_size, _ = cv2.getTextSize(text, font, scale, thickness)
             text_x = (width - text_size[0]) // 2
             text_y = height // 2 + y_offset
             cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness)

        draw_centered_text(-60, instruction_text, font_scale_medium, metrics.get('instruction_color', CALIBRATION_INSTRUCTION_COLOR), thickness_normal)
        draw_centered_text(0, countdown_text, font_scale_medium, metrics.get('countdown_color', CALIBRATION_COUNTDOWN_COLOR), thickness_normal)
        draw_centered_text(60, status_text, font_scale_medium, status_color, thickness_normal)

        # Draw Attention Point (Crosshair + Dot)
        center_x, center_y = width // 2, height // 2
        point_radius = 8
        line_length = 15
        # Draw crosshair lines
        cv2.line(frame, (center_x - line_length, center_y), (center_x + line_length, center_y), CALIBRATION_CROSSHAIR_COLOR, 2)
        cv2.line(frame, (center_x, center_y - line_length), (center_x, center_y + line_length), CALIBRATION_CROSSHAIR_COLOR, 2)
        # Draw central dot
        cv2.circle(frame, (center_x, center_y), point_radius, CALIBRATION_POINT_COLOR, -1) # Red dot

    elif phase == "Analysis":
        # --- Column 1: Presence & Posture ---
        x_col1 = 10
        y_pos = 30
        # Overall Confidence
        conf_color = colors['good'] if metrics['overall_confidence'] >= 75 else colors['warn'] if metrics['overall_confidence'] >= 50 else colors['error']
        cv2.putText(frame, f"Confidence: {metrics['overall_confidence']}%", (x_col1, y_pos), font, font_scale_large, conf_color, thickness_normal)
        y_pos += 35
        # Posture Score & Confidence
        posture_color = colors['good'] if metrics['posture_score'] >= 4 else colors['warn'] if metrics['posture_score'] >= 3 else colors['error']
        cv2.putText(frame, f"Posture: {metrics['posture_score']}/5 ({metrics['posture_confidence']}%)", (x_col1, y_pos), font, font_scale_medium, posture_color, thickness_normal)
        y_pos += 25
        # Posture Feedback
        cv2.putText(frame, metrics['posture_feedback'], (x_col1, y_pos), font, font_scale_small, text_color, thickness_thin)
        y_pos += 30 # Extra space after feedback
        # Distance
        dist_color = colors.get(metrics['distance_status'], colors['none'])
        dist_text = f"Distance: {metrics['distance_cm']:.1f} cm" if metrics['distance_cm'] is not None else "Distance: -- cm"
        cv2.putText(frame, dist_text, (x_col1, y_pos), font, font_scale_medium, dist_color, thickness_normal)
        y_pos += 25
        # Centering
        center_color = colors['center'] if metrics['centering_status'] else colors['not_center']
        center_text = "Centered" if metrics['centering_status'] else "Off-Center"
        cv2.putText(frame, f"Position: {center_text}", (x_col1, y_pos), font, font_scale_medium, center_color, thickness_normal)

        # --- Column 2: Head & Gaze ---
        x_col2 = width // 2 - 30 # Adjust starting position for balance
        y_pos = 30
        # Head Pose
        head_pose_color = colors['good'] if metrics['head_pose_confidence'] >= 70 else colors['warn'] if metrics['head_pose_confidence'] >= 40 else colors['error']
        cv2.putText(frame, f"Head: {metrics['head_pose']} ({metrics['head_pose_confidence']}%)", (x_col2, y_pos), font, font_scale_medium, head_pose_color, thickness_normal)
        y_pos += 35
        # Gaze Direction / Analysis State
        if analysis_state == "DISTANCE_UNSTABLE":
            gaze_text = "UNSTABLE"
            gaze_color = colors['unstable']
            # Display scale factor when unstable for debugging
            # <<< Optional: Display scale factor text if needed >>>
            # scale_text = f"Scale: {metrics.get('scale_factor', 1.0):.2f}"
            # cv2.putText(frame, scale_text, (x_col2, y_pos + 25), font, font_scale_small, colors['info'], thickness_thin)

        else:
            gaze_text = f"Gaze: {metrics['gaze_direction']}"
            gaze_color_map = {
                "LEFT": colors['gaze_l'], "RIGHT": colors['gaze_r'], "UP": colors['gaze_u'],
                "DOWN": colors['gaze_d'], "CENTER": colors['gaze_c'], "BLINK": colors['gaze_b'],
                "N/A": colors['text']
            }
            gaze_color = gaze_color_map.get(metrics['gaze_direction'], colors['text'])

        cv2.putText(frame, gaze_text, (x_col2, y_pos), font, font_scale_large, gaze_color, thickness_normal)

       
        # --- Draw Shoulder Line ---
        if metrics['shoulder_line_pts']:
            cv2.line(frame, metrics['shoulder_line_pts'][0], metrics['shoulder_line_pts'][1], metrics['shoulder_color'], 2)


    # --- Bottom Right: FPS (Common to Analysis, optional for others) ---
    fps_text = f"FPS: {metrics['fps']:.1f}"
    fps_size, _ = cv2.getTextSize(fps_text, font, font_scale_small, thickness_thin)
    cv2.putText(frame, fps_text, (width - fps_size[0] - 10, height - 10), font, font_scale_small, text_color, thickness_thin)


    return frame

# --- Phase Functions ---




def run_main_analysis(cap: cv2.VideoCapture, face_mesh: mp.solutions.face_mesh.FaceMesh, pose: mp.solutions.pose.Pose,
                      dynamic_thresholds: Dict[str, float], focal_length: Optional[float],
                      reference_iod_store: Dict[str, Optional[float]],
                      csv_metrics_path: Optional[str], csv_gaze_path: Optional[str],
                      separate_gaze_log: bool) -> str: # Added separate_gaze_log flag
    """
    Phase 3: Runs the main analysis loop combining all features (distance, posture, head pose, gaze).
    Handles distance changes using the UNSTABLE state and adapts pupil detection parameters.
    Returns: "EXIT", "RECALIBRATE", or "ERROR"
    """
    logging.info("Starting Main Analysis Phase...")
    # Get the reference IOD established during calibration (or None if calibration failed/skipped)
    reference_iod = reference_iod_store['value']
    if reference_iod is None:
        logging.warning("Reference IOD is None at start of main analysis. Distance instability detection and scaling may be unreliable until face is stable.")
    else:
        logging.info(f"Starting analysis with Reference IOD: {reference_iod:.2f}px")

    # --- State Variables ---
    analysis_state = "TRACKING" # Initial state: TRACKING or DISTANCE_UNSTABLE
    distance_unstable_start_time = 0.0
    unstable_iod_history = deque(maxlen=UNSTABLE_HISTORY_LEN) # History for checking stability after becoming unstable
    scale_factor = 1.0 # Initial scaling factor for pupil params (updated based on IOD ratio)

    # Smoothing Deques for various metrics
    posture_history = deque(maxlen=POSTURE_SMOOTHING)
    distance_buffer = deque(maxlen=DISTANCE_SMOOTHING)
    # Gaze smoothing (separate for left/right initially, then averaged)
    left_ear_hist = deque(maxlen=GAZE_SMOOTHING_WINDOW_SIZE)
    right_ear_hist = deque(maxlen=GAZE_SMOOTHING_WINDOW_SIZE)
    left_ratio_h_hist = deque(maxlen=GAZE_SMOOTHING_WINDOW_SIZE)
    right_ratio_h_hist = deque(maxlen=GAZE_SMOOTHING_WINDOW_SIZE)
    left_ratio_v_hist = deque(maxlen=GAZE_SMOOTHING_WINDOW_SIZE)
    right_ratio_v_hist = deque(maxlen=GAZE_SMOOTHING_WINDOW_SIZE)
    # Gaze Hysteresis States (remember previous state to avoid flickering)
    prev_vertical_gaze_state = "CENTER" # Combined state (BLINK, DOWN, UP, CENTER)
    prev_horizontal_gaze_state = "CENTER" # Combined state (LEFT, RIGHT, CENTER)

    # FPS Calculation
    fps_filter = 30.0 # Initial guess for smoothed FPS

    # CSV Writers (initialized outside loop)
    metrics_file = None
    metrics_writer = None
    gaze_file = None
    gaze_writer = None

    try:
        # --- Setup CSV Writers (if enabled) ---
        if SAVE_METRICS_LOG and csv_metrics_path:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(csv_metrics_path)
                if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
                # Open in append mode ('a')
                # Header is written during setup_logging
                metrics_file = open(csv_metrics_path, 'a', newline='', encoding='utf-8')
                metrics_writer = csv.writer(metrics_file)
                logging.info(f"Appending metrics to: {csv_metrics_path}")
            except IOError as e:
                logging.error(f"Error opening metrics CSV file for appending: {e}")
                metrics_writer = None # Disable writing

        # <<< MODIFIED: Use the separate_gaze_log flag passed from main >>>
        if separate_gaze_log and csv_gaze_path:
            try:
                # Ensure directory exists
                log_dir_gaze = os.path.dirname(csv_gaze_path)
                if log_dir_gaze and not os.path.exists(log_dir_gaze): os.makedirs(log_dir_gaze, exist_ok=True)
                # Open in append mode ('a')
                write_header_gaze = not os.path.exists(csv_gaze_path) or os.path.getsize(csv_gaze_path) == 0
                gaze_file = open(csv_gaze_path, 'a', newline='', encoding='utf-8')
                gaze_writer = csv.writer(gaze_file)
                if write_header_gaze:
                      gaze_writer.writerow(['Timestamp', 'Gaze Direction'])
                logging.info(f"Appending gaze log to: {csv_gaze_path}")
            except IOError as e:
                logging.error(f"Error opening gaze log CSV file for appending: {e}")
                gaze_writer = None # Disable writing

        # --- Main Analysis Loop ---
        while cap.isOpened():
            loop_start_time = time.time()

            # --- Frame Acquisition & Preprocessing ---
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame during main analysis.")
                break
                time.sleep(0.1)
            try:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            except cv2.error as e:
                logging.error(f"Failed to resize frame in main loop: {e}")
                continue
            frame = cv2.flip(frame, 1) # Horizontal flip
            img_h, img_w = frame.shape[:2]
            frame_dims = (img_w, img_h)

            # --- MediaPipe Processing ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            face_results = face_mesh.process(rgb_frame)
            pose_results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # --- Landmark Extraction ---
            face_landmarks = get_face_landmarks(face_results)
            pose_landmarks_obj = get_pose_landmarks_obj(pose_results)

            # --- Initialize Metrics Dict for this frame ---
            metrics = {
                'fps': fps_filter,
                'distance_cm': None, 'distance_status': 'none', 'centering_status': False,
                'posture_score': 1, 'posture_confidence': 0, 'posture_feedback': "No pose",
                'head_pose': "N/A", 'head_pose_confidence': 0,
                'gaze_direction': "N/A",
                'overall_confidence': 0,
                # Raw values (optional, could be removed if not needed)
                'raw_ear_left': None, 'raw_ear_right': None,
                'raw_ratio_h_left': None, 'raw_ratio_h_right': None,
                'raw_ratio_v_left': None, 'raw_ratio_v_right': None,
                # Smoothed values (used for logic)
                'smoothed_ear_avg': None, 'smoothed_ratio_h_avg': None, 'smoothed_ratio_v_avg': None,
                # Drawing helpers
                'head_pose_p1': None, 'head_pose_p2': None,
                'shoulder_line_pts': None, 'shoulder_color': COLORS['error'],
                'left_pupil_abs': None, 'right_pupil_abs': None,
                # State/Debug Info
                'reference_iod_px': reference_iod, # Log the current reference
                'scale_factor': scale_factor # Log the current scale factor
            }

            # --- Calculations (only if face landmarks available) ---
            current_iod_px = None # Initialize IOD for this frame
            if face_landmarks:
                # -- Distance, Centering, Scaling Factor, Unstable State Check --
                l_iod_pt = get_landmark_coords(face_landmarks, LEFT_EYE_IOD_LM, img_w, img_h)
                r_iod_pt = get_landmark_coords(face_landmarks, RIGHT_EYE_IOD_LM, img_w, img_h)
                current_iod_px = calculate_distance(l_iod_pt, r_iod_pt)

                if current_iod_px > 1:
                    # Ensure reference_iod is set if it was None initially (e.g., calibration skipped/failed)
                    if reference_iod is None:
                         reference_iod = current_iod_px # Set first valid IOD as reference
                         reference_iod_store['value'] = reference_iod
                         logging.info(f"Initial Reference IOD set during analysis: {reference_iod:.2f}px (using {IOD_LANDMARK_DESC})")
                         metrics['reference_iod_px'] = reference_iod # Update metrics dict

                    # Calculate Distance (if focal length is known)
                    if focal_length is not None and focal_length > 0:
                        raw_dist_cm = (REF_EYE_DISTANCE_CM * focal_length) / current_iod_px
                        metrics['distance_cm'] = smooth_value(distance_buffer, raw_dist_cm)
                        # Determine distance status
                        if metrics['distance_cm'] is not None:
                            status = "good"
                            if metrics['distance_cm'] < MIN_DISTANCE_CM: status = "close"
                            elif metrics['distance_cm'] > MAX_DISTANCE_CM: status = "far"
                            metrics['distance_status'] = status
                        else: metrics['distance_status'] = "none"
                    else:
                        metrics['distance_status'] = "none" # No focal length

                    # Calculate Scale Factor & Check Unstable State
                    if reference_iod and reference_iod > 0:
                        current_scale_factor_from_ref = current_iod_px / reference_iod
                        # Clamp scale factor to prevent extreme values (Improvement)
                        scale_factor = np.clip(current_scale_factor_from_ref, MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
                        if scale_factor != current_scale_factor_from_ref:
                             logging.debug(f"Analysis scale factor clamped: Original={current_scale_factor_from_ref:.3f}, Clamped={scale_factor:.3f}")
                        metrics['scale_factor'] = scale_factor # Log current (potentially clamped) scale factor

                        # --- Distance Unstable State Logic ---
                        if analysis_state == "TRACKING":
                            # Check if IOD deviates too much from the reference
                            if abs(current_scale_factor_from_ref - 1.0) > DISTANCE_CHANGE_THRESHOLD:
                                analysis_state = "DISTANCE_UNSTABLE"
                                distance_unstable_start_time = time.time()
                                unstable_iod_history.clear() # Clear history buffer
                                logging.info(f"STATE CHANGE: -> DISTANCE_UNSTABLE (IOD: {current_iod_px:.2f}, Ref: {reference_iod:.2f}, Scale: {current_scale_factor_from_ref:.3f})")
                                # Clear smoothing buffers for gaze/posture when entering unstable state
                                left_ear_hist.clear(); right_ear_hist.clear()
                                left_ratio_h_hist.clear(); right_ratio_h_hist.clear()
                                left_ratio_v_hist.clear(); right_ratio_v_hist.clear()
                                posture_history.clear()
                                # Reset gaze hysteresis states
                                prev_vertical_gaze_state = "CENTER"
                                prev_horizontal_gaze_state = "CENTER"

                        elif analysis_state == "DISTANCE_UNSTABLE":
                            unstable_iod_history.append(current_iod_px) # Add current IOD to history
                            # Check for stability only after cooldown and if buffer is sufficiently full
                            if time.time() - distance_unstable_start_time > DISTANCE_UNSTABLE_COOLDOWN_SEC and \
                               len(unstable_iod_history) >= UNSTABLE_HISTORY_LEN // 2:

                                avg_unstable_iod = np.mean(unstable_iod_history) if unstable_iod_history else 0
                                if avg_unstable_iod > 0:
                                    # Check max relative deviation within the history buffer
                                    # Calculate standard deviation as a measure of spread
                                    std_dev_iod = np.std(unstable_iod_history) if len(unstable_iod_history) > 1 else 0.0
                                    relative_deviation = std_dev_iod / avg_unstable_iod if avg_unstable_iod > 0 else 1.0

                                    if relative_deviation < DISTANCE_STABLE_THRESHOLD:
                                        # Distance has stabilized, transition back to TRACKING
                                        analysis_state = "TRACKING"
                                        # Adapt the reference IOD to the new stable average
                                        new_reference_iod = avg_unstable_iod
                                        logging.info(f"STATE CHANGE: -> TRACKING (Adapted Ref IOD from {reference_iod:.2f} to {new_reference_iod:.2f}, Stable Rel Dev: {relative_deviation:.4f})")
                                        reference_iod = new_reference_iod # Update local reference
                                        reference_iod_store['value'] = reference_iod # Update global store
                                        scale_factor = 1.0 # Reset scale factor relative to NEW reference
                                        metrics['reference_iod_px'] = reference_iod # Update metrics
                                        metrics['scale_factor'] = scale_factor
                                        # Buffers were already cleared when entering unstable state
                                    else:
                                        logging.debug(f"Still unstable: Avg IOD={avg_unstable_iod:.2f}, Rel Dev (StdDev/Avg)={relative_deviation:.4f} (Threshold={DISTANCE_STABLE_THRESHOLD:.3f})")
                                else:
                                    logging.debug("Average unstable IOD is zero, cannot check stability.")
                        # --- End Distance Unstable State Logic ---
                    else:
                        # Reference IOD is missing or invalid
                        scale_factor = 1.0 # Fallback scale
                        metrics['scale_factor'] = scale_factor
                        if analysis_state == "TRACKING": # If tracking was active, become unstable
                             analysis_state = "DISTANCE_UNSTABLE"
                             distance_unstable_start_time = time.time()
                             unstable_iod_history.clear()
                             logging.info("STATE CHANGE: -> DISTANCE_UNSTABLE (Reference IOD missing)")

                else: # current_iod_px <= 1 (Invalid IOD)
                    metrics['distance_status'] = "none"
                    scale_factor = 1.0 # Reset scale factor
                    metrics['scale_factor'] = scale_factor
                    distance_buffer.clear() # Clear distance buffer
                    if analysis_state == "TRACKING": # If tracking was active, enter unstable
                         analysis_state = "DISTANCE_UNSTABLE"
                         distance_unstable_start_time = time.time()
                         unstable_iod_history.clear()
                         logging.info("STATE CHANGE: -> DISTANCE_UNSTABLE (IOD calculation failed)")
                    # Reset reference IOD if calculation fails consistently? Maybe not, wait for stable recovery.
                    # reference_iod = None
                    # reference_iod_store['value'] = None
                    metrics['reference_iod_px'] = reference_iod # Log the last known reference

                # --- Centering Check (using IOD landmarks) ---
                if l_iod_pt and r_iod_pt:
                    user_x = (l_iod_pt[0] + r_iod_pt[0]) / 2
                    metrics['centering_status'] = abs(user_x - img_w // 2) < CENTERING_TOLERANCE_PX
                else:
                    metrics['centering_status'] = False

                # --- Calculate SCALED parameters for pupil detection ---
                # Use the current (potentially clamped) scale_factor calculated above
                scaled_block_size = max(3, round(BASE_ADAPTIVE_THRESH_BLOCK_SIZE * scale_factor))
                scaled_kernel_val = max(1, round(BASE_MORPH_KERNEL_SIZE_TUPLE[0] * scale_factor))
                scaled_roi_padding = max(1, round(BASE_ROI_PADDING * scale_factor))
                # Ensure odd values >= 1 (or 3 for block size)
                scaled_block_size = scaled_block_size if scaled_block_size % 2 == 1 else scaled_block_size + 1
                scaled_kernel_val = scaled_kernel_val if scaled_kernel_val % 2 == 1 else scaled_kernel_val + 1
                scaled_kernel_tuple = (scaled_kernel_val, scaled_kernel_val)

                # -- Head Pose Analysis -- (Calculated regardless of analysis state)
                # Pass the raw landmark list from results
                metrics['head_pose'], metrics['head_pose_confidence'], metrics['head_pose_p1'], metrics['head_pose_p2'] = analyze_head_pose(
                    face_landmarks, frame_dims, CAM_MATRIX, DIST_MATRIX
                )

                # -- Gaze Features (EAR, Pupil, Ratios) --
                # Calculate EAR (using standard landmarks)
                metrics['raw_ear_left'] = calculate_ear(face_landmarks, LEFT_EYE_EAR_INDICES, img_w, img_h)
                metrics['raw_ear_right'] = calculate_ear(face_landmarks, RIGHT_EYE_EAR_INDICES, img_w, img_h)

                # Pupil Detection & Ratios (Using SCALED parameters)
                # Initialize display ROIs for debug window
                left_roi_display = np.zeros((GAZE_DEBUG_WINDOW_SIZE[1], GAZE_DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8)
                right_roi_display = np.zeros((GAZE_DEBUG_WINDOW_SIZE[1], GAZE_DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8)
                left_thresh_display = np.zeros((GAZE_DEBUG_WINDOW_SIZE[1], GAZE_DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8)
                right_thresh_display = np.zeros((GAZE_DEBUG_WINDOW_SIZE[1], GAZE_DEBUG_WINDOW_SIZE[0], 3), dtype=np.uint8)

                # Left Eye Pupil & Ratios
                left_eye_pixels = [get_landmark_coords(face_landmarks, i, img_w, img_h) for i in LEFT_EYE_OUTLINE_INDICES if get_landmark_coords(face_landmarks, i, img_w, img_h)]
                if left_eye_pixels:
                    try:
                        lx, ly, lw, lh = cv2.boundingRect(np.array(left_eye_pixels))
                        # Apply scaled padding
                        lx1, ly1 = max(0, lx - scaled_roi_padding), max(0, ly - scaled_roi_padding)
                        lx2, ly2 = min(img_w, lx + lw + scaled_roi_padding), min(img_h, ly + lh + scaled_roi_padding)
                        if lx2 > lx1 + 2 and ly2 > ly1 + 2: # Min ROI size check
                            left_roi_color = frame_bgr[ly1:ly2, lx1:lx2].copy()
                            left_roi_gray = cv2.cvtColor(left_roi_color, cv2.COLOR_BGR2GRAY)
                            # Detect pupil using scaled parameters
                            pupil_coords_roi, thresh_l = detect_pupil(left_roi_gray, left_roi_color, scaled_block_size, scaled_kernel_tuple, FIXED_ADAPTIVE_THRESH_C, MIN_PUPIL_AREA_ROI_RATIO, MAX_PUPIL_AREA_ROI_RATIO, MIN_PUPIL_ASPECT_RATIO, "L")
                            if pupil_coords_roi:
                                metrics['left_pupil_abs'] = (lx1 + pupil_coords_roi[0], ly1 + pupil_coords_roi[1])
                                # Get landmarks for ratios (standard)
                                left_inner = get_landmark_coords(face_landmarks, LEFT_EYE_INNER_CORNER, img_w, img_h)
                                left_outer = get_landmark_coords(face_landmarks, LEFT_EYE_OUTER_CORNER, img_w, img_h)
                                left_upper = get_landmark_coords(face_landmarks, LEFT_EYE_UPPER_MIDPOINT, img_w, img_h)
                                left_lower = get_landmark_coords(face_landmarks, LEFT_EYE_LOWER_MIDPOINT, img_w, img_h)
                                # Calculate raw ratios only if all landmarks are valid
                                if left_inner and left_outer and left_upper and left_lower:
                                    metrics['raw_ratio_h_left'] = get_horizontal_gaze_direction_ratio(metrics['left_pupil_abs'][0], left_inner[0], left_outer[0], True)
                                    metrics['raw_ratio_v_left'] = get_vertical_gaze_direction_ratio(metrics['left_pupil_abs'][1], left_upper[1], left_lower[1])
                                else:
                                    logging.debug("Missing landmarks for left eye ratio calculation.")
                            # Prepare display windows (even if pupil not found)
                            left_roi_display = cv2.resize(left_roi_color, GAZE_DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
                            if thresh_l is not None:
                                left_thresh_display = cv2.cvtColor(cv2.resize(thresh_l, GAZE_DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                    except Exception as e_left_eye:
                         logging.debug(f"Error processing left eye ROI: {e_left_eye}")

                # Right Eye Pupil & Ratios
                right_eye_pixels = [get_landmark_coords(face_landmarks, i, img_w, img_h) for i in RIGHT_EYE_OUTLINE_INDICES if get_landmark_coords(face_landmarks, i, img_w, img_h)]
                if right_eye_pixels:
                    try:
                        rx, ry, rw, rh = cv2.boundingRect(np.array(right_eye_pixels))
                        # Apply scaled padding
                        rx1, ry1 = max(0, rx - scaled_roi_padding), max(0, ry - scaled_roi_padding)
                        rx2, ry2 = min(img_w, rx + rw + scaled_roi_padding), min(img_h, ry + rh + scaled_roi_padding)
                        if rx2 > rx1 + 2 and ry2 > ry1 + 2: # Min ROI size check
                            right_roi_color = frame_bgr[ry1:ry2, rx1:rx2].copy()
                            right_roi_gray = cv2.cvtColor(right_roi_color, cv2.COLOR_BGR2GRAY)
                            # Detect pupil using scaled parameters
                            pupil_coords_roi, thresh_r = detect_pupil(right_roi_gray, right_roi_color, scaled_block_size, scaled_kernel_tuple, FIXED_ADAPTIVE_THRESH_C, MIN_PUPIL_AREA_ROI_RATIO, MAX_PUPIL_AREA_ROI_RATIO, MIN_PUPIL_ASPECT_RATIO, "R")
                            if pupil_coords_roi:
                                metrics['right_pupil_abs'] = (rx1 + pupil_coords_roi[0], ry1 + pupil_coords_roi[1])
                                # Get landmarks for ratios (standard)
                                right_inner = get_landmark_coords(face_landmarks, RIGHT_EYE_INNER_CORNER, img_w, img_h)
                                right_outer = get_landmark_coords(face_landmarks, RIGHT_EYE_OUTER_CORNER, img_w, img_h)
                                right_upper = get_landmark_coords(face_landmarks, RIGHT_EYE_UPPER_MIDPOINT, img_w, img_h)
                                right_lower = get_landmark_coords(face_landmarks, RIGHT_EYE_LOWER_MIDPOINT, img_w, img_h)
                                # Calculate raw ratios only if all landmarks are valid
                                if right_inner and right_outer and right_upper and right_lower:
                                    metrics['raw_ratio_h_right'] = get_horizontal_gaze_direction_ratio(metrics['right_pupil_abs'][0], right_inner[0], right_outer[0], False)
                                    metrics['raw_ratio_v_right'] = get_vertical_gaze_direction_ratio(metrics['right_pupil_abs'][1], right_upper[1], right_lower[1])
                                else:
                                    logging.debug("Missing landmarks for right eye ratio calculation.")
                            # Prepare display windows
                            right_roi_display = cv2.resize(right_roi_color, GAZE_DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
                            if thresh_r is not None:
                                right_thresh_display = cv2.cvtColor(cv2.resize(thresh_r, GAZE_DEBUG_WINDOW_SIZE, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                    except Exception as e_right_eye:
                         logging.debug(f"Error processing right eye ROI: {e_right_eye}")

                # --- Smooth Gaze Values (only if tracking is stable) ---
                smoothed_ear_left, smoothed_ear_right = None, None
                smoothed_ratio_h_left, smoothed_ratio_h_right = None, None
                smoothed_ratio_v_left, smoothed_ratio_v_right = None, None

                if analysis_state == "TRACKING":
                    # Add current raw values to history deques and get smoothed value
                    smoothed_ear_left = smooth_value(left_ear_hist, metrics['raw_ear_left'])
                    smoothed_ear_right = smooth_value(right_ear_hist, metrics['raw_ear_right'])
                    smoothed_ratio_h_left = smooth_value(left_ratio_h_hist, metrics['raw_ratio_h_left'])
                    smoothed_ratio_h_right = smooth_value(right_ratio_h_hist, metrics['raw_ratio_h_right'])
                    smoothed_ratio_v_left = smooth_value(left_ratio_v_hist, metrics['raw_ratio_v_left'])
                    smoothed_ratio_v_right = smooth_value(right_ratio_v_hist, metrics['raw_ratio_v_right'])
                else: # If unstable, keep last known smoothed value (or None if buffer was empty)
                    # This helps prevent sudden jumps in displayed values when unstable
                    smoothed_ear_left = left_ear_hist[-1] if left_ear_hist else None
                    smoothed_ear_right = right_ear_hist[-1] if right_ear_hist else None
                    smoothed_ratio_h_left = left_ratio_h_hist[-1] if left_ratio_h_hist else None
                    smoothed_ratio_h_right = right_ratio_h_hist[-1] if right_ratio_h_hist else None
                    smoothed_ratio_v_left = left_ratio_v_hist[-1] if left_ratio_v_hist else None
                    smoothed_ratio_v_right = right_ratio_v_hist[-1] if right_ratio_v_hist else None

                # --- Calculate Average Smoothed Values ---
                # Average EAR
                valid_ears = [e for e in [smoothed_ear_left, smoothed_ear_right] if e is not None]
                metrics['smoothed_ear_avg'] = sum(valid_ears) / len(valid_ears) if valid_ears else None
                # Average Horizontal Ratio
                valid_h_ratios = [r for r in [smoothed_ratio_h_left, smoothed_ratio_h_right] if r is not None]
                metrics['smoothed_ratio_h_avg'] = sum(valid_h_ratios) / len(valid_h_ratios) if valid_h_ratios else None
                # Average Vertical Ratio
                valid_v_ratios = [r for r in [smoothed_ratio_v_left, smoothed_ratio_v_right] if r is not None]
                metrics['smoothed_ratio_v_avg'] = sum(valid_v_ratios) / len(valid_v_ratios) if valid_v_ratios else None

                # -- Determine Gaze Direction (only if tracking is stable and values available) --
                # This logic includes hysteresis to prevent rapid flickering between states.
                if analysis_state == "TRACKING" and \
                   metrics['smoothed_ear_avg'] is not None and \
                   metrics['smoothed_ratio_h_avg'] is not None and \
                   metrics['smoothed_ratio_v_avg'] is not None:

                    current_gaze_direction = "CENTER" # Default assumption

                    # 1. Check for Blink (Highest Priority)
                    if metrics['smoothed_ear_avg'] < dynamic_thresholds['EAR_THRESHOLD_BLINK']:
                        current_gaze_direction = "BLINK"
                    else:
                        # 2. Check Vertical Gaze (Down/Up/Center) using EAR and V_Ratio
                        # Determine vertical state based on EAR (Down/Center) with hysteresis
                        current_vertical_ear_state = "CENTER"
                        # If previously DOWN, need to cross higher CENTER threshold to switch back
                        if prev_vertical_gaze_state == "DOWN":
                            if metrics['smoothed_ear_avg'] > dynamic_thresholds['PARTIAL_SHUT_EAR_CENTER_THRESHOLD']:
                                current_vertical_ear_state = "CENTER"
                            else:
                                current_vertical_ear_state = "DOWN"
                        # If previously CENTER/UP/BLINK, need to cross lower DOWN threshold to switch
                        else:
                            if metrics['smoothed_ear_avg'] < dynamic_thresholds['PARTIAL_SHUT_EAR_DOWN_THRESHOLD']:
                                current_vertical_ear_state = "DOWN"
                            else:
                                current_vertical_ear_state = "CENTER"

                        # Determine vertical state based on V_Ratio (Up/Center) - No hysteresis needed for UP currently
                        current_vertical_ratio_state = "CENTER"
                        if metrics['smoothed_ratio_v_avg'] < dynamic_thresholds['GAZE_V_UP_THRESHOLD']:
                            current_vertical_ratio_state = "UP"

                        # Combine vertical states (Priority: Down > Up > Center)
                        if current_vertical_ear_state == "DOWN":
                            current_gaze_direction = "DOWN"
                        elif current_vertical_ratio_state == "UP":
                            current_gaze_direction = "UP"
                        else:
                            # Vertical gaze is Center, now check Horizontal
                            # 3. Check Horizontal Gaze (Left/Right/Center) using H_Ratio with hysteresis
                            current_horizontal_state = "CENTER"
                            # If previously LEFT, need to cross higher CENTER_FROM_LEFT threshold
                            if prev_horizontal_gaze_state == "LEFT":
                                if metrics['smoothed_ratio_h_avg'] > dynamic_thresholds['GAZE_H_CENTER_FROM_LEFT_THRESHOLD']:
                                    current_horizontal_state = "CENTER"
                                else:
                                    current_horizontal_state = "LEFT"
                            # If previously RIGHT, need to cross lower CENTER_FROM_RIGHT threshold
                            elif prev_horizontal_gaze_state == "RIGHT":
                                if metrics['smoothed_ratio_h_avg'] < dynamic_thresholds['GAZE_H_CENTER_FROM_RIGHT_THRESHOLD']:
                                    current_horizontal_state = "CENTER"
                                else:
                                    current_horizontal_state = "RIGHT"
                            # If previously CENTER, check against outer LEFT/RIGHT thresholds
                            else:
                                if metrics['smoothed_ratio_h_avg'] < dynamic_thresholds['GAZE_H_LEFT_THRESHOLD']:
                                    current_horizontal_state = "LEFT"
                                elif metrics['smoothed_ratio_h_avg'] > dynamic_thresholds['GAZE_H_RIGHT_THRESHOLD']:
                                    current_horizontal_state = "RIGHT"
                                else:
                                    current_horizontal_state = "CENTER"

                            current_gaze_direction = current_horizontal_state # Assign horizontal state if vertical is Center

                    # 4. Update Gaze Direction Metric and Previous States for Hysteresis
                    metrics['gaze_direction'] = current_gaze_direction
                    # Update previous states *unless* it was a blink
                    if current_gaze_direction != "BLINK":
                        # Determine combined vertical state for next frame's hysteresis
                        if current_gaze_direction == "DOWN": prev_vertical_gaze_state = "DOWN"
                        elif current_gaze_direction == "UP": prev_vertical_gaze_state = "UP"
                        else: prev_vertical_gaze_state = "CENTER" # Includes Left/Right/Center horizontal states
                        # Determine horizontal state for next frame's hysteresis
                        if current_gaze_direction == "LEFT": prev_horizontal_gaze_state = "LEFT"
                        elif current_gaze_direction == "RIGHT": prev_horizontal_gaze_state = "RIGHT"
                        else: prev_horizontal_gaze_state = "CENTER" # Includes Up/Down/Center vertical states

                elif analysis_state == "DISTANCE_UNSTABLE":
                    metrics['gaze_direction'] = "UNSTABLE"
                    # Hysteresis states were reset when entering unstable
                else: # Tracking is stable, but gaze values are None (e.g., pupil detection failed)
                    metrics['gaze_direction'] = "N/A"
                    # Reset hysteresis if gaze values lost
                    prev_vertical_gaze_state = "CENTER"
                    prev_horizontal_gaze_state = "CENTER"

            else: # No face landmarks detected
                # Clear all smoothing buffers
                left_ear_hist.clear(); right_ear_hist.clear()
                left_ratio_h_hist.clear(); right_ratio_h_hist.clear()
                left_ratio_v_hist.clear(); right_ratio_v_hist.clear()
                distance_buffer.clear()
                posture_history.clear()
                unstable_iod_history.clear()
                # Reset states and metrics
                scale_factor = 1.0 # Reset scale
                metrics['scale_factor'] = scale_factor
                metrics['distance_status'] = "none"
                metrics['gaze_direction'] = "N/A"
                metrics['head_pose'] = "N/A"
                # Reset state to TRACKING if face is lost, assuming user might reappear in a stable position
                if analysis_state == "DISTANCE_UNSTABLE":
                    logging.info("Face lost while unstable, resetting state to TRACKING.")
                analysis_state = "TRACKING"
                # Reset gaze hysteresis states
                prev_vertical_gaze_state = "CENTER"
                prev_horizontal_gaze_state = "CENTER"
                # Reset reference IOD if face lost
                reference_iod = None
                reference_iod_store['value'] = None
                metrics['reference_iod_px'] = None

            # --- Posture Analysis --- (Run regardless of unstable state, but history is cleared if unstable)
            if pose_landmarks_obj:
                (metrics['posture_score'], metrics['posture_confidence'],
                 metrics['posture_feedback'], metrics['shoulder_line_pts'],
                 metrics['shoulder_color'], posture_history) = analyze_posture(
                    pose_landmarks_obj, frame_dims, posture_history
                )
            else:
                # Clear posture history if pose is lost
                posture_history.clear()
                metrics['posture_feedback'] = "No pose detected"
                metrics['posture_score'] = 1
                metrics['posture_confidence'] = 0

            # --- Determine Gaze Confidence from Map ---
            gaze_confidence = GAZE_CONFIDENCE_MAP.get(metrics['gaze_direction'], 0)

            # --- Calculate Overall Confidence ---
            metrics['overall_confidence'] = calculate_overall_confidence(
                metrics['posture_confidence'],
                metrics['head_pose_confidence'],
                gaze_confidence
            )

            # --- FPS Update ---
            metrics['fps'] = update_fps(loop_start_time, fps_filter)
            fps_filter = metrics['fps'] # Update the filter state for next loop

            # --- Logging Metrics to CSV ---
            timestamp_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Milliseconds
            if metrics_writer and metrics_file:
                try:
                    # <<< MODIFIED: Conditionally exclude gaze_direction from the main metrics row >>>
                    row_data = [
                        timestamp_log,
                        f"{metrics['fps']:.1f}",
                        f"{metrics['distance_cm']:.1f}" if metrics['distance_cm'] is not None else "N/A",
                        "Centered" if metrics['centering_status'] else "Off-Center",
                        metrics['posture_score'],
                        metrics['posture_confidence'],
                        metrics['posture_feedback'],
                        metrics['head_pose'],
                        metrics['head_pose_confidence'],
                        # Conditionally add Gaze_Direction
                        f"{metrics['smoothed_ear_avg']:.3f}" if metrics['smoothed_ear_avg'] is not None else "N/A",
                        f"{metrics['smoothed_ratio_h_avg']:.3f}" if metrics['smoothed_ratio_h_avg'] is not None else "N/A",
                        f"{metrics['smoothed_ratio_v_avg']:.3f}" if metrics['smoothed_ratio_v_avg'] is not None else "N/A",
                        metrics['overall_confidence'],
                        analysis_state, # Log the current analysis state
                        f"{metrics['reference_iod_px']:.2f}" if metrics['reference_iod_px'] is not None else "N/A",
                        f"{metrics['scale_factor']:.3f}" if metrics['scale_factor'] is not None else "N/A"
                    ]
                    if not separate_gaze_log:
                         # Insert Gaze_Direction after Head_Pose_Confidence (index 9)
                         row_data.insert(9, metrics['gaze_direction'])

                    metrics_writer.writerow(row_data)
                    # Flush occasionally (e.g., every 5 seconds) to ensure data is written
                    if int(loop_start_time) % 5 == 0: metrics_file.flush()
                except ValueError as e: # Catch potential file closed errors
                    logging.error(f"Error writing metrics to CSV (file might be closed): {e}")
                    metrics_writer = None # Stop trying to write
                except Exception as e:
                    logging.error(f"Unexpected error writing metrics to CSV: {e}")

            # --- Logging Gaze Direction Separately (if enabled) ---
            # <<< MODIFIED: Use the separate_gaze_log flag passed from main >>>
            if separate_gaze_log and gaze_writer and gaze_file: # Log gaze direction regardless of state (includes UNSTABLE/N/A)
                try:
                    gaze_writer.writerow([timestamp_log, metrics['gaze_direction']])
                    if int(loop_start_time) % 5 == 0: gaze_file.flush()
                except ValueError as e:
                    logging.error(f"Error writing gaze log to CSV (file might be closed): {e}")
                    gaze_writer = None
                except Exception as e:
                    logging.error(f"Unexpected error writing gaze log to CSV: {e}")

            # --- Drawing Overlay ---
            frame_bgr = draw_overlay(frame_bgr, metrics, COLORS, phase="Analysis", analysis_state=analysis_state)

            # --- Display Main Window ---
            cv2.imshow(MAIN_WINDOW_NAME, frame_bgr)

            # --- Optional Debug Windows ---
            if SHOW_DEBUG_WINDOWS:
                # Combine ROI and Threshold images horizontally for display
                if left_roi_display is not None and left_thresh_display is not None:
                     # Ensure they have the same height before hstack
                     h_roi = left_roi_display.shape[0]
                     h_thresh = left_thresh_display.shape[0]
                     if h_roi == h_thresh and h_roi > 0: # Add check for non-zero height
                         combined_left = np.hstack((left_roi_display, left_thresh_display))
                         cv2.imshow('Left Eye Debug (ROI | Threshold)', combined_left)
                     else:
                          logging.debug("Left debug window height mismatch or zero height.")
                if right_roi_display is not None and right_thresh_display is not None:
                     h_roi = right_roi_display.shape[0]
                     h_thresh = right_thresh_display.shape[0]
                     if h_roi == h_thresh and h_roi > 0: # Add check for non-zero height
                          combined_right = np.hstack((right_roi_display, right_thresh_display))
                          cv2.imshow('Right Eye Debug (ROI | Threshold)', combined_right)
                     else:
                          logging.debug("Right debug window height mismatch or zero height.")

            # --- Handle User Input (Exit or Recalibrate) ---
            key = cv2.waitKey(5) & 0xFF # Use waitKey(5) for ~200fps theoretical max, adjust if needed
            if key == 27 or key == ord('q'): # ESC or Q to quit
                logging.info("Exit requested by user during main analysis.")
                return "EXIT" # Signal normal exit
            elif key == ord('r'): # 'R' to restart calibration
                 logging.info("User requested recalibration. Returning to distance monitoring.")
                 return "RECALIBRATE" # Signal main loop to restart

    except Exception as e:
        logging.critical(f"Unhandled exception in main analysis loop: {e}", exc_info=True)
        return "ERROR" # Signal critical error
    finally:
        # --- Cleanup CSV Files ---
        if metrics_file:
            try:
                metrics_file.close()
                logging.info("Metrics CSV file closed.")
            except Exception as e:
                logging.error(f"Error closing metrics CSV file: {e}")
        if gaze_file:
            try:
                gaze_file.close()
                logging.info("Gaze log CSV file closed.")
            except Exception as e:
                logging.error(f"Error closing gaze log CSV file: {e}")
    # Should only be reached if loop terminates unexpectedly without user input
    logging.warning("Main analysis loop terminated unexpectedly.")
    return "EXIT"




# --- Main Orchestrator ---
def interview_test(video_path, interview_id):

    log_file, csv_metrics_path = setup_logging(
        interview_id,
        log_to_file=SAVE_METRICS_LOG,
        log_to_console=LOG_TO_CONSOLE,
        separate_gaze_log=SAVE_GAZE_LOG_SEPARATE # Pass the flag here
    )
    csv_gaze_path = None
        
    if SAVE_GAZE_LOG_SEPARATE and csv_metrics_path:
         log_dir = os.path.dirname(csv_metrics_path)
         timestamp = os.path.basename(csv_metrics_path).replace('metrics_', '').replace('.csv', '')
         csv_gaze_path = os.path.join(log_dir, f'gaze_log_{timestamp}.csv')
    elif SAVE_GAZE_LOG_SEPARATE:
         log_dir = f'{BASE_PATH}/SoftSkills/logs'
         if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
         csv_gaze_path = os.path.join(log_dir, f'gaze_log_{timestamp}.csv')


    # --- Initialize Video Capture ---
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.critical("Error: Could not open video capture device (index 0).")
        print("CRITICAL ERROR: Cinterview_testould not open video capture. Check camera connection and permissions.")
        sys.exit(1)

    # Attempt to set a higher resolution, but processing is done on resized frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logging.info(f"Camera requested 1280x720, got {int(actual_width)}x{int(actual_height)}")

    # --- Create Main Display Window ---
    # cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL) # Allow resizing
    # cv2.resizeWindow(MAIN_WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT) # Initial size


    # mp_drawing = mp.solutions.drawing_utils # Optional for drawing landmarks
    # mp_drawing_styles = mp.solutions.drawing_styles # Optional for styled drawing

    try:
        # --- Main Workflow Loop (Allows restarting) ---

        # Get stored interview parameters
        dynamic_gaze_thresholds, reference_iod_store, calibrated_focal_length = get_interview_parameters(interview_id)

        # --- Phase 3: Main Analysis ---
        # Pass the potentially updated reference_iod_store and thresholds.
        # <<< MODIFIED: Pass SAVE_GAZE_LOG_SEPARATE to run_main_analysis >>>


        analysis_result = run_main_analysis(
            cap, face_mesh, pose,
            dynamic_gaze_thresholds,
            calibrated_focal_length,
            reference_iod_store, # Pass the dict containing the reference IOD
            csv_metrics_path,
            csv_gaze_path,
            SAVE_GAZE_LOG_SEPARATE # Pass the flag here
        )

        # --- Handle Analysis Result ---
        if analysis_result == "RECALIBRATE":
            logging.info("Recalibration requested. Restarting workflow from distance phase.")
            # Clear any potentially remaining debug windows before restarting
            if SHOW_DEBUG_WINDOWS:
                try:
                    cv2.destroyWindow('Left Eye Debug (ROI | Threshold)')
                    cv2.destroyWindow('Right Eye Debug (ROI | Threshold)')
                except cv2.error:
                    pass # Ignore errors if windows don't exist
                cv2.waitKey(1)
        else: # Includes "EXIT" or "ERROR"
            logging.info(f"Analysis phase ended with status: {analysis_result}. Exiting workflow.")

    except Exception as e:
        logging.critical(f"A critical error occurred in the main orchestrator: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}\nCheck the log file for details: {log_file}")
    finally:
        # --- Cleanup Resources ---
        logging.info("Releasing video capture...")
        if cap and cap.isOpened():
             cap.release()
        logging.info("Destroying OpenCV windows...")
        cv2.destroyAllWindows()

        
if __name__ == '__main__':
    pass
