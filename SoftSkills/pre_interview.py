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
from SoftSkills.soft_skills_all import calculate_dynamic_gaze_thresholds, calculate_ear, detect_pupil, get_face_landmarks, get_horizontal_gaze_direction_ratio, get_landmark_coords, calculate_distance, get_vertical_gaze_direction_ratio, smooth_value, update_fps, draw_overlay
from globals import BASE_PATH



'''
Phase 1: Monitors distance and centering until user proceeds.
Calibrates focal length based on the first valid IOD detection.
Returns: ProceedToNextPhase: bool, CalibratedFocalLength: float?
'''

def run_distance_monitoring(frame, state):

    # Resize and flip frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # Process with Face Mesh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True
    frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    face_landmarks = get_face_landmarks(results)
    loop_start_time = time.time()


    metrics = {
        'distance_cm': None,
        'distance_status': 'none', # 'none', 'close', 'good', 'far'
        'centering_status': False,
        'user_x': width // 2, # Default user x position
        'fps': state['fps_filter'] # Add FPS to metrics for drawing
    }


    focal_length = state.get('focal_length')

    if face_landmarks:
        # --- Calculate IOD using configured landmarks ---
        l_iod_pt = get_landmark_coords(face_landmarks, LEFT_EYE_IOD_LM, width, height)
        r_iod_pt = get_landmark_coords(face_landmarks, RIGHT_EYE_IOD_LM, width, height)
        current_iod_px = calculate_distance(l_iod_pt, r_iod_pt)

        if current_iod_px > 1: # Need a valid pixel distance
            # --- Focal Length Calibration (if not already done) ---
            if focal_length is None:
                # Estimate focal length using the first valid IOD and assumed distance
                focal_length = (current_iod_px * FOCAL_LENGTH_CALIBRATION_DISTANCE_CM) / REF_EYE_DISTANCE_CM
                state['focal_length'] = focal_length
                logging.info(f"Focal length calibrated: {focal_length:.2f} (based on IOD {current_iod_px:.2f}px at assumed {FOCAL_LENGTH_CALIBRATION_DISTANCE_CM}cm)")

            # --- Distance Calculation ---
            if focal_length is not None and focal_length > 0:
                raw_dist_cm = (REF_EYE_DISTANCE_CM * focal_length) / current_iod_px
                metrics['distance_cm'] = smooth_value(state['distance_buffer'], raw_dist_cm)

                # Determine distance status based on smoothed value
                if metrics['distance_cm'] is not None:
                    if metrics['distance_cm'] < MIN_DISTANCE_CM: metrics['distance_status'] = "close"
                    elif metrics['distance_cm'] > MAX_DISTANCE_CM: metrics['distance_status'] = "far"
                    else: metrics['distance_status'] = "good"
                else:
                    metrics['distance_status'] = "none" # Smoothing buffer not full yet
            else:
                metrics['distance_status'] = "none" # Focal length invalid

            # --- Centering Check (using the same IOD landmarks) ---
            if l_iod_pt and r_iod_pt:
                metrics['user_x'] = (l_iod_pt[0] + r_iod_pt[0]) / 2
                metrics['centering_status'] = abs(metrics['user_x'] - width // 2) < CENTERING_TOLERANCE_PX
            else:
                metrics['centering_status'] = False # Landmarks invalid for centering
        else:
            # Invalid IOD measurement
            metrics['distance_status'] = "none"
            state['distance_buffer'].clear()
            # Don't reset focal length here, wait for consistent failure or face loss
    else:
        # No face detected
        state['distance_buffer'].clear()
        state['focal_length'] = None # Reset focal length calibration if face is lost
        metrics['distance_status'] = "none"
        metrics['centering_status'] = False

    # --- FPS Update ---
    metrics['fps'] = update_fps(loop_start_time, state['fps_filter'])
    state['fps_filter'] = metrics['fps'] # Update filter state

    # --- Draw Distance UI using the helper ---
    frame_bgr = draw_overlay(frame_bgr, metrics, COLORS, phase="Distance")

    # --- Display Frame ---

    # Check user input
    phase_completed = state.get('user_proceed', False) or state.get('user_exit', False)
    if phase_completed:
        state['user_proceed'] = False
        state['user_exit'] = False
        if state.get('user_exit'):
            logging.info("Distance monitoring cancelled by user.")
            return {
                'frame': frame_bgr,
                'metrics': metrics,
                'phase_completed': True,
                'focal_length': focal_length,
                'proceed_to_calibration': False
            }
        logging.info("Distance monitoring complete, proceeding to gaze calibration.")
        return {
            'frame': frame_bgr,
            'metrics': metrics,
            'phase_completed': True,
            'focal_length': focal_length,
            'proceed_to_calibration': True
        }

    return {'frame': frame_bgr, 'metrics': metrics, 'phase_completed': False}



'''
Phase 2: Calibrates gaze tracking thresholds (EAR, H/V Ratios).
Collects data only when the user's IOD is stable relative to the start of calibration.
Updates `dynamic_thresholds` and `reference_iod_store` in-place.
Returns: True if calibration was successful (thresholds calculated), False otherwise (defaults used).
'''
def run_gaze_calibration(frame, state):

    # calibration_data = {"CENTER": {'ear': [], 'ratio_h': [], 'ratio_v': []}}
    calibration_state = state['calibration_state']

    loop_start_time = time.time()
    
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))    
    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True
    frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    face_landmarks = get_face_landmarks(results)

    metrics = {
            'instruction_text': "", 'countdown_text': "", 'status_text': "",
            'instruction_color': CALIBRATION_INSTRUCTION_COLOR,
            'countdown_color': CALIBRATION_TEXT_COLOR,
            'status_color': CALIBRATION_TEXT_COLOR,
            'fps': state['fps_filter']
    }
    is_stable_for_collection = False # Reset stability flag each frame

    # --- State Machine for Calibration ---
    if calibration_state == "INSTRUCTIONS":
        metrics['instruction_text'] = "Ready? Look at center point & press 'C' to start."
        metrics['countdown_text'] = "(ESC to Skip Calibration)"

        # key_instr = cv2.waitKey(5) & 0xFF
        if state.get('user_proceed'):
            logging.info("User initiated gaze calibration countdown.")
            state['calibration_state'] = "CALIBRATING"
            state['calibration_start_time'] = time.time()
            # Clear previous calibration attempts' data
            state['calibration_data'] = {"CENTER": {'ear': [], 'ratio_h': [], 'ratio_v': []}}
            state['cal_ear_hist'].clear()
            state['cal_ratio_h_hist'].clear()
            state['cal_ratio_v_hist'].clear()

            state['initial_calibration_iod'] = None
            state['user_proceed'] = False
            # reference_iod_store['value'] = None # Clear global store as well
        
        elif state.get('user_exit'):
            logging.warning("Gaze calibration skipped by user during instruction phase.")
            state['user_exit'] = False
            return {
                'frame': frame_bgr,
                'metrics': metrics,
                'phase_completed': True,
                'calibration_success': False,
                'dynamic_gaze_thresholds': state['dynamic_gaze_thresholds'],
                'reference_iod_store': state['reference_iod_store']
            }


    elif calibration_state == "CALIBRATING":
        elapsed_time = time.time() - state['calibration_start_time']
        remaining_time = max(0, GAZE_CALIBRATION_DURATION_SEC - elapsed_time)

        metrics['instruction_text'] = "Calibrating: Keep looking at the center point"
        metrics['countdown_text'] = f"Time: {remaining_time:.1f}s ('R' Restart, ESC Skip)"
        metrics['countdown_color'] = CALIBRATION_COUNTDOWN_COLOR

        if face_landmarks:
            # --- IOD Calculation for Stability Check & Scaling ---
            # Use the *configured* IOD landmarks for consistency
            l_iod_pt = get_landmark_coords(face_landmarks, LEFT_EYE_IOD_LM, img_w, img_h)
            r_iod_pt = get_landmark_coords(face_landmarks, RIGHT_EYE_IOD_LM, img_w, img_h)
            current_iod_px = calculate_distance(l_iod_pt, r_iod_pt)

            if current_iod_px > 1:
                # Set the initial reference IOD on the first valid frame *after* 'C' was pressed
                if state['initial_calibration_iod'] is None:
                    state['initial_calibration_iod'] = current_iod_px

                    logging.info(f"Gaze Calibration: Initial Reference IOD set to {state['initial_calibration_iod']:.2f}px (using {IOD_LANDMARK_DESC})")
                    # Set the global reference IOD store here as well. This will be the reference
                    # used in the main analysis phase unless adapted later.
                    state['reference_iod_store']['value'] = state['initial_calibration_iod']

                # Calculate current scale factor relative to the *initial* IOD of this calibration run
                current_scale_factor_from_initial = current_iod_px / state['initial_calibration_iod'] if state['initial_calibration_iod'] > 0 else 1.0

                # Clamp scale factor (Improvement)
                cal_scale_factor = np.clip(current_scale_factor_from_initial, MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
                if cal_scale_factor != current_scale_factor_from_initial:
                    logging.debug(f"Calibration scale factor clamped: Original={current_scale_factor_from_initial:.3f}, Clamped={cal_scale_factor:.3f}")


                # Check if distance is stable enough for data collection (compare to initial IOD)
                if abs(current_scale_factor_from_initial - 1.0) <= CALIBRATION_IOD_TOLERANCE:
                    is_stable_for_collection = True
                    metrics['status_text'] = "Collecting Data..."
                    metrics['status_color'] = CALIBRATION_STABLE_COLOR
                else:
                    is_stable_for_collection = False
                    metrics['status_text'] = "Hold Still / Adjust Distance"
                    metrics['status_color'] = CALIBRATION_UNSTABLE_COLOR
                    # logging.debug(f"Calibration unstable: IOD={current_iod_px:.2f}, Initial={initial_calibration_iod:.2f}, Scale={current_scale_factor_from_initial:.3f} (Tolerance={CALIBRATION_IOD_TOLERANCE:.2f})")

            else:
                # IOD calculation failed
                cal_scale_factor = 1.0 # Default scale
                is_stable_for_collection = False
                metrics['status_text'] = "Cannot detect eyes clearly"
                metrics['status_color'] = CALIBRATION_FAILURE_COLOR
                state['initial_calibration_iod'] = None # Reset initial IOD if eyes lost
                state['reference_iod_store']['value'] = None

            # --- Calculate SCALED parameters for pupil detection ---
            # Scale base parameters by the current CLAMPED scale factor
            scaled_block_size = max(3, round(BASE_ADAPTIVE_THRESH_BLOCK_SIZE * cal_scale_factor))
            scaled_kernel_val = max(1, round(BASE_MORPH_KERNEL_SIZE_TUPLE[0] * cal_scale_factor))
            scaled_roi_padding = max(1, round(BASE_ROI_PADDING * cal_scale_factor))
            # Ensure odd values >= 1 (or 3 for block size)
            scaled_block_size = scaled_block_size if scaled_block_size % 2 == 1 else scaled_block_size + 1
            scaled_kernel_val = scaled_kernel_val if scaled_kernel_val % 2 == 1 else scaled_kernel_val + 1
            scaled_kernel_tuple = (scaled_kernel_val, scaled_kernel_val)

            # --- Collect Gaze Data ONLY IF STABLE ---
            if is_stable_for_collection:
                # Calculate EAR (using standard landmarks)
                left_ear = calculate_ear(face_landmarks, LEFT_EYE_EAR_INDICES, img_w, img_h)
                right_ear = calculate_ear(face_landmarks, RIGHT_EYE_EAR_INDICES, img_w, img_h)
                avg_ear = (left_ear + right_ear) / 2.0 if left_ear is not None and right_ear is not None else None

                # Calculate Ratios (requires pupil detection using scaled params)
                left_pupil_center_abs, right_pupil_center_abs = None, None
                left_raw_ratio_h, right_raw_ratio_h = None, None
                left_raw_ratio_v, right_raw_ratio_v = None, None

                # --- Simplified Pupil Detection for Calibration (less drawing overhead) ---
                # Left Eye ROI & Pupil
                left_eye_pixels = [get_landmark_coords(face_landmarks, i, img_w, img_h) for i in LEFT_EYE_OUTLINE_INDICES if get_landmark_coords(face_landmarks, i, img_w, img_h)]
                if left_eye_pixels:
                    try:
                        lx, ly, lw, lh = cv2.boundingRect(np.array(left_eye_pixels))
                        # Apply scaled padding
                        lx1, ly1 = max(0, lx - scaled_roi_padding), max(0, ly - scaled_roi_padding)
                        lx2, ly2 = min(img_w, lx + lw + scaled_roi_padding), min(img_h, ly + lh + scaled_roi_padding)
                        if lx2 > lx1 + 2 and ly2 > ly1 + 2: # Min ROI size check
                            left_roi_color_cal = frame_bgr[ly1:ly2, lx1:lx2].copy() # Use separate copy
                            left_roi_gray = cv2.cvtColor(left_roi_color_cal, cv2.COLOR_BGR2GRAY)
                            pupil_coords_roi, _ = detect_pupil(left_roi_gray, left_roi_color_cal, scaled_block_size, scaled_kernel_tuple, FIXED_ADAPTIVE_THRESH_C, MIN_PUPIL_AREA_ROI_RATIO, MAX_PUPIL_AREA_ROI_RATIO, MIN_PUPIL_ASPECT_RATIO, "L_cal")
                            if pupil_coords_roi:
                                # Convert ROI coords to absolute frame coords
                                left_pupil_center_abs = (lx1 + pupil_coords_roi[0], ly1 + pupil_coords_roi[1])
                                # Get landmarks needed for ratios (standard landmarks)
                                left_inner = get_landmark_coords(face_landmarks, LEFT_EYE_INNER_CORNER, img_w, img_h)
                                left_outer = get_landmark_coords(face_landmarks, LEFT_EYE_OUTER_CORNER, img_w, img_h)
                                left_upper = get_landmark_coords(face_landmarks, LEFT_EYE_UPPER_MIDPOINT, img_w, img_h)
                                left_lower = get_landmark_coords(face_landmarks, LEFT_EYE_LOWER_MIDPOINT, img_w, img_h)
                                # Calculate ratios only if all landmarks are valid
                                if left_inner and left_outer and left_upper and left_lower:
                                    left_raw_ratio_h = get_horizontal_gaze_direction_ratio(left_pupil_center_abs[0], left_inner[0], left_outer[0], True)
                                    left_raw_ratio_v = get_vertical_gaze_direction_ratio(left_pupil_center_abs[1], left_upper[1], left_lower[1])
                                else:
                                    logging.debug("Missing landmarks for left eye ratio calculation during calibration.")

                    except Exception as e_l_cal: logging.debug(f"Error in left eye cal processing: {e_l_cal}")

                # Right Eye ROI & Pupil
                right_eye_pixels = [get_landmark_coords(face_landmarks, i, img_w, img_h) for i in RIGHT_EYE_OUTLINE_INDICES if get_landmark_coords(face_landmarks, i, img_w, img_h)]
                if right_eye_pixels:
                        try:
                            rx, ry, rw, rh = cv2.boundingRect(np.array(right_eye_pixels))
                            # Apply scaled padding
                            rx1, ry1 = max(0, rx - scaled_roi_padding), max(0, ry - scaled_roi_padding)
                            rx2, ry2 = min(img_w, rx + rw + scaled_roi_padding), min(img_h, ry + rh + scaled_roi_padding)
                            if rx2 > rx1 + 2 and ry2 > ry1 + 2: # Min ROI size check
                                right_roi_color_cal = frame_bgr[ry1:ry2, rx1:rx2].copy() # Use separate copy
                                right_roi_gray = cv2.cvtColor(right_roi_color_cal, cv2.COLOR_BGR2GRAY)
                                pupil_coords_roi, _ = detect_pupil(right_roi_gray, right_roi_color_cal, scaled_block_size, scaled_kernel_tuple, FIXED_ADAPTIVE_THRESH_C, MIN_PUPIL_AREA_ROI_RATIO, MAX_PUPIL_AREA_ROI_RATIO, MIN_PUPIL_ASPECT_RATIO, "R_cal")
                                if pupil_coords_roi:
                                    # Convert ROI coords to absolute frame coords
                                    right_pupil_center_abs = (rx1 + pupil_coords_roi[0], ry1 + pupil_coords_roi[1])
                                    # Get landmarks needed for ratios (standard landmarks)
                                    right_inner = get_landmark_coords(face_landmarks, RIGHT_EYE_INNER_CORNER, img_w, img_h)
                                    right_outer = get_landmark_coords(face_landmarks, RIGHT_EYE_OUTER_CORNER, img_w, img_h)
                                    right_upper = get_landmark_coords(face_landmarks, RIGHT_EYE_UPPER_MIDPOINT, img_w, img_h)
                                    right_lower = get_landmark_coords(face_landmarks, RIGHT_EYE_LOWER_MIDPOINT, img_w, img_h)
                                    # Calculate ratios only if all landmarks are valid
                                    if right_inner and right_outer and right_upper and right_lower:
                                        right_raw_ratio_h = get_horizontal_gaze_direction_ratio(right_pupil_center_abs[0], right_inner[0], right_outer[0], False)
                                        right_raw_ratio_v = get_vertical_gaze_direction_ratio(right_pupil_center_abs[1], right_upper[1], right_lower[1])
                                    else:1
                                        # logging.debug("Missing landmarks for right eye ratio calculation.")

                        except Exception as e_r_cal: logging.debug(f"Error in right eye cal processing: {e_r_cal}")


                # Average the valid ratios from both eyes
                avg_ratio_h = None
                valid_h_ratios = [r for r in [left_raw_ratio_h, right_raw_ratio_h] if r is not None]
                if valid_h_ratios: avg_ratio_h = sum(valid_h_ratios) / len(valid_h_ratios)

                avg_ratio_v = None
                valid_v_ratios = [r for r in [left_raw_ratio_v, right_raw_ratio_v] if r is not None]
                if valid_v_ratios: avg_ratio_v = sum(valid_v_ratios) / len(valid_v_ratios)

                # Smooth values using the calibration deques
                smoothed_ear = smooth_value(state['cal_ear_hist'], avg_ear)
                smoothed_ratio_h = smooth_value(state['cal_ratio_h_hist'], avg_ratio_h)
                smoothed_ratio_v = smooth_value(state['cal_ratio_v_hist'], avg_ratio_v)

                # Add smoothed values to calibration data if the smoothing buffer is reasonably full
                # And ensure the values are not None before adding
                if len(state['cal_ear_hist']) >= SMOOTHING_CAL // 2:
                    if smoothed_ear is not None: state['calibration_data']["CENTER"]['ear'].append(smoothed_ear)
                    if smoothed_ratio_h is not None: state['calibration_data']["CENTER"]['ratio_h'].append(smoothed_ratio_h)
                    if smoothed_ratio_v is not None: state['calibration_data']["CENTER"]['ratio_v'].append(smoothed_ratio_v)
            # --- End of Stable Data Collection Block ---
        else: # No face landmarks detected
                metrics['status_text'] = "Face not detected"
                metrics['status_color'] = CALIBRATION_FAILURE_COLOR
                initial_calibration_iod = None # Reset initial IOD if face lost
                state['reference_iod_store']['value'] = None # Clear global reference

        # --- Check for Calibration Completion or User Input ---
        if elapsed_time >= GAZE_CALIBRATION_DURATION_SEC  or state.get('user_exit'):
            # Time's up, attempt to calculate thresholds
            num_samples = len(state['calibration_data']["CENTER"]['ear']) # Check length of one list (assume others similar)
            
            if num_samples >= MIN_CALIBRATION_SAMPLES and not state.get('user_exit'):
                
                calibration_success = calculate_dynamic_gaze_thresholds(state['calibration_data'], state['dynamic_gaze_thresholds']) 
                
                if calibration_success:
                    metrics['instruction_text'] = "Calibration Successful!"
                    metrics['instruction_color'] = CALIBRATION_SUCCESS_COLOR
                    metrics['countdown_text'] = ""
                    metrics['status_text'] = ""
                    # logging.info(f"Gaze calibration successful ({num_samples} stable samples). Dynamic thresholds applied.")
                else:
                    metrics['instruction_text'] = "Calibration Error. Using Defaults."
                    metrics['instruction_color'] = CALIBRATION_FAILURE_COLOR
                    metrics['countdown_text'] = ""
                    metrics['status_text'] = ""
                    # logging.warning("Gaze threshold calculation failed despite sufficient samples. Using default thresholds.")
            else:
                metrics['instruction_text'] = "Calibration Failed: Too much movement."
                metrics['instruction_color'] = CALIBRATION_FAILURE_COLOR
                metrics['countdown_text'] = "Using default thresholds."
                metrics['countdown_color'] = CALIBRATION_FAILURE_COLOR
                metrics['status_text'] = "Press 'R' to Retry or ESC to Skip."
                # logging.warning(f"Gaze calibration failed: Insufficient stable samples ({num_samples}/{MIN_CALIBRATION_SAMPLES}). Using defaults.")
            
            
            frame_bgr = draw_overlay(frame_bgr, metrics, COLORS, phase="Calibration")
            state['user_exit'] = False
            return {
                'frame': frame_bgr,
                'metrics': metrics,
                'phase_completed': True,
                'calibration_success': calibration_success,
                'dynamic_gaze_thresholds': state['dynamic_gaze_thresholds'],
                'reference_iod_store': state['reference_iod_store']
            }

        # Check for intermediate user input (Restart or Skip) during countdown
        # key_cal = cv2.waitKey(5) & 0xFF
        # if key_cal == ord('r'): # Restart calibration
        #     logging.info("Restarting gaze calibration.")
        #     calibration_state = "INSTRUCTIONS" # Go back to instructions
        # elif key_cal == 27: # ESC to skip
        #     logging.warning("Gaze calibration skipped by user during calibration. Using default thresholds.")
        #     # Keep reference_iod_store['value'] if it was set, otherwise it's None
        #     return False # Skipped

    
    
    metrics['fps'] = update_fps(loop_start_time, state['fps_filter'])
    state['fps_filter'] = metrics['fps']

    frame_bgr = draw_overlay(frame_bgr, metrics, COLORS, phase="Calibration")
    return {'frame': frame_bgr, 'metrics': metrics, 'phase_completed': False}






def init_pre_interview(interview_id):
    pass    


def pre_interview_test(frame, state):

    phase = state.get('phase', 'distance_monitoring')
    
    # Initialize state variables if not present
    if state.get('distance_buffer') is None:
        state['distance_buffer'] = deque(maxlen=DISTANCE_SMOOTHING)
    if state.get('cal_ear_hist') is None:
        state['cal_ear_hist'] = deque(maxlen=SMOOTHING_CAL)
        state['cal_ratio_h_hist'] = deque(maxlen=SMOOTHING_CAL)
        state['cal_ratio_v_hist'] = deque(maxlen=SMOOTHING_CAL)

    if phase == 'distance_monitoring':
        return run_distance_monitoring(frame, state)
    elif phase == 'gaze_calibration':
        return run_gaze_calibration(frame, state)
    else:
        1
        # return {'frame': frame_bgr, 'metrics': {}, 'phase_completed': False}


parameters_file_path = f'{BASE_PATH}/SoftSkills/interviews_parameters.json'

def save_parameters(interview_id, dynamic_gaze_thresholds, reference_iod_store, calibrated_focal_length, calibration_success):
    
    try:
        with open(parameters_file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    data[interview_id] = {
        "dynamic_gaze_thresholds": dynamic_gaze_thresholds,
        "reference_iod_store": reference_iod_store,
        "calibrated_focal_length": calibrated_focal_length
    }

    with open(parameters_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    if calibration_success:
        logging.info("Gaze calibration successful. Using dynamic thresholds.")
    else:
        logging.warning("Gaze calibration failed or skipped. Using default thresholds.")



def get_interview_parameters(interview_id):
    with open(parameters_file_path, 'r') as file:
        data = json.load(file)
        
    interviewee_params = data.get(interview_id, {})

    gaze_thresholds = interviewee_params.get("dynamic_gaze_thresholds", {})
    reference_iod_store = interviewee_params.get("reference_iod_store", {})
    focal_length = interviewee_params.get("calibrated_focal_length", 0.0)

    return gaze_thresholds, reference_iod_store, focal_length
