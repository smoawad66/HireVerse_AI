import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import csv
from datetime import datetime
import sys
import os
from collections import deque




def log_metrics(csv_path, metrics):
    """Log detailed metrics to CSV file"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            metrics['posture_score'],
            metrics['posture_confidence'],
            metrics['posture_feedback'],
            metrics['head_pose'],
            metrics['head_pose_confidence'],
            metrics['overall_confidence'],
            metrics['fps']
        ])

# --- Global Constants and UI Settings ---

# Distance Monitoring Constants (Less relevant for video, but keeping for structure)
REF_EYE_DISTANCE_CM = 6.3
MIN_DISTANCE_CM, MAX_DISTANCE_CM = 57, 65
FOCAL_LENGTH_CALIBRATION_DISTANCE_CM = 60 # Assume user is ~60cm away during calibration

# Posture and Head Pose Analysis Constants
POSTURE_SMOOTHING = 10
SHOULDER_DIFF_THRESHOLD = 0.05
HEAD_DISTANCE_THRESHOLD = 0.10 # Threshold for head slump relative to shoulders
POSTURE_WEIGHTS = np.array([0.5, 0.3, 0.2]) # Weights for shoulder level, head slump, centering
HEAD_POSE_FORWARD_THRESHOLD_Y = 1.5 # Threshold for left/right head turn angle (degrees)
HEAD_POSE_FORWARD_THRESHOLD_X = 1.0 # Threshold for looking down angle (degrees)

# Confidence Weights
POSTURE_CONFIDENCE_WEIGHT = 0.6
HEAD_POSE_CONFIDENCE_WEIGHT = 0.4

# Frame Dimensions (Should match desired processing size - will resize video frames)
FRAME_WIDTH = 800
FRAME_HEIGHT = 600

# Camera Matrix for Head Pose (Based on FRAME_WIDTH/HEIGHT)
# This will be used for head pose estimation regardless of input source (camera or video)
CAM_MATRIX = np.array([
    [FRAME_WIDTH, 0, FRAME_WIDTH / 2],
    [0, FRAME_WIDTH, FRAME_HEIGHT / 2],
    [0, 0, 1]
], dtype=np.float32)
DIST_MATRIX = np.zeros((4, 1), dtype=np.float32) # Assuming no lens distortion

# MediaPipe Landmark Indices for Head Pose (Nose, Right Eye, Left Eye, Right Mouth, Left Mouth)
LANDMARK_INDICES_HEAD_POSE = [1, 199, 33, 263, 61, 291]

# UI Colors (Used for drawing overlay on video frames)
COLORS = {
    'text': (255, 255, 255), # White
    'good': (0, 255, 0),     # Green
    'close': (0, 0, 255),    # Red
    'far': (0, 165, 255),    # Orange
    'none': (255, 0, 0),     # Blue (for 'no face detected' status text)
    'center': (0, 255, 0),   # Green
    'not_center': (0, 0, 255)# Red
}

# UI Element Settings (Used for drawing overlay)
BAR_SETTINGS = (50, 120, 350, 70) # x, y, width, height for the distance bar (less relevant for video, but drawing logic remains)


# --- Helper Functions for analyze_interview_presence ---

def _initialize_metrics():
    """Initializes the metrics dictionary for a frame."""
    return {
        'posture_score': 1,
        'posture_confidence': 20,
        'posture_feedback': "No pose detected",
        'head_pose': "N/A",
        'head_pose_confidence': 0,
        'overall_confidence': 0,
        'fps': 0 # FPS will be calculated based on processing speed, not video's original FPS
    }

def _analyze_head_pose(face_landmarks, frame_dims, cam_matrix, dist_matrix, landmark_indices, thresholds):
    """Analyzes head pose from face landmarks."""
    width, height = frame_dims
    nose_2d = None
    face_2d = []
    face_3d = []
    p1 = p2 = None
    head_pose = "N/A"
    head_pose_confidence = 0

    # Extract 2D and 3D points for head pose estimation
    for idx in landmark_indices:
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            # Convert normalized coordinates to pixel coordinates
            x, y = int(lm.x * width), int(lm.y * height)
            face_2d.append([x, y])
            # Use Z coordinate from MediaPipe for 3D point (relative depth)
            face_3d.append([x, y, lm.z])

            if idx == 1:  # Nose tip landmark index
                nose_2d = (x, y)

    # Perform PnP (Perspective-n-Point) if enough landmarks are found
    if nose_2d and len(face_2d) == len(landmark_indices):
        try:
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # SolvePnP requires corresponding 3D model points and 2D image points.
            # Here, we're using the 2D pixel coordinates as the "image points"
            # and creating synthetic "3D model points" using the 2D pixel coords
            # plus the relative Z coordinate from MediaPipe.
            success_pnp, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_DLS
            )

            if success_pnp:
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rot_vec)
                # Decompose the rotation matrix to get Euler angles (pitch, yaw, roll)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                # Convert angles from radians to degrees (RQDecomp3x3 returns radians)
                # Multiplying by 360 is mathematically incorrect for radians to degrees,
                # but maintaining original code's scaling to match its thresholds.
                x_angle = angles[0] * 360  # Pitch (looking up/down)
                y_angle = angles[1] * 360  # Yaw (looking left/right)
                z_angle = angles[2] * 360  # Roll (tilting head side to side)

                # Determine head direction based on yaw (y_angle) and pitch (x_angle)
                if y_angle < -thresholds['y_left_right'] * 2:
                    head_pose = "Looking Left"
                elif y_angle > thresholds['y_left_right'] * 2:
                    head_pose = "Looking Right"
                elif x_angle < -thresholds['x_down']:
                    head_pose = "Looking Down"
                elif x_angle > thresholds['x_up'] * 6.5:
                    head_pose = "Looking Up"
                else:
                    head_pose = "Forward"

                # Assign confidence based on head pose
                head_pose_confidence = (
                    100 if head_pose == "Forward"
                    else 60 if "Down" in head_pose or "Up" in head_pose
                    else 30 # Left/Right or other non-forward
                )

                # Calculate endpoint for drawing the head direction line
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2_x = int(nose_2d[0] + y_angle * 5)
                p2_y = int(nose_2d[1] - x_angle * 5)
                p2 = (p2_x, p2_y)

        except Exception as e:
            logging.error(f"Head pose estimation error: {e}")

    return head_pose, head_pose_confidence, p1, p2


def _analyze_posture(pose_landmarks, frame_dims, posture_history, smoothing_length, threshold_params, weight_params):
    """Analyzes posture from pose landmarks."""
    width, height = frame_dims
    posture_score = 1
    posture_confidence = 20
    posture_feedback = "No pose detected"
    shoulder_line_pts = None
    shoulder_color = (0, 0, 255) # Default red

    try:
        landmarks = pose_landmarks.landmark
        # Get shoulder and nose landmarks using MediaPipe's enum
        sl = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        sr = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]

        # Check visibility - landmarks must be sufficiently visible to be used
        if sl.visibility > 0.5 and sr.visibility > 0.5 and nose.visibility > 0.5:
            # Calculate metrics based on normalized landmark coordinates (0 to 1)
            shoulder_center_x = (sl.x + sr.x) * 0.5
            shoulder_avg_y = (sl.y + sr.y) * 0.5
            shoulder_diff_y = abs(sl.y - sr.y) # Vertical difference between shoulders
            head_shoulder_distance_y = nose.y - shoulder_avg_y # Vertical distance from nose to shoulder midpoint

            # Store current measurements in history for smoothing
            current_measurement = np.array([
                shoulder_diff_y,
                head_shoulder_distance_y,
                shoulder_center_x
            ])
            posture_history.append(current_measurement)

            # Get pixel coordinates for drawing the shoulder line
            shoulder_line_pts = (
                (int(sl.x * width), int(sl.y * height)),
                (int(sr.x * width), int(sr.y * height))
            )

            # Perform analysis only if history is sufficiently filled for smoothing
            if len(posture_history) >= smoothing_length // 2:
                # Calculate average metrics over the history buffer
                avg_metrics = np.mean(posture_history, axis=0)
                avg_shoulder_diff, avg_head_dist, avg_center_x = avg_metrics

                # Calculate scores for different aspects of posture (scaled 0 to 1)
                shoulder_level_score = 1.0 - np.clip(
                    avg_shoulder_diff / threshold_params['shoulder_diff'], 0, 1
                )
                head_slump_score = np.clip(
                    (avg_head_dist - (-threshold_params['head_distance'] * 1.5)) / threshold_params['head_distance'], 0, 1
                )
                center_score = 1.0 - np.clip(abs(avg_center_x - 0.5) / 0.3, 0, 1)

                # Combine scores using weights to get an overall posture score
                combined_posture_score = np.dot(
                    weight_params['posture'], # POSTURE_WEIGHTS
                    [shoulder_level_score, head_slump_score, center_score]
                )
                # Scale combined score to 1-5 for feedback and 0-100 for confidence
                posture_score = int(np.clip(combined_posture_score * 5, 1, 5))
                posture_confidence = int(np.clip(combined_posture_score * 100, 0, 100))

                # Generate posture feedback string
                if posture_score >= 4:
                    posture_feedback = "Good posture"
                elif posture_score >= 3:
                    posture_feedback = "Fair posture"
                else:
                    posture_feedback = "Posture needs improvement"

                # Add specific tips if certain scores are low
                if shoulder_level_score < 0.6:
                    posture_feedback += " (Level shoulders)"
                if head_slump_score < 0.6:
                    posture_feedback += " (Sit upright)"
                if center_score < 0.6:
                    posture_feedback += " (Center yourself)"

                # Determine shoulder line color based on shoulder level score
                shoulder_color = (
                    (0, 255, 0) if shoulder_level_score > 0.7 # Green for good
                    else (0, 165, 255) if shoulder_level_score > 0.4 # Orange for fair
                    else (0, 0, 255) # Red for poor
                )
            else:
                 # Not enough history for smoothed analysis
                 posture_feedback = "Analyzing posture..."
                 # Keep default color/score/confidence

        else:
            # Shoulders or head not sufficiently visible
            posture_feedback = "Ensure shoulders and head are visible"
            posture_history.clear() # Clear history when pose is lost to start fresh

    except Exception as e:
        logging.error(f"Error processing pose landmarks: {e}")
        posture_feedback = "Pose landmarks not available"
        posture_history.clear() # Clear history on error

    # Return calculated metrics, drawing points, color, and the updated history
    return posture_score, posture_confidence, posture_feedback, shoulder_line_pts, shoulder_color, posture_history

def _calculate_overall_confidence(posture_confidence, head_pose_confidence, weight_params):
    """Calculates the overall confidence score based on individual confidences and weights."""
    # Weighted average of posture and head pose confidences
    return int(
        (posture_confidence * weight_params['posture']) +
        (head_pose_confidence * weight_params['head_pose'])
    )

def _update_fps(loop_start_time, fps_filter):
    """Calculates current FPS and updates a smoothed FPS value."""
    loop_time = time.time() - loop_start_time
    current_fps = 1.0 / max(loop_time, 0.001) # Avoid division by zero
    # Simple exponential moving average for smoothing FPS
    fps_filter = fps_filter * 0.9 + current_fps * 0.1
    return fps_filter, current_fps # Return both filtered and current (though only filtered is used)


def _draw_overlay_analysis(frame, metrics, p1, p2, shoulder_line_pts, shoulder_color):
    """Draws information overlay for the analysis phase on the frame."""
    height, width = frame.shape[:2]

    # Draw translucent background rectangle at the top for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1) # Black rectangle
    alpha = 0.6 # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) # Blend overlay with frame

    # Confidence display
    confidence_color = (
        (0, 255, 0) if metrics['overall_confidence'] >= 75 # Green for high confidence
        else (0, 255, 255) if metrics['overall_confidence'] >= 50 # Yellow for medium confidence
        else (0, 0, 255) # Red for low confidence
    )
    cv2.putText(frame, f"Confidence: {metrics['overall_confidence']}%",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, confidence_color, 2)

    # Posture display (Score, Confidence, Feedback)
    posture_color = (
        (0, 255, 0) if metrics['posture_score'] >= 4 # Green for good posture
        else (0, 255, 255) if metrics['posture_score'] >= 3 # Yellow for fair posture
        else (0, 0, 255) # Red for poor posture
    )
    cv2.putText(frame, f"Posture: {metrics['posture_score']}/5 ({metrics['posture_confidence']}%)",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
    cv2.putText(frame, metrics['posture_feedback'],
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) # Grey feedback text

    # Head pose display (Direction, Confidence)
    head_pose_color = (
        (0, 255, 0) if metrics['head_pose_confidence'] >= 70 # Green for high confidence (e.g., Forward)
        else (0, 255, 255) if metrics['head_pose_confidence'] >= 40 # Yellow for medium confidence (e.g., Up/Down)
        else (0, 0, 255) # Red for low confidence (e.g., Left/Right or no face)
    )
    # Position head pose text on the right side of the frame
    text_size, _ = cv2.getTextSize(f"Head: {metrics['head_pose']} ({metrics['head_pose_confidence']}%)", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    text_x = width - text_size[0] - 10 # Align right with 10px margin
    cv2.putText(frame, f"Head: {metrics['head_pose']} ({metrics['head_pose_confidence']}%)",
               (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, head_pose_color, 1)

    # FPS display (This will show processing FPS, not video's original FPS)
    cv2.putText(frame, f"Processing FPS: {metrics['fps']:.1f}",
               (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2) # Blue

    # Draw head direction line (from nose to estimated gaze point)
    if p1 and p2:
        cv2.line(frame, p1, p2, (255, 100, 0), 3) # Cyan-like line

    # Draw shoulder line
    if shoulder_line_pts:
        cv2.line(frame, shoulder_line_pts[0], shoulder_line_pts[1], shoulder_color, 2)

    return frame

# --- Main Function for analyze_interview_presence (Modified for Video File) ---

def analyze_interview_presence(video_path, csv_path, frame_skip=1):

    # Initialize video capture from file
    cap = cv2.VideoCapture(video_path)

    # Check if video file was opened successfully
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        print(f"Error: Could not open video file {video_path}")
        return # Exit function if video cannot be opened

    # Get video properties for output writer
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    

    # logging.info(f"Input video properties: {original_width}x{original_height} @ {original_fps:.2f} FPS, {frame_count_total} frames")

    # MediaPipe Initialization for Face Mesh and Pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
        smooth_landmarks=True
    ) as pose:

        # State Variables for the analysis loop
        posture_history = deque(maxlen=POSTURE_SMOOTHING)
        fps_filter = 0.0 # Smoothed processing FPS
        start_time = time.time() # Time when analysis started
        frame_count_processed = 0 # Counter for frames where analysis was performed
        frame_count_read = 0 # Counter for all frames read from the video

        # Group parameters for easier passing to helper functions
        head_pose_thresholds = {
            'y_left_right': HEAD_POSE_FORWARD_THRESHOLD_Y,
            'x_down': HEAD_POSE_FORWARD_THRESHOLD_X,
            'x_up': HEAD_POSE_FORWARD_THRESHOLD_X
        }
        posture_threshold_params = {
            'shoulder_diff': SHOULDER_DIFF_THRESHOLD,
            'head_distance': HEAD_DISTANCE_THRESHOLD,
        }
        confidence_weight_params = {
            'posture': POSTURE_CONFIDENCE_WEIGHT,
            'head_pose': HEAD_POSE_CONFIDENCE_WEIGHT
        }
        posture_weight_params_dict = {'posture': POSTURE_WEIGHTS}


        # Main Analysis Loop: Process frames from the video file
        while cap.isOpened():
            loop_start_time = time.time() # Time at the start of the current loop iteration

            # Read a frame from the video capture
            success, frame = cap.read()
            if not success:
                logging.info("End of video stream.")
                break # Exit loop when no more frames are successfully read

            frame_count_read += 1 # Increment total frames read

            # --- Frame Skipping Logic ---
            # Only process and analyze every 'frame_skip' frame
            if frame_count_read % frame_skip != 0:
                continue 


            try:
                # --- Frame Processing ---
                # Resize frame for consistent processing size
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                # No need to flip horizontally for video files unless desired for specific effect
                # frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                frame_dims = (width, height)
                frame_count_processed += 1 # Increment counter for processed frames

                # Convert frame to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # --- MediaPipe Processing ---
                face_results = face_mesh.process(image_rgb)
                pose_results = pose.process(image_rgb)

                image_rgb.flags.writeable = True
                # (Note: We draw on the original 'frame' BGR image later)

                # --- Analysis and Metric Calculation ---

                metrics = _initialize_metrics()

                # Head Pose Estimation
                p1 = p2 = None
                if face_results and face_results.multi_face_landmarks:
                    metrics['head_pose'], metrics['head_pose_confidence'], p1, p2 = _analyze_head_pose(
                        face_results.multi_face_landmarks[0],
                        frame_dims,
                        CAM_MATRIX,
                        DIST_MATRIX,
                        LANDMARK_INDICES_HEAD_POSE,
                        head_pose_thresholds
                    )

                # Posture Analysis
                shoulder_line_pts = None
                shoulder_color = (0, 0, 255)
                if pose_results and pose_results.pose_landmarks:
                     (metrics['posture_score'], metrics['posture_confidence'],
                      metrics['posture_feedback'], shoulder_line_pts, shoulder_color,
                      posture_history) = _analyze_posture(
                         pose_results.pose_landmarks,
                         frame_dims,
                         posture_history,
                         POSTURE_SMOOTHING,
                         posture_threshold_params,
                         posture_weight_params_dict
                     )
                else:
                    posture_history.clear()


                # Final Overall Confidence Score
                metrics['overall_confidence'] = _calculate_overall_confidence(
                    metrics['posture_confidence'],
                    metrics['head_pose_confidence'],
                    confidence_weight_params
                )

                # FPS Calculation (Processing FPS)
                fps_filter, _ = _update_fps(loop_start_time, fps_filter)
                metrics['fps'] = fps_filter

                # Log metrics to CSV
                log_metrics(csv_path, metrics)


            except Exception as e:
                logging.error(f"Frame processing error during analysis: {e}", exc_info=True)

            # Allow interruption by pressing 'q' or ESC
            # cv2.waitKey(1) is needed for cv2.imshow to work, but also allows key press detection
            # For pure video processing without display, waitKey(1) is still useful for interruption.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logging.info("User requested exit from analysis.")
                break
            elif key == ord('r'): # Reset posture history - less useful for video but kept for consistency
                posture_history.clear()
                logging.info("Posture history reset by user.")


        # --- Analysis Loop Ends ---

        # Calculate and log summary statistics after the loop finishes
        duration = time.time() - start_time
        avg_processing_fps = frame_count_processed / duration if duration > 0 else 0
        logging.info(f"Video analysis completed. Duration: {duration:.2f}s, Frames Read: {frame_count_read}, Frames Processed (Analyzed): {frame_count_processed}, Avg Processing FPS: {avg_processing_fps:.2f}")

    # MediaPipe instances are automatically closed when exiting the 'with' block
    logging.info("MediaPipe resources released for analysis.")

    # Release video capture and writer resources
    cap.release()
    logging.info("Video capture and writer resources released.")

    # Destroy any OpenCV windows that might have been created (e.g., if imshow was used)
    cv2.destroyAllWindows()
    logging.info("All OpenCV windows closed.")


# --- Main Orchestrator Function (Modified for Video File Input) ---

def analyze_head(video_path, csv_path):
 
    frame_skip_value = 1 

    if not os.path.exists(video_path):
        print(f"Error: Input file not found at {video_path}")
        logging.error(f"Input file not found at {video_path}")
        sys.exit(1)


    analyze_interview_presence(video_path, csv_path, frame_skip=frame_skip_value)
    logging.info("Video analysis process finished.")

    print("-" * 30)
    print(f"Susccess, detailed metrics logged to: {csv_path}")

