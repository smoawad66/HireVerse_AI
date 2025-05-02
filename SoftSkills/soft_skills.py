import mediapipe as mp, cv2, numpy as np, time, logging, sys, os, csv
from datetime import datetime
from collections import deque
from SoftSkills.constants import *
from globals import BASE_PATH

def setup_logging():
    """Configure logging system with both file and console output"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Current timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/interview_analysis_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create CSV file for detailed metrics
    csv_path = f'logs/analysis_metrics_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'posture_score', 'posture_confidence', 'posture_feedback',
            'head_pose', 'head_pose_confidence', 'overall_confidence', 'fps'
        ])
    
    return csv_path

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
        'fps': 0
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
            # plus the relative Z coordinate from MediaPipe. This is a common
            # simplification but not a true 3D model. A more accurate method
            # would use a generic face model's known 3D points.
            success_pnp, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_DLS
                # Using DLS flag (or EPNP) can be more robust than default ITERATIVE
            )

            if success_pnp:
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rot_vec)
                # Decompose the rotation matrix to get Euler angles (pitch, yaw, roll)
                # MediaPipe often gives angles in a different order or coordinate system.
                # RQDecomp3x3 decomposition order can affect angle interpretation.
                # The original code's angles seem to align with (pitch, yaw, roll) roughly.
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                # Convert angles from radians to degrees (RQDecomp3x3 returns radians)
                # The original code multiplied by 360 - this is incorrect for radians to degrees (should be 180/pi).
                # Assuming the original thresholds were tuned for angles * 360, we will keep that for parity,
                # but note this is mathematically unusual. Correct conversion is angles * 180 / np.pi.
                # Let's stick to the original's scaling (multiply by 360) to match its thresholds.
                x_angle = angles[0] * 360  # Pitch (looking up/down)
                y_angle = angles[1] * 360  # Yaw (looking left/right)
                z_angle = angles[2] * 360  # Roll (tilting head side to side)

                # Determine head direction based on yaw (y_angle) and pitch (x_angle)
                # Original thresholds are used here.
                if y_angle < -thresholds['y_left_right'] * 2:
                    head_pose = "Looking Left"
                elif y_angle > thresholds['y_left_right'] * 2:
                    head_pose = "Looking Right"
                elif x_angle < -thresholds['x_down']:
                    head_pose = "Looking Down"
                # Original code used x_angle > HEAD_POSE_FORWARD_THRESHOLD_X * 6.5 for looking up
                elif x_angle > thresholds['x_up'] * 6.5: # Using x_up key with original 6.5 multiplier
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
                # The length of the line is scaled by the angle magnitude.
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                # Scaling factors 5 and -5 are arbitrary from the original code, influencing line length.
                p2_x = int(nose_2d[0] + y_angle * 5)
                p2_y = int(nose_2d[1] - x_angle * 5) # Negative because positive pitch is looking down in this angle convention
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
                # Shoulder level: 1.0 for level shoulders (diff close to 0), decreases as diff increases
                shoulder_level_score = 1.0 - np.clip(
                    avg_shoulder_diff / threshold_params['shoulder_diff'], 0, 1
                )
                # Head slump: 1.0 for upright head (head above shoulders), decreases as head slumps below shoulders
                # The term (-threshold_params['head_distance'] * 1.5) in the original code is unusual.
                # It seems intended to shift the range, assuming head_dist around this value is poor.
                # Let's interpret avg_head_dist > 0 as head above shoulders, < 0 as head below.
                # Score should decrease as head_dist decreases (slumps).
                # A simplified approach: Score is high if head_dist is positive and above a small margin.
                # The original formula np.clip((avg_head_dist - (-HEAD_DISTANCE_THRESHOLD * 1.5)) / HEAD_DISTANCE_THRESHOLD, 0, 1)
                # seems to treat `avg_head_dist` relative to a negative offset.
                # Let's keep the original formula's structure to match behavior.
                head_slump_score = np.clip(
                    (avg_head_dist - (-threshold_params['head_distance'] * 1.5)) / threshold_params['head_distance'], 0, 1
                )
                # Centering: 1.0 for being horizontally centered (center_x close to 0.5), decreases as face moves away from center.
                # The 0.3 divisor is an arbitrary scaling factor.
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
                if shoulder_level_score < 0.6: # Threshold 0.6 is arbitrary from original code
                    posture_feedback += " (Level shoulders)"
                if head_slump_score < 0.6: # Threshold 0.6 is arbitrary from original code
                    posture_feedback += " (Sit upright)"
                if center_score < 0.6: # Threshold 0.6 is arbitrary from original code
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

    # FPS display
    cv2.putText(frame, f"FPS: {metrics['fps']:.1f}",
               (width - 120, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2) # Blue

    # Draw head direction line (from nose to estimated gaze point)
    if p1 and p2:
        cv2.line(frame, p1, p2, (255, 100, 0), 3) # Cyan-like line

    # Draw shoulder line
    if shoulder_line_pts:
        cv2.line(frame, shoulder_line_pts[0], shoulder_line_pts[1], shoulder_color, 2)

    return frame

# --- Main Function for analyze_interview_presence ---

def analyze_interview_presence(cap, csv_path):
    """
    Analyzes interview presence from a video capture feed,
    providing feedback on posture and head pose.

    Args:
        cap: OpenCV VideoCapture object.
        csv_path: Path to the CSV file for logging metrics.
    """
    logging.info("Starting interview presence analysis")

    # MediaPipe Initialization for Face Mesh and Pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    # Use context managers for MediaPipe instances to ensure resources are released
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, # Only interested in the primary user
        refine_landmarks=True, # Get more detailed face landmarks
        min_detection_confidence=0.5, # Minimum confidence for detecting a face
        min_tracking_confidence=0.5 # Minimum confidence for tracking a face
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.6, # Minimum confidence for detecting pose
        min_tracking_confidence=0.6, # Minimum confidence for tracking pose
        model_complexity=1, # Complexity of the pose landmark model (0, 1, or 2)
        smooth_landmarks=True # Smooth pose landmarks across frames
    ) as pose:

        # State Variables for the analysis loop
        posture_history = deque(maxlen=POSTURE_SMOOTHING) # Buffer for smoothing posture metrics
        fps_filter = 0.0 # Smoothed FPS value
        start_time = time.time() # Time when analysis started
        frame_count = 0 # Counter for processed frames

        # Group parameters for easier passing to helper functions
        head_pose_thresholds = {
            'y_left_right': HEAD_POSE_FORWARD_THRESHOLD_Y,
            'x_down': HEAD_POSE_FORWARD_THRESHOLD_X,
            'x_up': HEAD_POSE_FORWARD_THRESHOLD_X # Using same threshold value for up as down from constants
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


        # Main Analysis Loop: Process frames from the video feed
        while cap.isOpened(): # Continue as long as the capture is open
            loop_start_time = time.time() # Time at the start of the current loop iteration

            # Read a frame from the video capture
            success, frame = cap.read()
            if not success:
                logging.warning("Failed to grab frame during analysis. Stream ended or error.")
                # If using a video file, this is the normal end. If using camera, it's an issue.
                time.sleep(0.1) # Wait a bit before trying again if it's a camera issue
                # Consider adding a counter to break after multiple failures
                break # Exit loop if frame cannot be read

            try:
                # --- Frame Processing ---
                # Resize frame for consistent processing size
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                # Flip horizontally for selfie-view consistency
                frame = cv2.flip(frame, 1)


                height, width = frame.shape[:2]
                frame_dims = (width, height) # Store dimensions as a tuple
                frame_count += 1 # Increment frame counter

                # Convert frame to RGB for MediaPipe (which uses RGB)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Set flag to non-writeable to improve MediaPipe performance
                image_rgb.flags.writeable = False

                # --- MediaPipe Processing ---
                # Process the frame to get face and pose landmarks
                face_results = face_mesh.process(image_rgb)
                pose_results = pose.process(image_rgb)

                # Set flag back to writeable to allow drawing on the frame
                image_rgb.flags.writeable = True
                # (Note: We draw on the original 'frame' BGR image later)

                # --- Analysis and Metric Calculation ---

                # Initialize metrics dictionary for this frame
                metrics = _initialize_metrics()

                # Head Pose Estimation
                p1 = p2 = None # Reset drawing points for head direction line
                if face_results and face_results.multi_face_landmarks:
                    # Analyze head pose using the first detected face
                    metrics['head_pose'], metrics['head_pose_confidence'], p1, p2 = _analyze_head_pose(
                        face_results.multi_face_landmarks[0],
                        frame_dims,
                        CAM_MATRIX,
                        DIST_MATRIX,
                        LANDMARK_INDICES_HEAD_POSE,
                        head_pose_thresholds
                    )

                # Posture Analysis
                shoulder_line_pts = None # Reset drawing points for shoulder line
                shoulder_color = (0, 0, 255) # Reset shoulder line color (default red)
                # Pass the posture_history deque by reference to be updated within the function
                if pose_results and pose_results.pose_landmarks:
                     (metrics['posture_score'], metrics['posture_confidence'],
                      metrics['posture_feedback'], shoulder_line_pts, shoulder_color,
                      posture_history) = _analyze_posture(
                         pose_results.pose_landmarks,
                         frame_dims,
                         posture_history, # Pass the deque
                         POSTURE_SMOOTHING,
                         posture_threshold_params,
                         posture_weight_params_dict # Pass posture weights
                     )
                else:
                    # If no pose is detected, clear posture history as the user is no longer visible
                    posture_history.clear()


                # Final Overall Confidence Score
                metrics['overall_confidence'] = _calculate_overall_confidence(
                    metrics['posture_confidence'],
                    metrics['head_pose_confidence'],
                    confidence_weight_params
                )

                # FPS Calculation (Using smoothed FPS for display)
                fps_filter, _ = _update_fps(loop_start_time, fps_filter)
                metrics['fps'] = fps_filter

                # Log metrics to CSV (Uses the placeholder log_metrics function)
                log_metrics(csv_path, metrics)

                # --- Drawing ---
                # Draw the analysis overlay (text, lines, indicators) on the frame
                frame = _draw_overlay_analysis(frame, metrics, p1, p2, shoulder_line_pts, shoulder_color)

                # --- Display ---
                # Show the frame with the overlay
                cv2.imshow("Interview Presence Analyzer", frame)

            except Exception as e:
                # Catch any errors during frame processing and log them
                logging.error(f"Frame processing error during analysis: {e}", exc_info=True) # exc_info logs traceback

            # --- Exit Conditions and User Input ---
            key = cv2.waitKey(10) & 0xFF # Wait 10ms for a key press
            if key == ord('q') or key == 27:  # 'q' or ESC key to exit analysis
                logging.info("User requested exit from analysis.")
                break # Exit the main analysis loop
            elif key == ord('r'): # 'r' key to reset posture history
                posture_history.clear()
                logging.info("Posture history reset by user.")

        # --- Analysis Loop Ends ---

        # Calculate and log summary statistics after the loop finishes
        duration = time.time() - start_time
        # Avoid division by zero if no frames were processed
        avg_fps = frame_count / duration if duration > 0 else 0
        logging.info(f"Analysis completed. Duration: {duration:.2f}s, Frames: {frame_count}, Avg FPS: {avg_fps:.2f}")

    # MediaPipe instances are automatically closed when exiting the 'with' block
    logging.info("MediaPipe resources released for analysis.")

    # Destroy the OpenCV window for analysis
    cv2.destroyWindow("Interview Presence Analyzer")
    logging.info("Analysis window closed.")

# --- Helper Functions for monitor_distance ---

def _process_frame_monitor(cap, face_mesh):
    """Reads a frame, preprocesses it, and runs MediaPipe face mesh."""
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to grab frame during monitoring.")
        time.sleep(0.1)
        return None, None, None # Indicate failure

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    frame_dims = (frame_width, frame_height)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True # Restore writeable flag

    return frame, results, frame_dims

def _calculate_distance_and_status(landmarks, frame_dims, ref_eye_distance_cm, min_distance_cm, max_distance_cm, distance_buffer, focal_length):
    """Calculates distance from face landmarks and determines status."""
    width, height = frame_dims
    distance = None
    status = "none"
    new_focal_length = focal_length # Start with the current focal length

    try:
        # Indices for left and right eyes for distance calculation
        # Using landmarks 234 (right eye outer) and 454 (left eye outer) as in original code
        left_eye_landmark = landmarks.landmark[234]
        right_eye_landmark = landmarks.landmark[454]

        # Calculate eye distance in pixels using Euclidean distance
        eye_px = np.linalg.norm(np.array([
            left_eye_landmark.x * width - right_eye_landmark.x * width,
            left_eye_landmark.y * height - right_eye_landmark.y * height
        ]))

        if eye_px > 0:
            # Calibration Step: Calculate focal length if not already known
            # Assumes the user is approximately at FOCAL_LENGTH_CALIBRATION_DISTANCE_CM (e.g., 60cm)
            # the first time a face is successfully detected with sufficient eye separation.
            if new_focal_length is None:
                 # Formula: Focal Length = (Pixel Distance * Real World Distance) / Real World Object Size
                 # Real World Object Size is REF_EYE_DISTANCE_CM
                 # Real World Distance is the assumed calibration distance (e.g., 60 cm)
                 new_focal_length = (eye_px * FOCAL_LENGTH_CALIBRATION_DISTANCE_CM) / ref_eye_distance_cm
                 logging.info(f"Focal length calibrated to {new_focal_length:.2f} based on {FOCAL_LENGTH_CALIBRATION_DISTANCE_CM}cm assumption.")


            # Calculate distance if focal length is known
            if new_focal_length is not None:
                # Formula: Distance = (Real World Object Size * Focal Length) / Pixel Distance
                raw_dist = (ref_eye_distance_cm * new_focal_length) / eye_px
                distance_buffer.append(raw_dist) # Add current distance to the buffer
                distance = np.mean(distance_buffer) # Calculate smoothed distance

                # Determine status based on the smoothed distance relative to thresholds
                if distance < min_distance_cm:
                    status = "close"
                elif distance > max_distance_cm:
                    status = "far"
                else:
                    status = "good" # Within the desired range

    except Exception as e:
        logging.debug(f"Distance calculation error: {e}") # Use debug level for less critical errors here
        status = "none" # Set status to none on error or if landmarks missing
        distance = None # Reset distance on error
        # Do not reset focal_length here if it was already calibrated; keep it for next attempts.
        # Resetting focal_length happens in monitor_distance's main loop if no face is detected.

    # Return calculated distance, status, and the potentially updated focal length
    return distance, status, new_focal_length, distance_buffer # Return updated buffer too

def _check_centering(landmarks, frame_dims):
    """Checks if the face is centered horizontally."""
    width, height = frame_dims
    center_aligned = False
    user_x = width // 2 # Default user x position for drawing if landmarks not available

    try:
        # Calculate the horizontal center of the face using eye landmarks
        left_eye_landmark = landmarks.landmark[234]
        right_eye_landmark = landmarks.landmark[454]
        user_x = (left_eye_landmark.x + right_eye_landmark.x) * width / 2
        frame_center_x = width // 2 # Horizontal center of the frame

        # Check if the user's horizontal position is within a tolerance range of the frame center
        center_aligned = abs(user_x - frame_center_x) < 50 # Tolerance of 50 pixels (arbitrary value from original code)

    except Exception as e:
        logging.debug(f"Centering check error: {e}") # Use debug level

    # Return centering status and the user's calculated horizontal position (for drawing a marker)
    return center_aligned, user_x

def _draw_distance_ui(frame, distance, status, center_aligned, user_x, frame_dims, colors, bar_settings, min_distance_cm, max_distance_cm, landmarks_results):
    """Draws the distance monitoring UI on the frame."""
    frame_width, frame_height = frame_dims

    # Draw translucent background for the text information at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, 95), (0, 0, 0), -1) # Black rectangle
    alpha = 0.6 # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) # Blend overlay with the original frame

    # Display Distance
    distance_text = f"Distance: {distance:.1f} cm" if distance is not None else "Distance: -- cm"
    cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)

    # Display Status (Good, Close, Far, None)
    cv2.putText(frame, f"Status: {status.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.get(status, colors['none']), 2)

    # Display Instructions
    cv2.putText(frame, "Press 'q' to Start Analysis (ESC to Cancel)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1) # Cyan text

    # Draw Distance Bar if distance is available
    if distance is not None:
        x, y, w, h = bar_settings
        # Calculate position of the indicator on the bar (0 for min, 1 for max)
        if max_distance_cm > min_distance_cm: # Avoid division by zero if thresholds are identical
            pos_ratio = (distance - min_distance_cm) / (max_distance_cm - min_distance_cm)
            pos = np.clip(pos_ratio, 0, 1) # Clamp position between 0 and 1
        else:
             pos = 0.5 # Default to middle if range is invalid

        # Draw the background bar
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), -1) # Grey background

        # Draw the position indicator (circle)
        indicator_x = x + int(pos * w)
        cv2.circle(frame, (indicator_x, y + h // 2), 10, colors.get(status, colors['none']), -1) # Circle colored by status

        # Draw markers for min and max distance on the bar for reference
        min_marker_x = x + int(0 * w) # Start of the bar corresponds to min distance
        max_marker_x = x + int(1 * w) # End of the bar corresponds to max distance
        cv2.line(frame, (min_marker_x, y + h), (min_marker_x, y + h + 10), colors['text'], 2)
        cv2.line(frame, (max_marker_x, y + h), (max_marker_x, y + h + 10), colors['text'], 2)
        # Add text labels for min and max distance
        cv2.putText(frame, str(min_distance_cm), (min_marker_x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)
        # Adjust text position slightly for the max distance label so it doesn't go off screen
        max_text_size, _ = cv2.getTextSize(str(max_distance_cm), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, str(max_distance_cm), (max_marker_x - max_text_size[0], y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)


    # Draw Center Alignment Guide
    line_color = colors['center'] if center_aligned else colors['not_center']
    cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), line_color, 1) # Vertical center line of the frame

    # Draw a marker for the user's horizontal position if face landmarks are available
    if landmarks_results and landmarks_results.multi_face_landmarks:
         try:
             # Recalculate user_x specifically for drawing based on available landmarks
             left_eye_landmark = landmarks_results.multi_face_landmarks[0].landmark[234]
             right_eye_landmark = landmarks_results.multi_face_landmarks[0].landmark[454]
             user_x_draw = (left_eye_landmark.x + right_eye_landmark.x) * frame_width / 2
             # Draw a small vertical line segment at the bottom indicating user's horizontal center
             cv2.line(frame, (int(user_x_draw), frame_height - 20), (int(user_x_draw), frame_height), colors['text'], 3) # White line
         except:
             pass # Ignore drawing user line if landmark access fails


    return frame


# --- Main Function for monitor_distance ---

def monitor_distance(cap):
    """
    Monitors the user's distance from the camera and centering as a setup phase.

    Args:
        cap: OpenCV VideoCapture object.

    Returns:
        True if monitoring was completed successfully (user pressed 'q'),
        False if cancelled (user pressed ESC) or an error occurred.
    """
    logging.info("Starting distance monitoring setup phase...")

    # Initialize MediaPipe Face Mesh instance for this function
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, # Only need to track one face
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # State variables for distance monitoring
    distance_buffer = deque(maxlen=5) # Buffer to smooth distance readings
    focal_length = None # Variable to store calculated focal length (calibrated once)

    try:
        # Main loop for distance monitoring
        while cap.isOpened(): # Continue as long as the capture is open
            # Process the frame using the helper function
            frame, results, frame_dims = _process_frame_monitor(cap, face_mesh)
            if frame is None: # If frame reading failed, continue loop
                continue

            # Get face landmarks if detected
            landmarks = results.multi_face_landmarks[0] if results and results.multi_face_landmarks else None

            # --- Distance Calculation and Status ---
            distance = None
            status = "none"
            # Only calculate distance if face landmarks are available
            if landmarks:
                 # Use the helper function to calculate distance and status
                 # Pass focal_length to potentially update it during calibration
                 # Get the updated distance_buffer back
                distance, status, focal_length, distance_buffer = _calculate_distance_and_status(
                    landmarks,
                    frame_dims,
                    REF_EYE_DISTANCE_CM,
                    MIN_DISTANCE_CM,
                    MAX_DISTANCE_CM,
                    distance_buffer,
                    focal_length # Pass the current focal length state
                )
            else:
                 # If no face is detected, clear the distance buffer
                 distance_buffer.clear()
                 # Also reset focal length so calibration can happen again when face reappears
                 focal_length = None
                 status = "none" # Explicitly set status when no face

            # --- Centering Check ---
            center_aligned = False
            user_x = frame_dims[0] // 2 # Default user x position for drawing if no face
            # Only check centering if face landmarks are available
            if landmarks:
                center_aligned, user_x = _check_centering(landmarks, frame_dims)

            # --- Drawing UI ---
            # Draw the distance monitoring UI on the frame using the helper function
            frame = _draw_distance_ui(
                frame,
                distance,
                status,
                center_aligned,
                user_x,
                frame_dims,
                COLORS, # Pass global COLORS dictionary
                BAR_SETTINGS, # Pass global BAR_SETTINGS tuple
                MIN_DISTANCE_CM, # Pass global min distance
                MAX_DISTANCE_CM, # Pass global max distance
                landmarks_results=results # Pass the original results object for drawing user line in _draw_distance_ui
            )

            # --- Display ---
            # Show the frame with the monitoring UI
            cv2.imshow('Distance Monitor', frame)

            # --- User Input (Exit Conditions) ---
            key = cv2.waitKey(10) & 0xFF # Wait 10ms for a key press
            if key == ord('q'): # Press 'q' to complete setup and proceed
                logging.info("Distance monitoring completed successfully by user.")
                return True # Return True to indicate success
            elif key == 27:  # Press ESC to cancel the setup
                logging.info("Distance monitoring cancelled by user.")
                return False # Return False to indicate cancellation

        # If the loop exits for reasons other than 'q' or ESC (e.g., camera closed)
        logging.warning("Distance monitoring loop exited unexpectedly.")
        return False

    except Exception as e:
        # Catch any unhandled exceptions during monitoring
        logging.error(f"Distance monitoring error: {e}", exc_info=True)
        return False # Return False on error

    finally:
        # --- Cleanup ---
        # Ensure the OpenCV window for monitoring is closed
        cv2.destroyWindow('Distance Monitor')
        logging.info("Distance monitoring window closed.")
        # Ensure the MediaPipe Face Mesh instance is closed and resources are released
        face_mesh.close()
        logging.info("MediaPipe Face Mesh resources released for monitoring.")






def test_interview(frame, posture_history, interview_id): 
    
    fps_filter = 0.0
    start_time = time.time()
    frame_count = 0

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


    loop_start_time = time.time()

    # success, frame = cap.read()

    try:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        frame_dims = (width, height)
        frame_count += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        face_results = face_mesh_model.process(image_rgb)
        pose_results = pose_model.process(image_rgb)

        image_rgb.flags.writeable = True

        metrics = _initialize_metrics()

    
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

        metrics['overall_confidence'] = _calculate_overall_confidence(
            metrics['posture_confidence'],
            metrics['head_pose_confidence'],
            confidence_weight_params
        )

        fps_filter, _ = _update_fps(loop_start_time, fps_filter)
        metrics['fps'] = fps_filter

        csv_path = f"{BASE_PATH}/SoftSkills/logs/analysis_metrics_{interview_id}.csv"
        
        log_metrics(csv_path, metrics)
        
        frame = _draw_overlay_analysis(frame, metrics, p1, p2, shoulder_line_pts, shoulder_color)

        # cv2.imshow("Interview Presence Analyzer", frame)

    except Exception as e:
        logging.error(f"Frame processing error during analysis: {e}", exc_info=True)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        logging.info("User requested exit from analysis.")
    elif key == ord('r'):
        posture_history.clear()
        logging.info("Posture history reset by user.")


    duration = time.time() - start_time

    avg_fps = frame_count / duration if duration > 0 else 0

    logging.info(f"Analysis completed. Duration: {duration:.2f}s, Frames: {frame_count}, Avg FPS: {avg_fps:.2f}")
    logging.info("MediaPipe resources released for analysis.")
    logging.info("Analysis window closed.")



# --- Main Orchestrator Function ---

def main():
    """
    Main function to run the interview presence analysis workflow.
    Starts with distance monitoring setup, then proceeds to analysis
    if the setup is successfully completed by the user.
    """
    # Setup logging at the beginning of the application
    csv_log_path = setup_logging() # Call the placeholder setup_logging

    # Initialize video capture (OpenCV VideoCapture)
    logging.info("Initializing video capture...")
    cap = cv2.VideoCapture(0) # Use 0 for the default camera. Change if needed.

    # Check if the camera was opened successfully
    if not cap.isOpened():
        logging.error("Error: Could not open video capture. Please ensure your camera is connected and accessible.")
        print("Error: Could not open video capture. Please ensure your camera is connected and accessible.")
        sys.exit(1) # Exit the program if camera access fails

    logging.info("Video capture initialized successfully.")

    # --- Phase 1: Distance Monitoring Setup ---
    logging.info("Starting distance monitoring phase...")
    # Call the monitor_distance function and store its return value
    setup_successful = monitor_distance(cap)

    # --- Decide Next Phase based on Setup Result ---
    if setup_successful:
        # --- Phase 2: Interview Presence Analysis ---
        logging.info("Distance monitoring completed successfully. Proceeding to analysis.")
        # Call the analyze_interview_presence function, passing the same cap object
        # and the path for logging.
        analyze_interview_presence(cap, csv_log_path)
        logging.info("Interview presence analysis completed.")
    else:
        # If setup was not successful (cancelled or error), skip analysis
        logging.info("Distance monitoring cancelled or failed. Analysis skipped.")

    # --- Cleanup ---
    # Release the video capture resource
    logging.info("Releasing video capture resource.")
    cap.release()
    # Destroy any remaining OpenCV windows (redundant after functions call destroyWindow, but safe)
    cv2.destroyAllWindows()
    logging.info("All OpenCV windows closed.")
    logging.info("Application finished.")



if __name__ == "__main__":
    main()