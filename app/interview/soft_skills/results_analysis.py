import pandas as pd
import numpy as np
from scipy.stats import zscore

def calculate_final_score(scores, metrics):
    """
    Calculates the final weighted score using the user-provided weights,
    excluding the 'gaze' component.
    """
    # Define user-provided weights for each component score
    # EXCLUDING 'gaze' component
    weights = {
        'posture': 0.25,
        'distance': 0.15,
        'eyes': 0.10,
        'gaze_steadiness': 0.10,
        'head_pose': 0.25,
        'system quality': 0.15
    }
    # Normalize weights to sum to 1 (should already sum to 1, but good practice)
    total_weight_sum = sum(weights.values())
    # Handle case where total weight sum is zero after removing components
    if total_weight_sum <= 0:
         print("Warning: Total weight sum is zero or negative after excluding components. Final score will be 0.")
         return 0
    weights = {k: v / total_weight_sum for k, v in weights.items()}

    # Adjust weights based on data quality/stability issues (optional, based on original code logic)
    # Reduce weight of metrics affected by unstable tracking or low FPS
    # Use .get() with default values for safe access
    unstable_pct = metrics.get('distance', {}).get('unstable_pct', 0)
    low_fps_pct = metrics.get('data_quality', {}).get('low_fps_pct', 0)

    unstable_weight_factor = 1 - (unstable_pct / 100)
    low_fps_weight_factor = 1 - (low_fps_pct / 100)

    # Apply weight adjustments to relevant components (excluding 'gaze')
    if 'distance' in weights: weights['distance'] *= unstable_weight_factor
    # Removed gaze weight adjustment: if 'gaze' in weights: weights['gaze'] *= low_fps_weight_factor # Gaze can be affected by low FPS
    if 'eyes' in weights: weights['eyes'] *= low_fps_weight_factor # Eyes can be affected by low FPS
    if 'gaze_steadiness' in weights: weights['gaze_steadiness'] *= low_fps_weight_factor # Steadiness can be affected by low FPS
    if 'head_pose' in weights: weights['head_pose'] *= low_fps_weight_factor # Head pose can be affected by low FPS
    # Note: Posture and System are less directly affected by frame-level issues in this model, adjustments can be debated.
    # Keeping original adjustments for consistency where applicable to new weights.

    # Re-normalize weights after adjustments
    total_weight_sum_adjusted = sum(weights.values())
    # Handle case where total adjusted weight sum is zero
    if total_weight_sum_adjusted <= 0:
         print("Warning: Total adjusted weight sum is zero or negative. Final score will be 0.")
         return 0
    weights = {k: v / total_weight_sum_adjusted for k, v in weights.items()}

    # Calculate the weighted sum of component scores, skipping 'gaze'
    # Use .get() with default 0 to handle potential missing score keys safely
    weighted_sum = sum(scores.get(key, 0) * weights.get(key, 0) for key in weights.keys()) # Iterate over weights keys to exclude 'gaze'
    # Ensure final score is within 0-100 range
    return min(100, max(0, weighted_sum))


# =============================================================================
# Analysis Function 
# =============================================================================

def analyze_interview_performance(csv_path, rolling_window=5):
    """
    Analyzes interview performance metrics from a CSV file, tailored to the
    specific column names provided by the user, with added optimizations
    and enhanced gaze-based potential cheating analysis.
    Returns calculated metrics and scores, without generating plots.

    Args:
        csv_path (str or io.StringIO): Path or file-like object for the CSV data.
        rolling_window (int): The window size for calculating rolling averages in seconds.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary containing the final score, component scores, metrics,
                    and gaze cheating analysis results, or None if an error occurs.
            - pd.DataFrame: The processed DataFrame with added rolling columns,
                            or None if an error occurs.
        Returns (None, None) if an error occurs during loading or processing.
    """
    # Load and preprocess data
    try:
        # Handle both file path and file-like object (from upload)
        # Corrected: Removed redundant pd.read_csv call
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None

    # Ensure Timestamp column exists and is in datetime format
    if 'Timestamp' not in df.columns:
        print("Error: 'Timestamp' column not found in the CSV.")
        return None, None
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Set Timestamp as index for time-based operations
        df.set_index('Timestamp', inplace=True)
        # Sort by index to ensure correct rolling calculations
        df.sort_index(inplace=True)
    except Exception as e:
        print(f"Error processing 'Timestamp' column: {e}")
        return None, None


    # --- Data Cleaning and Preprocessing ---
    # Handle potential infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Basic Outlier Handling (using Z-score)
    # Apply Z-score only to selected numerical columns
    numerical_cols_for_outliers = ['Distance_cm', 'Smoothed_EAR_Avg', 'Smoothed_Ratio_H_Avg', 'Smoothed_Ratio_V_Avg', 'FPS']
    for col in numerical_cols_for_outliers:
        # Check if column exists and is numeric before applying zscore
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Calculate Z-scores, omitting NaNs
                valid_data = df[col].dropna()
                if not valid_data.empty:
                    # Use a robust method if data distribution is non-normal
                    # For simplicity, sticking to Z-score as in original, but consider Median Absolute Deviation (MAD)
                    z_scores = np.abs(zscore(valid_data))
                    threshold = 3 # Common threshold, adjust as needed
                    # Get indices of outliers within the valid data
                    outlier_indices = valid_data.index[z_scores > threshold]
                    # Set outliers to NaN in the original DataFrame
                    df.loc[outlier_indices, col] = np.nan
            except Exception as e:
                 print(f"Warning: Could not calculate Z-scores for column {col}. Skipping outlier detection. Error: {e}")

    # Interpolate missing numerical data after outlier removal
    # Use 'time' method for time series data, then ffill/bfill for edges
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time').fillna(method='ffill').fillna(method='bfill')


    # --- Helper Functions for Metrics and Scores ---
    def calculate_metrics(full_df, rolling_window):
        """Calculates various performance metrics."""
        # Separate stable and unstable segments based on Analysis_State
        # Ensure Analysis_State column exists before using it
        stable_mask = (full_df['Analysis_State'] == 'TRACKING') if 'Analysis_State' in full_df.columns else pd.Series(True, index=full_df.index)
        stable_df = full_df[stable_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        unstable_df = full_df[~stable_mask].copy()

        # Calculate scaling factor (using stable data)
        iod_mean = stable_df['Reference_IOD_px'].mean() if not stable_df.empty and 'Reference_IOD_px' in stable_df.columns else 0
        scale_factor = stable_df['Scale_Factor'].mean() if not stable_df.empty and 'Scale_Factor' in stable_df.columns else 0

        # Posture Metrics
        posture_mean = 0
        posture_std = 0
        posture_confidence_mean = 0
        if 'Posture_Score' in stable_df.columns:
             # Ensure column is numeric before calculating rolling stats
             stable_df['Posture_Score_Numeric'] = pd.to_numeric(stable_df['Posture_Score'], errors='coerce')
             # Calculate rolling mean using a time-based window (e.g., '5s')
             # Ensure window is not larger than data duration if data is short
             window_size = f'{rolling_window}s' if len(stable_df) > 1 else '1s' # Fallback for short data
             stable_df['Posture_Score_Rolling'] = stable_df['Posture_Score_Numeric'].rolling(window=window_size, min_periods=1).mean()
             posture_mean = stable_df['Posture_Score_Rolling'].mean()
             posture_std = stable_df['Posture_Score_Rolling'].std()
             if 'Posture_Confidence' in stable_df.columns:
                 posture_confidence_mean = stable_df['Posture_Confidence'].mean()

        # Gaze Metrics
        # Calculate value counts for gaze direction
        gaze_counts = full_df['Gaze_Direction'].value_counts(normalize=True) * 100 if 'Gaze_Direction' in full_df.columns else pd.Series()
        center_gaze_pct = gaze_counts.get('CENTER', 0)
        down_gaze_pct = gaze_counts.get('DOWN', 0)
        left_gaze_pct = gaze_counts.get('LEFT', 0)
        right_gaze_pct = gaze_counts.get('RIGHT', 0)
        up_gaze_pct = gaze_counts.get('UP', 0)
        away_gaze_pct = left_gaze_pct + right_gaze_pct + up_gaze_pct + down_gaze_pct # Sum of all non-CENTER gazes

        # Calculate gaze shifts (simple count of changes in direction)
        gaze_shifts = 0
        if 'Gaze_Direction' in full_df.columns:
             # Compare current gaze with the previous one, count where they differ
             gaze_shifts = (full_df['Gaze_Direction'] != full_df['Gaze_Direction'].shift(1)).sum()

 
        # Better logic to detect continuous "away" segments
        away_segments = []
        if 'Gaze_Direction' in full_df.columns and not full_df.empty:
            # Identify where Gaze_Direction is 'AWAY' (i.e., LEFT, RIGHT, UP, DOWN)
            away_mask = full_df['Gaze_Direction'].isin(['LEFT', 'RIGHT', 'UP', 'DOWN'])
            
            # Create group IDs for consecutive 'away' segments
            group_id = (~away_mask).cumsum()
            away_groups = full_df[away_mask].groupby(group_id)

            # Collect each 'away' segment's start and end times
            for name, group in away_groups:
                start_time = group.index[0]
                end_time = group.index[-1]
                duration = (end_time - start_time).total_seconds()
                away_segments.append(duration)

            # Calculate average duration of 'away' glances
            if away_segments:
                away_glance_duration_avg = np.mean(away_segments)
                away_glance_duration_std = np.std(away_segments)
            else:
                away_glance_duration_avg = 0
                away_glance_duration_std = 0
        else:
            away_glance_duration_avg = 0
            away_glance_duration_std = 0
                

        # Centering Status
        # Calculate value counts for centering status
        centering_counts = full_df['Centering_Status'].value_counts(normalize=True) * 100 if 'Centering_Status' in full_df.columns else pd.Series()
        centered_status_pct = centering_counts.get('Centered', 0)

        # Distance Metrics
        stable_distance = 0
        distance_std = 0
        total_frames = len(full_df)
        unstable_count = len(unstable_df)
        unstable_pct = (unstable_count / total_frames) * 100 if total_frames > 0 else 0

        if 'Distance_cm' in full_df.columns:
            # Ensure column is numeric before calculating rolling stats
            full_df['Distance_cm_Numeric'] = pd.to_numeric(full_df['Distance_cm'], errors='coerce')
            # Calculate rolling mean for distance
             # Ensure window is not larger than data duration if data is short
            window_size = f'{rolling_window}s' if len(full_df) > 1 else '1s' # Fallback for short data
            full_df['Distance_cm_Rolling'] = full_df['Distance_cm_Numeric'].rolling(window=window_size, min_periods=1).mean()
            # Calculate mean and std for stable segments
            stable_distance = full_df.loc[stable_mask, 'Distance_cm_Rolling'].mean()
            distance_std = full_df['Distance_cm_Rolling'].std()


        # Eye Metrics (EAR - Eye Aspect Ratio)
        mean_ear = 0
        ear_std = 0
        if 'Smoothed_EAR_Avg' in stable_df.columns:
            # Ensure column is numeric before calculating rolling stats
            stable_df['Smoothed_EAR_Avg_Numeric'] = pd.to_numeric(stable_df['Smoothed_EAR_Avg'], errors='coerce')
            # Calculate rolling mean for EAR
             # Ensure window is not larger than data duration if data is short
            window_size = f'{rolling_window}s' if len(stable_df) > 1 else '1s' # Fallback for short data
            stable_df['Smoothed_EAR_Avg_Rolling'] = stable_df['Smoothed_EAR_Avg_Numeric'].rolling(window=window_size, min_periods=1).mean()
            mean_ear = stable_df['Smoothed_EAR_Avg_Rolling'].mean()
            ear_std = stable_df['Smoothed_EAR_Avg_Rolling'].std()

        # Gaze Steadiness Metrics (Std Dev of Horizontal and Vertical Ratios)
        std_h_ratio_rolling = 0
        if 'Smoothed_Ratio_H_Avg' in full_df.columns:
            full_df['Smoothed_Ratio_H_Avg_Numeric'] = pd.to_numeric(full_df['Smoothed_Ratio_H_Avg'], errors='coerce')
            # Calculate rolling standard deviation for horizontal ratio
             # Ensure window is not larger than data duration if data is short
            window_size = f'{rolling_window}s' if len(full_df) > 1 else '1s' # Fallback for short data
            std_h_ratio_rolling = full_df['Smoothed_Ratio_H_Avg_Numeric'].rolling(window=window_size, min_periods=1).std().mean()

        std_v_ratio_rolling = 0
        if 'Smoothed_Ratio_V_Avg' in full_df.columns:
            full_df['Smoothed_Ratio_V_Avg_Numeric'] = pd.to_numeric(full_df['Smoothed_Ratio_V_Avg'], errors='coerce')
            # Calculate rolling standard deviation for vertical ratio
             # Ensure window is not larger than data duration if data is short
            window_size = f'{rolling_window}s' if len(full_df) > 1 else '1s' # Fallback for short data
            std_v_ratio_rolling = full_df['Smoothed_Ratio_V_Avg_Numeric'].rolling(window=window_size, min_periods=1).std().mean()

        # Combine horizontal and vertical steadiness
        combined_steadiness_std = (std_h_ratio_rolling + std_v_ratio_rolling) / 2 if not (np.isnan(std_h_ratio_rolling) or np.isnan(std_v_ratio_rolling)) else 0

        # Head Pose Metrics
        # Calculate value counts for head pose
        head_pose_counts = full_df['Head_Pose'].value_counts(normalize=True) * 100 if 'Head_Pose' in full_df.columns else pd.Series()
        head_forward_pct = head_pose_counts.get('Forward', 0)
        head_pose_confidence_mean = full_df['Head_Pose_Confidence'].mean() if 'Head_Pose_Confidence' in full_df.columns else 0

        # System Confidence
        # Calculate weighted average confidence based on stable/unstable segments
        stable_conf = stable_df['Overall_Confidence'].mean() if not stable_df.empty and 'Overall_Confidence' in stable_df.columns else 0
        unstable_conf = unstable_df['Overall_Confidence'].mean() if not unstable_df.empty and 'Overall_Confidence' in unstable_df.columns else 0
        weighted_conf = (stable_conf * len(stable_df) + unstable_conf * len(unstable_df)) / total_frames if total_frames > 0 else 0

        # Data Quality Metric (FPS - Frames Per Second)
        mean_fps = 0
        low_fps_threshold = 15 # Define a threshold for low FPS
        low_fps_pct = 0
        if 'FPS' in full_df.columns:
             full_df['FPS_Numeric'] = pd.to_numeric(full_df['FPS'], errors='coerce')
             mean_fps = full_df['FPS_Numeric'].mean()
             low_fps_pct = (full_df['FPS_Numeric'] < low_fps_threshold).sum() / total_frames * 100 if total_frames > 0 else 0

        # Assemble metrics into a dictionary
        metrics_dict = {
            'posture': {'mean': posture_mean, 'std': posture_std, 'confidence_mean': posture_confidence_mean},
            'gaze': {'center_gaze': center_gaze_pct, 'down_gaze': down_gaze_pct, 'left_gaze': left_gaze_pct,
                     'right_gaze': right_gaze_pct, 'up_gaze': up_gaze_pct, 'away_gaze': away_gaze_pct,
                     'gaze_shifts': gaze_shifts, 'away_glance_duration_avg': away_glance_duration_avg,
                     'centered_status': centered_status_pct, 'counts': gaze_counts},
            'distance': {'stable': stable_distance, 'std': distance_std, 'unstable_pct': unstable_pct, 'iod': iod_mean, 'scale': scale_factor},
            'eyes': {'mean_ear': mean_ear, 'std_ear': ear_std},
            'gaze_steadiness': {'std_h_ratio_rolling': std_h_ratio_rolling, 'std_v_ratio_rolling': std_v_ratio_rolling, 'combined_steadiness_std': combined_steadiness_std},
            'head_pose': {'forward_pct': head_forward_pct, 'confidence_mean': head_pose_confidence_mean, 'counts': head_pose_counts},
            'system_conf': weighted_conf,
            'data_quality': {'mean_fps': mean_fps, 'low_fps_pct': low_fps_pct},
            'segment_stats': {'stable_count': len(stable_df), 'unstable_count': unstable_count, 'total_frames': total_frames}
        }

        # Recursively fill NaNs with 0 and convert pandas Series and numpy types to standard types
        def fill_nan_recursive(item):
            if isinstance(item, dict):
                # Process dictionary items recursively
                return {k: fill_nan_recursive(v) for k, v in item.items()}
            elif isinstance(item, (float, np.floating)) and np.isnan(item):
                # Replace NaN floats/numpy floats with 0.0 (standard float)
                return 0.0
            elif isinstance(item, np.integer):
                 # Convert numpy integers to standard Python integers
                 return int(item)
            elif isinstance(item, np.floating):
                 # Convert numpy floats to standard Python floats
                 return float(item)
            elif isinstance(item, pd.Series):
                 # Convert pandas Series to dictionary for JSON compatibility
                 # Ensure NaN values within the series are also handled
                 return item.fillna(0).to_dict() # Fill NaNs before converting to dict
            # Return item as is if it's not a dict, NaN number, numpy type, or Series
            return item

        # Apply the recursive function to the metrics dictionary
        return fill_nan_recursive(metrics_dict)


    def calculate_scores(metrics):
        """Calculates component scores based on the metrics."""
        # Posture Score Calculation
        # Use .get() with default 0 to handle potential missing keys safely
        posture_mean = metrics.get('posture', {}).get('mean', 0)
        posture_std = metrics.get('posture', {}).get('std', 0)
        posture_confidence_mean = metrics.get('posture', {}).get('confidence_mean', 0)

        # Normalize confidence to a 0-1 scale
        posture_confidence = posture_confidence_mean / 100 if not np.isnan(posture_confidence_mean) else 0
        # Base score based on mean posture (assuming 4 is ideal)
        posture_score = (posture_mean / 4) * 100 if posture_mean > 0 else 0
        # Apply penalty based on standard deviation, scaled by confidence (lower confidence, less penalty)
        confidence_scale_factor = (1 - posture_confidence * 0.5) # Adjust penalty scaling
        posture_penalty = posture_std * 15 * confidence_scale_factor
        posture_score -= posture_penalty

        # Gaze Score Calculation
        # Note: This score is still calculated but will be excluded from the final weighted score.
        # It's kept here so it can still be displayed in the dashboard if needed.
        center_gaze_pct = metrics.get('gaze', {}).get('center_gaze', 0)
        down_gaze_pct = metrics.get('gaze', {}).get('down_gaze', 0)
        centered_status_pct = metrics.get('gaze', {}).get('centered_status', 0)

        gaze_score = center_gaze_pct # Start with percentage of time looking center
        # Penalize for looking down
        down_penalty = min(30, down_gaze_pct * 0.7) # Cap penalty
        gaze_score -= down_penalty
        # Penalize for not being centered
        not_centered_pct = 100 - centered_status_pct
        centering_penalty = min(20, not_centered_pct * 0.5) # Cap penalty
        gaze_score -= centering_penalty

        # Distance Score Calculation
        unstable_pct = metrics.get('distance', {}).get('unstable_pct', 0)
        distance_std = metrics.get('distance', {}).get('std', 0)

        # Score is higher for more stable tracking
        distance_score = 100 - unstable_pct
        # Penalize for high standard deviation in distance
        distance_score -= distance_std * 0.5

        # Eye Score Calculation (based on EAR)
        iod_scale = metrics.get('distance', {}).get('scale', 1.0)
        mean_ear = metrics.get('eyes', {}).get('mean_ear', 0)
        ear_std = metrics.get('eyes', {}).get('std_ear', 0)

        # Normalize EAR by IOD scale if available and valid
        normalized_ear = mean_ear / iod_scale if iod_scale > 0 and not np.isnan(iod_scale) else mean_ear
        ideal_ear = 0.3 # Approximate ideal EAR value
        # Score based on deviation from ideal EAR
        eye_score = max(0, 100 - abs(normalized_ear - ideal_ear) * 200)
        # Penalize for high standard deviation in EAR
        eye_score -= ear_std * 20

        # Gaze Steadiness Score Calculation
        combined_steadiness_std = metrics.get('gaze_steadiness', {}).get('combined_steadiness_std', 0)
        # Score is higher for lower combined standard deviation
        gaze_steadiness_score = max(0, 100 - combined_steadiness_std * 100)

        # Head Pose Score Calculation
        head_forward_pct = metrics.get('head_pose', {}).get('forward_pct', 0)
        head_pose_confidence_mean = metrics.get('head_pose', {}).get('confidence_mean', 0)

        # Normalize confidence to a 0-1 scale
        head_pose_confidence = head_pose_confidence_mean / 100 if not np.isnan(head_pose_confidence_mean) else 0
        # Score is percentage of time looking forward, scaled by confidence
        head_pose_score = head_forward_pct
        if not np.isnan(head_pose_confidence):
            head_pose_score *= head_pose_confidence

        # System Score Calculation (based on confidence and data quality)
        weighted_conf = metrics.get('system_conf', 0)
        low_fps_pct = metrics.get('data_quality', {}).get('low_fps_pct', 0)

        system_score = weighted_conf # Start with weighted overall confidence
        # Penalize for low FPS percentage
        system_score -= low_fps_pct * 0.5

        # Return component scores, ensuring they are within 0-100 range
        return {
            'posture': max(0, min(100, posture_score)),
            'gaze': max(0, min(100, gaze_score)), # Gaze score is still calculated here
            'distance': max(0, min(100, distance_score)),
            'eyes': max(0, min(100, eye_score)),
            'gaze_steadiness': max(0, min(100, gaze_steadiness_score)),
            'head_pose': max(0, min(100, head_pose_score)),
            'system quality': max(0, min(100, system_score))
        }

    def analyze_gaze_cheating(metrics, df):
        """
        Enhanced analysis of gaze patterns for potential cheating indicators in online interviews.
        Considers percentage of time looking in specific directions, frequency of gaze shifts,
        glance durations, and directional patterns.
        Assigns higher suspicion to downward glances (reading) compared to upward (recalling).

        Note: This analysis uses fixed thresholds and weights. For a production system,
        these should ideally be calibrated against baseline data from typical interviews
        and potentially adjusted for cultural norms. This requires external data and
        more complex modeling not included in this script.
        """
        # Extract relevant gaze metrics
        # Use .get() with a default value (0 or empty Series/list) to safely access nested metrics
        gaze_metrics = metrics.get('gaze', {})
        center_gaze_pct = gaze_metrics.get('center_gaze', 0)
        down_gaze_pct = gaze_metrics.get('down_gaze', 0)
        left_gaze_pct = gaze_metrics.get('left_gaze', 0)
        right_gaze_pct = gaze_metrics.get('right_gaze', 0)
        up_gaze_pct = gaze_metrics.get('up_gaze', 0)
        away_gaze_pct = gaze_metrics.get('away_gaze', 0)  # Total away percentage

        segment_stats = metrics.get('segment_stats', {})
        total_frames = segment_stats.get('total_frames', 0)

        data_quality_metrics = metrics.get('data_quality', {})
        mean_fps = data_quality_metrics.get('mean_fps', 0)

        gaze_shifts = gaze_metrics.get('gaze_shifts', 0)


        # Calculate gaze shifts per minute
        total_time_seconds = total_frames / mean_fps if mean_fps > 0 else 0
        gaze_shifts_per_minute = (gaze_shifts / total_time_seconds) * 60 if total_time_seconds > 0 else 0

        # --- Enhanced Glance Duration Analysis ---
        glance_durations = []
        avg_glance_duration = 0
        std_glance_duration = 0
        short_glances_pct = 0
        long_glances_pct = 0

        if 'Gaze_Direction' in df.columns and not df.empty:
            away_mask_duration = df['Gaze_Direction'].isin(['LEFT', 'RIGHT', 'UP', 'DOWN'])
            # Find the start and end points of consecutive 'away' segments
            # Handle first/last frame edge cases more robustly
            is_away = away_mask_duration.astype(int)
            away_diff = is_away.diff().fillna(is_away.iloc[0]) # Compare with previous, handle first element

            away_starts = df.index[away_diff == 1] # Transition from non-away to away
            away_ends = df.index[away_diff == -1] # Transition from away to non-away

            # Adjust for segments starting at the very beginning or ending at the very end
            if is_away.iloc[0] == 1:
                 away_starts = df.index[[0]].append(away_starts)
            if is_away.iloc[-1] == 1:
                 away_ends = away_ends.append(df.index[[-1]])

            # Calculate duration for each detected glance
            # Ensure starts and ends match up - sometimes diff can create mismatches
            # A more robust approach might involve iterating through segments directly
            # For simplicity and to match original logic intent, pair up starts and ends
            min_len = min(len(away_starts), len(away_ends))
            for start, end in zip(away_starts[:min_len], away_ends[:min_len]):
                 duration = (end - start).total_seconds()
                 if duration > 0:
                     glance_durations.append(duration)

            # Basic stats on glance durations
            if glance_durations:
                avg_glance_duration = np.mean(glance_durations)
                std_glance_duration = np.std(glance_durations)
                # Percentage of glances that are very short (< 0.3 seconds)
                short_glances_pct = (np.sum(np.array(glance_durations) < 0.3) / len(glance_durations)) * 100
                # Percentage of glances that are longer (> 2 seconds)
                long_glances_pct = (np.sum(np.array(glance_durations) > 2.0) / len(glance_durations)) * 100


        # --- Enhanced Cheating Scoring System ---
        # Define thresholds for interpolation
        threshold_away_pct_low = 10  # Normal range
        threshold_away_pct_high = 30  # Suspicious range

        threshold_shifts_low = 10  # Normal range
        threshold_shifts_high = 40  # Suspicious range

        threshold_suspicious_lr_low = 5
        threshold_suspicious_lr_high = 20

        threshold_down_low = 5
        threshold_down_high = 15

        threshold_up_low = 5
        threshold_up_high = 15

        # 1. Time Looking Away (Total)
        away_gaze_score = np.interp(away_gaze_pct, [threshold_away_pct_low, threshold_away_pct_high], [0, 100])

        # 2. Gaze Shifts per Minute
        shifts_score = np.interp(gaze_shifts_per_minute, [threshold_shifts_low, threshold_shifts_high], [0, 100])

        # 3. Direction-Specific Analysis (higher penalty for left/right)
        # This score focuses on left/right glances, often associated with looking at another screen.
        suspicious_lr_pct = left_gaze_pct + right_gaze_pct
        direction_lr_score = np.interp(suspicious_lr_pct, [threshold_suspicious_lr_low, threshold_suspicious_lr_high], [0, 100])

        # 4. Downward Glances (reading notes/phone)
        # This score specifically penalizes looking down, often associated with reading.
        down_score = np.interp(down_gaze_pct, [threshold_down_low, threshold_down_high], [0, 100])

        # 5. Upward Glances (recalling information - potentially less suspicious than downward)
        # This score penalizes looking up, potentially less severely than downward or left/right.
        up_score = np.interp(up_gaze_pct, [threshold_up_low, threshold_up_high], [0, 100])


        # 6. Glance Duration Patterns
        # Penalize both very short frequent glances (monitoring) and very long glances (reading)
        # Ensure scores are not negative if percentages are 0
        duration_pattern_score = (short_glances_pct * 0.4 + long_glances_pct * 0.6)
        duration_pattern_score = max(0, duration_pattern_score) # Ensure non-negative


        # Combine scores with weighted importance
        # Adjusted weights to prioritize downward glances as requested
        # Ensure scores used in calculation are not NaN, replace with 0 if necessary
        potential_cheating_indicator = (
            np.nan_to_num(away_gaze_score) * 0.10 +  # Overall away time (reduced weight)
            np.nan_to_num(shifts_score) * 0.15 +  # Frequent shifts (reduced weight)
            np.nan_to_num(direction_lr_score) * 0.25 +  # Left/right glances (another screen) (reduced weight)
            np.nan_to_num(down_score) * 0.35 +  # Downward glances (notes/phone) (increased weight - highest)
            np.nan_to_num(up_score) * 0.10 + # Upward glances (added with lower weight than down)
            np.nan_to_num(duration_pattern_score) * 0.05 # Glance duration patterns (reduced weight)
        )

        # Ensure indicator is within 0-100 range
        potential_cheating_indicator = max(0, min(100, potential_cheating_indicator))

        # Qualitative assessment with more detailed thresholds
        if potential_cheating_indicator < 20:
            assessment = "Normal gaze patterns. Low indication of cheating."
        elif potential_cheating_indicator < 40:
            assessment = "Mildly unusual gaze patterns. Possible signs of distraction."
        elif potential_cheating_indicator < 60:
            assessment = "Moderately unusual gaze patterns. Potential cheating indicators present, especially downward glances."
        elif potential_cheating_indicator < 80:
            assessment = "Highly unusual gaze patterns. Strong potential cheating indicators, particularly frequent or prolonged downward gazes."
        else:
            assessment = "Extremely unusual gaze patterns. Very high likelihood of cheating based on gaze analysis."

        return {
            'potential_cheating_indicator': potential_cheating_indicator,
            'away_gaze_percentage': away_gaze_pct,
            'center_gaze_percentage': center_gaze_pct,
            'left_gaze_percentage': left_gaze_pct,
            'right_gaze_percentage': right_gaze_pct,
            'up_gaze_percentage': up_gaze_pct,
            'down_gaze_percentage': down_gaze_pct,
            'gaze_shifts_per_minute': gaze_shifts_per_minute,
            'average_away_glance_duration_seconds': avg_glance_duration,
            'std_away_glance_duration_seconds': std_glance_duration,
            'short_glances_percentage': short_glances_pct,
            'long_glances_percentage': long_glances_pct,
            'assessment': assessment,
            'score_components': {  # Added for transparency in scoring
                'away_gaze_score': away_gaze_score,
                'shifts_score': shifts_score,
                'direction_lr_score': direction_lr_score, # Renamed for clarity
                'down_score': down_score,
                'up_score': up_score, # Added up score
                'duration_pattern_score': duration_pattern_score
            }
        }


    # --- Analysis Pipeline ---
    # Calculate metrics from the processed dataframe
    metrics = calculate_metrics(df, rolling_window)

    # Handle case where metrics calculation failed
    if metrics is None:
         print("Error: Metric calculation failed.")
         return None, None

    # Calculate component scores based on metrics
    scores = calculate_scores(metrics)

    # Handle case where scores calculation failed (unlikely if metrics succeeded, but for safety)
    if scores is None:
         print("Error: Score calculation failed.")
         return None, None

    # Calculate the final overall score using the specified weights
    # This function is now defined above analyze_interview_performance
    final_score = calculate_final_score(scores, metrics)

    # Perform enhanced gaze-based potential cheating analysis
    gaze_cheating_analysis = analyze_gaze_cheating(metrics, df)


    # --- Prepare Results Dictionary ---
    # This dictionary will be stored and used by callbacks
    results_dict = {
        'final_score': final_score,
        'component_scores': scores,
        'metrics': metrics, # Metrics dictionary now contains dictionaries instead of Series
        'gaze_cheating_analysis': gaze_cheating_analysis # Include enhanced cheating analysis results
    }

    # Print summary to console (optional, useful for debugging)
    print("\n=== Interview Performance Analysis Summary ===")
    print(f"Final Performance Score: {final_score:.1f}/100")
    print("\nComponent Scores:")
    for name, score in scores.items(): print(f"- {name.replace('_', ' ').capitalize()}: {score:.1f}/100")
    print("\nKey Metrics Summary:")
    # Safely access nested dictionary values using .get()
    posture_metrics = metrics.get('posture', {})
    gaze_metrics = metrics.get('gaze', {})
    distance_metrics = metrics.get('distance', {})
    eyes_metrics = metrics.get('eyes', {})
    head_pose_metrics = metrics.get('head_pose', {})
    data_quality_metrics = metrics.get('data_quality', {})
    segment_stats_metrics = metrics.get('segment_stats', {})
    cheating_analysis_metrics = results_dict.get('gaze_cheating_analysis', {}) # Use results_dict for cheating analysis

    print(f"- Posture Mean: {posture_metrics.get('mean', 0):.2f}, Gaze Center: {gaze_metrics.get('center_gaze', 0):.1f}%")
    print(f"- Distance Stable Mean: {distance_metrics.get('stable', 0):.1f}cm, Unstable: {distance_metrics.get('unstable_pct', 0):.1f}%")
    print(f"- Mean EAR: {eyes_metrics.get('mean_ear', 0):.3f}, Head Forward: {head_pose_metrics.get('forward_pct', 0):.1f}%")
    print(f"- Mean FPS: {data_quality_metrics.get('mean_fps', 0):.1f}, Low FPS: {data_quality_metrics.get('low_fps_pct', 0):.1f}%")
    print("\nEnhanced Potential Gaze Cheating Indicators:")
    print(f"- Indicator Score: {cheating_analysis_metrics.get('potential_cheating_indicator', 0):.1f}/100")
    print(f"- Assessment: {cheating_analysis_metrics.get('assessment', 'N/A')}")
    print(f"- Time Looking Away: {cheating_analysis_metrics.get('away_gaze_percentage', 0):.1f}%")
    print(f"- Time Looking Left: {cheating_analysis_metrics.get('left_gaze_percentage', 0):.1f}%")
    print(f"- Time Looking Right: {cheating_analysis_metrics.get('right_gaze_percentage', 0):.1f}%")
    print(f"- Time Looking Up: {cheating_analysis_metrics.get('up_gaze_percentage', 0):.1f}%")
    print(f"- Time Looking Down: {cheating_analysis_metrics.get('down_gaze_percentage', 0):.1f}%")
    print(f"- Gaze Shifts per Minute: {cheating_analysis_metrics.get('gaze_shifts_per_minute', 0):.1f}")
    # Corrected typo here
    print(f"- Avg Away Glance Duration: {cheating_analysis_metrics.get('average_away_glance_duration_seconds', 0):.2f} seconds")
    print(f"- Std Dev Away Glance Duration: {cheating_analysis_metrics.get('std_away_glance_duration_seconds', 0):.2f} seconds")
    print(f"- Percentage of Short Glances (< 0.3s): {cheating_analysis_metrics.get('short_glances_percentage', 0):.1f}%")
    print(f"- Percentage of Long Glances (> 2.0s): {cheating_analysis_metrics.get('long_glances_percentage', 0):.1f}%")
    print("\nAnalysis Complete.")


    # Return the results dictionary and the processed dataframe
    return results_dict, df
