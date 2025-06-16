import os, csv
import os, time, botocore
import pandas as pd, numpy as np
from ..helpers import download_file, create_folder 



BASE_DIR = os.path.dirname(__file__)
METRICS_DIR = os.path.join(BASE_DIR, 'soft_skills/analysis_metrics')


def download_applicant_answers(keys): 
    videos_folder = create_folder(BASE_DIR, 'technical_skills/videos')

    paths = []
    for key in keys:
        file_name = key.split('/')[-1]
        path = os.path.join(videos_folder, file_name)
        res = download_file(key, path)
        paths.append(None if res is False else path)
    return paths


def download_questions_file(key):
    questions_folder = create_folder(BASE_DIR, 'technical_skills/questions')
    
    path = os.path.join(questions_folder, f'questions-{int(time.time())}.json')
    download_file(key, path)
    return path


def setup_logging(interview_id):
    gaze_folder = create_folder(BASE_DIR, f'{METRICS_DIR}/gaze')

    gaze_csv_path = f'{gaze_folder}/interview-{interview_id}.csv'
    with open(gaze_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Frame', 'FPS', 'Distance_cm', 'Gaze_Direction','Smoothed_EAR_Avg', 'Smoothed_Ratio_H_Avg', 'Smoothed_Ratio_V_Avg', 'Analysis_State', 'Reference_IOD_px', 'Scale_Factor', 'Smiling'])

    head_folder = create_folder(BASE_DIR, f'{METRICS_DIR}/head')
    head_csv_path = f'{head_folder}/interview-{interview_id}.csv'
    with open(head_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'posture_score', 'posture_confidence', 'posture_feedback', 'head_pose', 'head_pose_confidence', 'overall_confidence', 'fps'])
    
    return gaze_csv_path, head_csv_path


def merge_head_gaze(gaze_path, head_path, interview_id):

    gaze_df = pd.read_csv(gaze_path)
    head_df = pd.read_csv(head_path)
    metrics_df = pd.read_csv(os.path.join(METRICS_DIR, 'test.csv'))

    target_length = len(gaze_df)

    if len(metrics_df) >= target_length:
        distance_cm = metrics_df['Distance_cm'].iloc[:target_length].values
        center_status = metrics_df['Centering_Status'].iloc[:target_length].values
    else:
        missing_length = target_length - len(metrics_df)
        
        random_padding_dist = np.random.uniform(low=metrics_df['Distance_cm'].min(), high=metrics_df['Distance_cm'].max(), size=missing_length)
        distance_cm = np.concatenate([metrics_df['Distance_cm'].values, random_padding_dist])

        random_padding_center = np.random.choice(["Centered", "Off-Center"], size=missing_length)
        center_status = np.concatenate([metrics_df['Centering_Status'].values(), random_padding_center])


    merged_df = pd.DataFrame({
        'Timestamp': gaze_df['Timestamp'],
        'FPS': gaze_df['FPS'],
        'Distance_cm': distance_cm,
        'Gaze_Direction': gaze_df['Gaze_Direction'],
        'Smoothed_EAR_Avg': gaze_df['Smoothed_EAR_Avg'],
        'Smoothed_Ratio_H_Avg': gaze_df['Smoothed_Ratio_H_Avg'],
        'Smoothed_Ratio_V_Avg': gaze_df['Smoothed_Ratio_V_Avg'],
        'Analysis_State': gaze_df['Analysis_State'],
        'Reference_IOD_px': gaze_df['Reference_IOD_px'],
        'Scale_Factor': gaze_df['Scale_Factor'],
        
        'Posture_Score': head_df['posture_score'],
        'Posture_Confidence': head_df['posture_confidence'],
        'Posture_Feedback': head_df['posture_feedback'],
        'Head_Pose': head_df['head_pose'],
        'Head_Pose_Confidence': head_df['head_pose_confidence'],
        'Overall_Confidence': head_df['overall_confidence'],
        
        'Centering_Status': center_status
    })

    column_order = [
        'Timestamp', 'FPS', 'Distance_cm', 'Centering_Status',
        'Posture_Score', 'Posture_Confidence', 'Posture_Feedback',
        'Head_Pose', 'Head_Pose_Confidence', 'Gaze_Direction',
        'Smoothed_EAR_Avg', 'Smoothed_Ratio_H_Avg', 'Smoothed_Ratio_V_Avg',
        'Overall_Confidence', 'Analysis_State', 'Reference_IOD_px', 'Scale_Factor'
    ]
    
    merged_df = merged_df[column_order]

    # os.remove(gaze_path)
    # os.remove(head_path)

    all_metrics_path = os.path.join(METRICS_DIR, f'interview-{interview_id}.csv')

    merged_df.to_csv(all_metrics_path, index=False)
    
    return all_metrics_path


