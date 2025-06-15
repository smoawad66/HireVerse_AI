import os, csv
import botocore.exceptions
from dotenv import load_dotenv
import os, boto3, time, botocore
import pandas as pd



BASE_DIR = os.path.dirname(__file__)
METRICS_DIR = os.path.join(BASE_DIR, 'soft_skills/analysis_metrics')


load_dotenv()
BUCKET_NAME = 'myawshierbucket'
s3 = boto3.client('s3')

def download_file(key, local_path):
    try:
        s3.download_file(BUCKET_NAME, key, local_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Error: File not found in S3: {key}")
        else:
            print(f"Error downloading file: {e}")
        return False
    print('File downloaded')
    return True



def upload_file(local_path, key):
    try:
        s3.upload_file(local_path, BUCKET_NAME, key)
    except botocore.exceptions.ClientError as e:
        print(f"Error uploading file: {e}")
        return False
    return True




def download_applicant_answers(keys): 
    videos_folder = create_folder('technical_skills/videos')

    paths = []
    for key in keys:
        file_name = key.split('/')[-1]
        path = os.path.join(videos_folder, file_name)
        res = download_file(key, path)
        paths.append(None if res is False else path)
    return paths


def download_questions_file(key):
    questions_folder = create_folder('technical_skills/questions')
    
    path = os.path.join(questions_folder, f'questions-{int(time.time())}.json')
    download_file(key, path)
    return path


def setup_logging(interview_id):
    gaze_folder = create_folder(f'{METRICS_DIR}/gaze')

    gaze_csv_path = f'{gaze_folder}/interview-{interview_id}.csv'
    with open(gaze_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Frame', 'FPS', 'Distance_cm', 'Gaze_Direction','Smoothed_EAR_Avg', 'Smoothed_Ratio_H_Avg', 'Smoothed_Ratio_V_Avg', 'Analysis_State', 'Reference_IOD_px', 'Scale_Factor', 'Smiling'])

    head_folder = create_folder(f'{METRICS_DIR}/head')
    head_csv_path = f'{head_folder}/interview-{interview_id}.csv'
    with open(head_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'posture_score', 'posture_confidence', 'posture_feedback', 'head_pose', 'head_pose_confidence', 'overall_confidence', 'fps'])
    
    return gaze_csv_path, head_csv_path




def create_folder(folder_path):
    folder_path = os.path.join(BASE_DIR, folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path


def merge_head_gaze(gaze_path, head_path, interview_id):

    # Read both CSV files
    gaze_df = pd.read_csv(gaze_path)
    head_df = pd.read_csv(head_path)

    merged_df = pd.DataFrame({
        'Timestamp': gaze_df['Timestamp'],
        'FPS': gaze_df['FPS'],
        'Distance_cm': gaze_df['Distance_cm'],
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
        
        'Centering_Status': None
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


