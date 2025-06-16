import botocore.exceptions
from dotenv import load_dotenv
import os, boto3, time, botocore
import pandas as pd

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




def create_folder(base, folder_path):
    folder_path = os.path.join(base, folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path
