from flask import request, json
from ..helpers import download_file, create_folder
import os


BASE_DIR = os.path.dirname(__file__)


def download_cvs(keys): 
    cvs_folder = create_folder(BASE_DIR, 'cvs')

    paths = []
    for key in keys:
        file_name = key.split('/')[-1]
        path = os.path.join(cvs_folder, file_name)
        res = download_file(key, path)
        paths.append(None if res is False else path)
    return paths


def validated_data():
    data = request.get_json()

    job_description = data['jobDescription']
    cvs_paths = data['cvsPaths']

    local_cvs_paths = download_cvs(cvs_paths)

    # if 'cvFiles' not in request.files:
    #     raise ValueError('CV files are required.')
    
    # if 'jobDescription' not in request.form:
    #     raise ValueError('The job description is required.')
    
    # cv_files = request.files.getlist('cvFiles')

    # job_description = request.form['jobDescription'].strip()

    # for cv_file in cv_files:
    #     if not cv_file.filename.lower().endswith('.pdf'):
    #         raise ValueError(f'The file {cv_file.filename} must be of type PDF.')
        
    # if not job_description:
    #     raise ValueError('The job description field mustn\'t be empty.')

    formatted_job, skills = format_job(job_description)

    return local_cvs_paths, formatted_job, skills



def format_job(job_description):
    try:
        lines = []
        job_description_dict = json.loads(job_description)
        attributes = job_description_dict.get('attributes', {})

        lines.append(f"Job Title: {attributes['title']}.")
        lines.append(f"Experience Level: {attributes['experienceLevel']}")
        lines.append(f"Requirements: {attributes['requirements']}")
        lines.append(f"Job Location: {attributes['jobLocation']}")
        lines.append(f"Summary: {attributes['summary']}")
        lines.append(f"Responsibilities: {attributes['responsibilities']}")
 
        skills = ", ".join(attributes['skills'])

        return "\n".join(lines), skills
     
    except Exception as e:
        return str(e)