from flask import request, json


def validated_data():
    if 'cvFiles' not in request.files:
        raise ValueError('CV files are required.')
    
    if 'jobDescription' not in request.form:
        raise ValueError('The job description is required.')
    
    cv_files = request.files.getlist('cvFiles')

    job_description = request.form['jobDescription'].strip()

    for cv_file in cv_files:
        if not cv_file.filename.lower().endswith('.pdf'):
            raise ValueError(f'The file {cv_file.filename} must be of type PDF.')
        
    if not job_description:
        raise ValueError('The job description field mustn\'t be empty.')

    formatted_job, skills = format_job(job_description)

    return cv_files, formatted_job, skills



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