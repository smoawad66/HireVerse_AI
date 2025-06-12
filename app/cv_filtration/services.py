from ..shared.models import bert_model, pipeline_info, pipeline_skills
from langchain_huggingface import HuggingFacePipeline
from sentence_transformers import util
from .validator import validated_data
from .constants import *
import fitz, re



def evaluate_cvs():
    cv_files, job_description, required_skills = validated_data()
    cv_scores = []
    
    if not required_skills:
        required_skills = extract_skills(job_description)
     
    for cv_file in cv_files:
        resume_text = extract_text_from_pdf(cv_file)
        personal_text, remaining_text = extract_and_concatenate_profile(resume_text)
        academic_info, remaining_text = extract_academic_info(remaining_text)

        applicant_data = {}
        for question, field in questions1.items():
            answer = extract_information(personal_text, question)
            applicant_data[field] = answer

        for question, field in questions3.items():
            answer = extract_information(academic_info, question)
            applicant_data[field] = answer
        
        applicant_data['skills'] = extract_skills(remaining_text)
        applicant_data['skills'] = filter_text(applicant_data['skills'])

        similarity_score = bert_similarity(required_skills, applicant_data['skills'])
        applicant_data['match_percentage'] = similarity_score

        cv_scores.append(applicant_data['match_percentage'])
        
    return cv_scores


def extract_text_from_pdf(pdf_file):

    # pdf_document = fitz.open(pdf_path)
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    extracted_text = ""

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        blocks = page.get_text("blocks")  # Extract blocks of text

        for block in blocks:
            block_text = block[4]  # The text of the block
            extracted_text += block_text + " "  # Add a space instead of a line break

    # Remove any extra whitespace
    extracted_text = " ".join(extracted_text.split())

    return extracted_text


def extract_and_concatenate_profile(text):
    # List of possible section titles for the profile
    possible_sections = ['Profile', 'About Me']
    
    # Create a regex pattern to match any of the section titles
    pattern = r'(' + '|'.join(possible_sections) + r')\s+(.*?)(?=\b(Education|Professional Experience|Skills|Courses|Organizations|$)\b)'
    
    # Use regex to find the "Profile" section and capture the preceding text and the profile section
    profile_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if profile_match:
        # Extract the text preceding the Profile section
        preceding_text = text[:profile_match.start()].strip()
        
        # Extract the Profile section
        profile_section = profile_match.group(2).strip()
        
        # Extract the remaining text after the Profile section
        remaining_text = text[profile_match.end():].strip()
        
        # Concatenate the preceding text with the Profile section
        combined_text = preceding_text + "\n\n" + profile_match.group(1) + "\n" + profile_section
        
        return combined_text, remaining_text
    else:
        return "Profile section not found.", text


def extract_academic_info(text):
    # Use a generalized regex pattern to find the academic information
    pattern = r'(education.*?)(?=\b(Professional Experience|Skills|Courses|Organizations|$)\b)'
    academic_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if academic_match:
        # Extract the academic information section
        academic_info = academic_match.group(1).strip()
        text = text[academic_match.end():].strip()
        
        return academic_info , text
    else:
        return "Academic information not found.", text
    

def extract_information(resume_text, question=""):
    llm = HuggingFacePipeline(pipeline=pipeline_info)

    prompt =f"""You are an intelligent assistant designed to extract information from resumes.
    Below is a specific question related to a candidate's resume.
    Please read the provided resume text carefully and answer the question based on the information contained within it.
      
        Resume Text:
        {resume_text}

        Question:
        {question}
        If the answer cannot be found, respond with "Not found."""
    
    return llm.invoke(prompt)


def extract_skills(text):

    try:
        llm = HuggingFacePipeline(pipeline=pipeline_skills)
        
        prompt = f"""  You are an advanced AI assistant specialized in analyzing and extracting valuable information from resume text. Your task is to extract and list all the skills mentioned in the provided text. Focus solely on identifying skills, including technical skills, soft skills. 
        
        Resume Text:
        {text}
        """

        # Call the model with the prompt
        response = llm.invoke(prompt)

        if isinstance(response, str):
            return response

        return "No skills found."

    except ValueError as ve:
        return f"ValueError encountered: {ve}"

    except TypeError as te:
        return f"TypeError encountered: {te}"

    except Exception as e:
        return f"An error occurred: {e}"


def filter_text(text, word="Courses"):
    # Check if the word exists first
    if word.lower() in text.lower():  
        match = re.search(rf'\b{word}\b', text, flags=re.IGNORECASE)
        if match:
            return text[:match.start()].strip()  # Cut everything after the word
    return text  # Return original text if "Courses" isn't found


def remove_dates(text):
    """Removes dates from text using regex."""
    text = re.sub(DATE_PATTERN, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text


def process_text(text):
    """Extracts skills from text while removing stop words and dates."""
    # **Step 1: Remove Dates**
    text = remove_dates(text)

    # **Step 2: Preprocess Text Efficiently**
    text = re.sub(r'[•,.-]', ' ', text)  # Replace bullets, commas, dashes with space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

    # **Step 3: Process Text with spaCy**
    doc = nlp(text)

    extracted_skills = set()  # Use a set to store unique skills
    temp_skill = []

    # **Step 4: Extract Skills Efficiently**
    for token in doc:
        token_text = token.text.lower()

        # **Skip stop words and unwanted words**
        if token.is_stop or token_text in CUSTOM_STOP_WORDS:
            if temp_skill:
                extracted_skills.add(" ".join(temp_skill))
                temp_skill = []
            continue

        # **Capture meaningful words (nouns, adjectives) for skills**
        if token.pos_ in {'NOUN', 'PROPN', 'ADJ'}:
            temp_skill.append(token_text)
        else:
            if temp_skill:
                extracted_skills.add(" ".join(temp_skill))
                temp_skill = []

    # **Ensure last skill is added**
    if temp_skill:
        extracted_skills.add(" ".join(temp_skill))

    # **Step 5: Return Sorted Unique Skills**
    return sorted(extracted_skills)


def bert_similarity(text1, text2):
    # Process the texts to filter out unwanted elements
    filter_require = process_text(text1)
    filter_skills = process_text(text2)
    
    # Encode the filtered texts to get embeddings
    embeddings = bert_model.encode([' '.join(filter_require), ' '.join(filter_skills)], convert_to_tensor=True)
    
    # Calculate similarity using PyTorch's cosine similarity
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    
    # Convert similarity to float
    similarity_score = float(similarity.item()) * 100
    
    return round(similarity_score, 2)

