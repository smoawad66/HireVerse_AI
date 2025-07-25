import spacy



# if not hasattr(bert_model.tokenizer, 'pad_token') or bert_model.tokenizer.pad_token is None:
#     bert_model.tokenizer.pad_token = '[PAD]'
    
# if not hasattr(bert_model.tokenizer, 'pad_token_id') or bert_model.tokenizer.pad_token_id is None:
#     bert_model.tokenizer.pad_token_id = 0


questions3 = {
    "What faculty or department is associated with the institution mentioned in the text?": "faculty",
    "What are the dates of attendance mentioned in the text? (e.g., 2018 - 2022)": "dates",
    "What is the location of the institution mentioned in the text?": "location",
    "What is the cumulative GPA mentioned in the text? (e.g., Cumulative GPA 3.73)": "cumulative_gpa",
}


questions1 = {
    "What is the candidate's full name?": "full_name",
    "What is the candidate's email address?": "email",
    "What is the candidate's phone number?": "phone_number",
    "What is the candidate's current academic level and major?": "academic_level",

}


# Load spaCy NLP model (Ensure 'en_core_web_sm' is installed)
nlp = spacy.load("en_core_web_sm")

# Custom stop words list (O(1) lookup with a set)
CUSTOM_STOP_WORDS = {
    'activities', 'volunteer', 'proficient', 'course', 'courses', 'excellent', 'student', 'frameworks',
    'introduction', 'competitive', 'strong', 'skills', 'skill', 'management', 'using', 'organizations', 'proven',
    'represent', 'experience', 'familiar', 'learning'
}

# Regular expression pattern to detect common date formats
DATE_PATTERN = r'\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\b\d{4}\b)\b'

