from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pickle, os


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flan-t5 model
flan_model_path = os.path.join(MODEL_DIR, 'flan-t5-large-8bit')
tokenizer = AutoTokenizer.from_pretrained(flan_model_path)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_path)

# Pipelines
# pipeline_device = 0 if torch.cuda.is_available() else -1
pipeline_info = pipeline("text2text-generation", model=flan_model, tokenizer=tokenizer, max_length=256, do_sample=True, temperature=0.2)
pipeline_skills = pipeline("text2text-generation", model=flan_model, tokenizer=tokenizer, max_length=256, do_sample=True, temperature=0.1, top_p=0.9)


# BERT model
bert_model_path = os.path.join(MODEL_DIR, "bert_model.pkl")
with open(bert_model_path, 'rb') as f:
    bert_model=pickle.load(f)