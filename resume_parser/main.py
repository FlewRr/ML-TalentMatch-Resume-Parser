from utils import read_docx, read_pdf, get_raw_texts, get_resume_blocks, extract_name, extract_email, extract_education, find_contacts, extract_experience, JSON_FORMAT
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
# todo: github/linkedin, work exp


DIRECTORY = "resume_parser/resumes"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

resumes = get_raw_texts(DIRECTORY)
resumes = get_resume_blocks(resumes, model)

for id, resume in enumerate(resumes):
    name = extract_name(resume, pipeline)
    phone_number, github = find_contacts(resume)
    email = extract_email(resume)
    education_array = extract_education(pipeline, resume)
    experience_array = []
    if 'Experience' in resume:
        experience_array = extract_experience(pipeline, resume['Experience'])
    elif 'Employment' in resume:
        experience_array = extract_experience(pipeline, resume['Experience'])

    resume = JSON_FORMAT['resume']
    resume['resume_id'] = id+1
    if name:
        resume['first_name'] = ''.join(x.lower() for x in name[0])
        resume['last_name'] =  ''.join(x.lower() for x in name[1])
        
    i = 1
    contact_dict = {
          "resume_contact_item_id": "",
          "value": "",
          "comment": "",
          "contact_type": ""
        }
    if phone_number:
        contact_dict["resume_contact_item_id"] = i
        contact_dict["value"] = phone_number
        contact_dict["contact_type"] = 1
        i += 1
    if email:
        contact_dict["resume_contact_item_id"] = i
        contact_dict["value"] = email
        contact_dict["contact_type"] = 2
        i += 1
    if github:
        contact_dict["resume_contact_item_id"] = i
        contact_dict["value"] = github
        contact_dict["contact_type"] = 5
        i += 1
    
    education = {
          "resume_education_item_id": "",
          "year": "",
          "organization": "",
          "faculty": "",
          "specialty": "",
          "result": "",
          "education_type": "",
          "education_level": ""
        }

    for i, array in enumerate(education_array):
        education['resume_education_item_id'] = i + 1
        if len(array[1]) > 1:
            education['year'] = array[1][1]
        elif len(array[1]) == 1:
            education['year'] = array[1][0]
        education['organization'] = array[0]
        if array[2] != -1:
            education['education_type'] = array[2]
        if array[3] != -1:
            education['education_level'] = array[3]
    

    experience = {
          "resume_experience_item_id": "",
          "starts": "",
          "ends": "",
          "employer": "",
          "city": "",
          "url": "",
          "position": "",
          "description": "",
          "order": ""
        }
    for i, arr in enumerate(experience_array):
        if len(arr) == 3:
            experience["resume_experience_item_id"] = i + 1
            experience["city"] = arr[1]
            experience['employer'] = arr[0]

            if len(arr[2]) == 2:
                experience['starts'] = arr[2][0]
                experience['ends'] = arr[2][1]
            else:
                experience['starts'] = arr[2][0]
                experience['ends'] = 'Present'
        
    resume['contactItems'] = contact_dict
    resume['educationItems'] = education
    resume['experienceItems'] = experience

    with open(f'resume_parser/jsons/{id}_resume.json', 'w', encoding='utf-8') as f:
        json.dump(resume, f, ensure_ascii=False, indent=4)