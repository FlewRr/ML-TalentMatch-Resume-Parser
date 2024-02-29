import io
import os
import docx2txt
import re
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from sentence_transformers import SentenceTransformer, util
# import torch
from dateparser.search import search_dates #install
from tqdm import tqdm

def read_docx(file_path):
    text = docx2txt.process(file_path).split('\n')
    text = [line.replace('\t', ' ') for line in text if line != '']

    return '\n'.join(text)


def read_pdf(pdf_path):
    resume = ''
    with open(pdf_path, 'rb') as f:
        for page in PDFPage.get_pages(f,
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle, codec='utf-8', laparams=LAParams())
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
            resume += text

            converter.close()
            fake_file_handle.close()
    return resume


def get_raw_texts(directory):
    resumes = []
    for filename in tqdm(os.listdir(directory)):
        f = os.path.join(directory, filename)
        if f[-4:] == 'docx':
            text = read_docx(f)
            resumes.append(text)
        elif f[-3:] == 'pdf':
            text = read_pdf(f)
            resumes.append(text)
    
    return resumes

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def get_resume_blocks(resumes, model, columns=['Education', 'Experience', 
                                               'Languages', 'Skills', 'Proficiencies', 
                                               'Employment', 'Achievements', 'Projects']):
    stop_word = '\x0c'
    emb = [model.encode(x, convert_to_tensor=True) for x in columns]
    resumes_new = []
    for resume in tqdm(resumes):
        column = 'About me'
        curr_st = {}
        for line in resume.split('\n'):
            if line != '':
                flg = True
            if len([*line.replace(' ', '')]) < 17:
                if len(line.split()) > 5:
                    line = ''.join(line.split())
                emb1 = model.encode(line, convert_to_tensor=True)
                similarities = [util.pytorch_cos_sim(emb1, emb2).item() for emb2 in emb]
                id, _ = similarities.index(max(similarities)), similarities

                if similarities[id] > 0.65:
                    column = columns[id]
                    flg = False
            if flg:
                line = line.replace(stop_word, '')
                if column in curr_st:
                    curr_st[column] += line + '\n'
                else:
                    curr_st[column] = line + '\n'
        resumes_new.append(curr_st)

    return resumes_new

def extract_name(resume, pipeline):
    x = []
    if 'About me' in resume:
        inf = pipeline(resume['About me'])
        for ent in inf:
            if ent['entity_group'] == 'PER' and len(ent['word'].split()) == 2:
                name = (ent['word'].replace('\t', ' ').replace('\n', ' ').split())
                if len(name) == 3 and name[2][0].islower():
                    name = name[:2]
                return name
    # find a way to look for name when name not in the upper body
    return ''


def get_phone(text):
  return  [number for number in re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text) if check(number)]


def check(text):
  text = text.replace(' ', '')
  if len(text) != 9:
    return True
  if text[:4].isdigit() and text[-4:].isdigit() and text[4] == '-':
    return False
  return True


def get_email(text):
  return re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)

def extract_email(resume):
     for key in resume.keys():
        email = get_email(resume[key])
        if email:
           return email
        
def check_year(year):
  if len(year.split()) == 1:
    if year.isdigit() and int(year) > 1950:
      return True
    return False
  else:
    f = False
    g = False
    for y in year.split():
      if y.isdigit():
        g = True
      if not y.isdigit():
        f = True
    return f and g


def extract_education(pipeline, resume):
  if 'Education' not in resume:
     return []
  text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", " ", resume['Education'])
  ner_text = pipeline(text)
  universities = []
  for x in ner_text:
    if x['entity_group'] == 'ORG':
      x['word'] = x['word'].strip()
      if len(x['word']) == 1:
        continue
      if x['word'] == 'University' and len(universities) > 0:
        universities[-1] += ' ' + x['word']
      else:
        universities.append(x['word'])
  x = []
  i = 0
  for line in text.split('\n'):
    dates = search_dates(line)
    if dates:
      small = ''.join([x.lower() for x in line]) +' ' + ''.join([x.lower() for x in universities[i]])
      t = -1
      d = -1
      years = [year[0] for year in dates if check_year(year[0])]  
      if len(years) != 0:
        if 'college' in small:
          t = 4
          d = 2
        elif 'university' in small:
          t = 4
          d = 3 if len(years) == 2 and years[0].isdigit() and int(years[1]) < 2024 or len(years) == 1 and years[0].isdigit() and int(years[0]) < 2024 else 4
        elif 'course' in small:
          t = 3
        else:
          t = 1
        if 'bachelor' in small:
          d = 5
        elif 'major' in small:
          d = 6
        elif 'candidate of science' in small:
          d = 7
        elif 'phd' in small:
          d = 8
      
        x.append([universities[i], years, t, d])
        if i < len(universities) - 1:
          i += 1
  
  # dates = search_dates(text.replace('.', ''))
  # if dates:
    # dates = [year[0] for year in dates]

  # return universities, dates
  return x
                

def extract_urls(text):
    resume_urls = {}
    urls = re.findall(r'(https?://[^\s]+)', text)
    for url in urls:
        if 'linkedin' in url:
            resume_urls['linkedin'] = url
        if 'github' in url:
            resume_urls['github'] = url


def find_contacts(resume):
    github = ''
    phone_number = ''

    for key in resume.keys():
        if not phone_number:
            phone_number = get_phone(resume[key])
        if not github:
            urls = extract_urls(resume[key])
            if urls and 'github' in urls:
                github = urls['github']

        if phone_number and github:
            return phone_number, github

    return phone_number, github


def extract_experience(pipeline, text):
  text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", " ", text)
  ner_text = pipeline(text)

  experience = []

  i = -1
  for j, x in enumerate(ner_text):
    if x['entity_group'] == 'ORG' and x['word'][0].isupper():
      experience.append([x['word']])
      i += 1
    if x['entity_group'] == 'LOC':
      if i != -1 and i < len(experience) and len(experience[i]) == 1:
        experience[i].append(x['word'])

  i = 0
  for line in text.split('\n'):
    dates = search_dates(line)
    if dates:
      years = [year[0] for year in dates if check_year(year[0])]
      if years and i < len(experience):
        experience[i].append(years)
        if i < len(experience) - 1:
          i += 1
  return experience


JSON_FORMAT = {
  "resume": {
      "resume_id": "",
      "first_name": "",
      "last_name": "",
      "middle_name": "",
      "birth_date": "",
      "birth_date_year_only": "",
      "country": "",
      "city": "",
      "about": "",
      "key_skills": "",
      "salary_expectations_amount": "",
      "salary_expectations_currency": "",
      "photo_path": "",
      "gender": "",
      "language": "",
      "resume_name": "",
      "source_link": "",
      "contactItems": [
        {
          "resume_contact_item_id": "",
          "value": "",
          "comment": "",
          "contact_type": ""
        }
      ],
      "educationItems": [
        {
          "resume_education_item_id": "",
          "year": "",
          "organization": "",
          "faculty": "",
          "specialty": "",
          "result": "",
          "education_type": "",
          "education_level": ""
        }
      ],
      "experienceItems": [
        {
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
      ],
      "languageItems": [
        {
          "resume_language_item_id": "",
          "language": "",
          "language_level": ""
        }
      ]
    }
}