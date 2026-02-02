import pdfplumber
import docx
import re
from typing import Dict, List

class ResumeParser:
    def __init__(self):
        from src.preprocess import JobDataPreprocessor
        self.preprocess = JobDataPreprocessor()
        
    def extract_text_from_file(self, file_path):
        """Extract text from PDF or DOCX files"""
        text = ""
        
        try:
            if file_path.endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                        
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
        
        return text
    
    def parse_resume(self, resume_text):
        """Parse resume and extract key information"""
        # Clean resume text
        cleaned_text = self.preprocess.clean_text(resume_text)
        
        # Extract skills (common technical terms)
        skills_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'sql', 'html', 'css'],
            'tools': ['git', 'docker', 'aws', 'jenkins', 'tableau'],
            'ml': ['machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch']
        }
        
        extracted_skills = []
        for category, keywords in skills_keywords.items():
            for keyword in keywords:
                if keyword in cleaned_text:
                    extracted_skills.append(keyword)
        
        # Extract keywords
        keywords = self.preprocess.extract_keywords(cleaned_text, max_keywords=20)
        
        return {
            'cleaned_text': cleaned_text,
            'skills': list(set(extracted_skills)),
            'keywords': keywords,
            'experience_years': self.extract_experience(cleaned_text)
        }
    
    def extract_experience(self, text):
        """Extract years of experience from resume"""
        patterns = [
            r'(\d+)\+?\s*years?.*experience',
            r'experience.*(\d+)\+?\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return int(match.group(1))
                except:
                    return 0
        
        return 0