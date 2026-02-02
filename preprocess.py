import re
import spacy
from nltk.corpus import stopwords
import nltk
from collections import Counter

class JobDataPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Download stopwords if needed
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text, max_keywords=15):
        """Extract important keywords using spaCy"""
        doc = self.nlp(text)
        
        # Extract nouns and proper nouns
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                token.text not in self.stop_words and
                len(token.text) > 2):
                keywords.append(token.lemma_)
        
        # Get most common keywords
        keyword_counts = Counter(keywords)
        return [kw for kw, _ in keyword_counts.most_common(max_keywords)]
    
    def preprocess_dataset(self, df):
        """Preprocess the entire dataset"""
        # Clean job descriptions
        df['cleaned_description'] = df['Job Description'].apply(self.clean_text)
        
        # Extract keywords from job titles and descriptions
        df['job_keywords'] = df.apply(
            lambda x: self.extract_keywords(f"{x['Job Title']} {x['cleaned_description']}"), 
            axis=1
        )
        
        return df