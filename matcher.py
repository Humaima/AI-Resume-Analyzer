import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResumeMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Suppress Hugging Face warnings during model loading
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Load model with minimal logging
        self.model = SentenceTransformer(model_name, device='cpu')
        self.job_embeddings = None
        self.job_data = None
        
    def prepare_job_embeddings(self, job_data):
        """Create embeddings for job descriptions"""
        self.job_data = job_data
        
        # Combine title and description for better representation
        job_texts = [
            f"{row['Job Title']} {row['cleaned_description']}" 
            for _, row in job_data.iterrows()
        ]
        
        # Generate embeddings with progress bar
        self.job_embeddings = self.model.encode(
            job_texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return self.job_embeddings
    
    def match_resume_to_jobs(self, resume_text, top_k=10):
        """Match resume to job descriptions"""
        # Check if embeddings are prepared
        if self.job_embeddings is None or self.job_data is None:
            raise ValueError("Job embeddings not prepared. Call prepare_job_embeddings() first.")
        
        # Generate embedding for resume
        resume_embedding = self.model.encode([resume_text])
        
        # Calculate similarity scores
        similarities = cosine_similarity(resume_embedding, self.job_embeddings)[0]
        
        # Get top matching jobs
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'job_title': self.job_data.iloc[idx]['Job Title'],
                'similarity_score': float(similarities[idx]),
                'description_preview': self.job_data.iloc[idx]['cleaned_description'][:200] + "..." 
                if isinstance(self.job_data.iloc[idx]['cleaned_description'], str) else "",
                'keywords': self.job_data.iloc[idx]['job_keywords'][:5] 
                if isinstance(self.job_data.iloc[idx]['job_keywords'], list) else []
            })
        
        return results
    
    def get_skill_gap_analysis(self, resume_skills, job_keywords):
        """Analyze skill gaps between resume and job"""
        if not isinstance(job_keywords, list):
            job_keywords = []
        
        resume_skills_set = set(resume_skills)
        job_keywords_set = set(job_keywords)
        
        matching_skills = resume_skills_set.intersection(job_keywords_set)
        missing_skills = job_keywords_set - resume_skills_set
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'match_percentage': len(matching_skills) / len(job_keywords_set) * 100 if job_keywords_set else 0
        }