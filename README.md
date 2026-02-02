# 📄 AI-Powered Resume Analyzer

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

🎯 **Project Overview**

An intelligent resume analysis system that uses Natural Language Processing (NLP) to match resumes with job descriptions. This tool helps job seekers understand how well their skills align with job opportunities and identifies areas for improvement.

<img width="1912" height="1022" alt="image" src="https://github.com/user-attachments/assets/17e778b4-0982-4882-b52a-27a218e110be" />


<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/3587150b-a941-4421-a26f-ad95dfd31404" />

---

## ✨ Features

### 🔍 Core Functionality

- Resume Parsing (PDF/DOCX)
- Automatic Skill Extraction
- Semantic Matching using Transformers
- Skill Gap Analysis
- Interactive Visual Analytics

### 📊 Dashboard Insights

- Match Scores
- Skill Inventory
- Experience Detection
- Improvement Recommendations
- Industry Trends Visualization

---

## 🏗️ Architecture

```bash
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Resume Upload │    │  NLP Processing │    │  Job Matching   │
│    (PDF/DOCX)   │───▶│   & Analysis    │───▶│   & Ranking     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Extract  │    │  Skill Mapping  │    │  Visualization  │
│  & Cleaning     │    │   & Keyword     │    │   & Reporting   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```


---

## 🛠️ Technology Stack

### Backend & NLP
- Python 3.10+
- spaCy
- Hugging Face Transformers
- Sentence Transformers
- scikit-learn

### Data Processing
- pandas
- NumPy
- NLTK

### File Processing
- pdfplumber
- python-docx

### Frontend & Visualization
- Streamlit
- Plotly
- Matplotlib

### Machine Learning
- PyTorch
- scikit-learn

---

## 📁 Project Structure

```
resume-analyzer/
├── 📁 data/                          # Dataset storage
│   ├── job_title_des.csv            # Original dataset (from Kaggle)
│   └── job_data_clean.csv           # Cleaned dataset (auto-generated)
├── 📁 src/                           # Source code modules
│   ├── __init__.py                  # Package initialization
│   ├── preprocessor.py              # Text preprocessing and cleaning
│   ├── resume_parser.py             # Resume parsing and extraction
│   └── matcher.py                   # Semantic matching engine
├── 📁 cached_embeddings/             # Cached job embeddings (auto-generated)
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # This documentation
├── setup.py                         # Package installation script
└── .env.example                     # Environment variables template
```


---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git
- 4GB RAM (8GB Recommended)

---

### Installation

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/resume-analyzer.git
cd resume-analyzer
```

#### 2. Create Virtual Environment

```
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Download Models

```
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

### 5. Run Application

```
streamlit run app.py
```
Open in browser:

```
http://localhost:8501
```

## 📖 Usage Guide

### Step 1: Load Dataset
- Click **Load Dataset**
- Wait for embeddings (first run only)

### Step 2: Upload Resume
- Upload PDF or DOCX file

### Step 3: Analyze Results
- View extracted skills
- Check job matches
- Explore visualizations

### Step 4: Export Results
- Screenshot charts
- Save recommendations

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
HF_TOKEN=your_token_here
EMBEDDINGS_CACHE=true
CACHE_EXPIRY_DAYS=30
MODEL_NAME=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.3
```

## Model Options

Edit src/matcher.py:

```
FAST_MODEL = 'all-MiniLM-L6-v2'
BALANCED_MODEL = 'all-mpnet-base-v2'
ACCURATE_MODEL = 'all-distilroberta-v1'
```

## 📊 Dataset Information
- Source: Kaggle (Jobs Dataset)
- Size: 2,277 Records
- Format: CSV
- License: CC0

| Column          | Description  | Example        |
|-----------------|--------------|----------------|
| Job Title       | Role Name    | Data Scientist |
| Job Description | Requirements | 5+ years exp   |

## 🔍 How It Works
1. Resume Pipeline
```
Resume → Extraction → Cleaning → Skill Detection → Experience
```

2. Matching Logic
```
resume_embedding = model.encode(resume_text)
similarities = cosine_similarity(resume_embedding, job_embeddings)
```

4. Skill Gap Example
```
User Skills: Python, SQL, ML
Job Needs: Python, SQL, AWS, Docker
Missing: AWS, Docker
Match: 66.7%
```

## 🚢 Deployment

## Local
```
streamlit run app.py --server.address 0.0.0.0
```

## Docker
```
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

docker build -t resume-analyzer .
docker run -p 8501:8501 resume-analyzer
```

## 🧪 Testing

## Generate Test Resume
```
python generate_test_resume.py
```
## Run Tests
```
pytest tests/
```
## 🔄 API Integration (Optional)
```
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile):
    return {"score": 0.85}
```
## 🤝 Contributing

## Steps
```
git checkout -b feature/new-feature
git commit -m "Add feature"
git push origin feature/new-feature
```
## Areas
- Bug Fixes
- Optimization
- Multilingual Support
- API Integration
- Mobile App

## 📝 License
```
MIT License
```
## 🙏 Acknowledgments
- Kaggle Dataset
- Hugging Face
- Streamlit
- Plotly

