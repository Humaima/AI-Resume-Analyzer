import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# Import from src modules
from src.preprocess import JobDataPreprocessor
from src.resume_parser import ResumeParser
from src.matcher import ResumeMatcher

class ResumeAnalyzerApp:
    def __init__(self):
        st.set_page_config(
            page_title="AI Resume Analyzer",
            page_icon="📄",
            layout="wide"
        )
        
        # Initialize components
        self.matcher = ResumeMatcher()
        self.parser = ResumeParser()
        self.df_processed = None
        
    def load_and_prepare_data(self):
        """Load and prepare the job dataset"""
        try:
            # Try to load cleaned dataset first
            if os.path.exists('data/job_data_clean.csv'):
                df = pd.read_csv('data/job_data_clean.csv')
                st.write("✅ Loaded cleaned dataset")
            else:
                # Load original dataset
                df = pd.read_csv('data/job_title_des.csv')
                
                # FIX: Check if first column is an index column and rename properly
                if len(df.columns) == 3 and 'Unnamed: 0' in df.columns:
                    # If we have 3 columns with an Unnamed column, drop it
                    df = df.drop(columns=['Unnamed: 0'])
                elif len(df.columns) == 1:
                    # If all data is in one column, try to split it
                    st.warning("Dataset seems to have formatting issues. Attempting to fix...")
                    df = self.fix_single_column_dataset(df)
                
                # Save cleaned version for future use
                df.to_csv('data/job_data_clean.csv', index=False)
                st.write("✅ Created and saved cleaned dataset")
            
            # Display dataset info
            with st.expander("📊 Dataset Info"):
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {df.columns.tolist()}")
                st.write("**Sample Job Titles:**")
                st.write(df['Job Title'].head(10).tolist())
            
            # Preprocess
            preprocessor = JobDataPreprocessor()
            self.df_processed = preprocessor.preprocess_dataset(df)
            
            # Prepare embeddings
            with st.spinner("Creating semantic embeddings (this may take a minute)..."):
                self.matcher.prepare_job_embeddings(self.df_processed)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.write("Debug info - Available files:", os.listdir('data') if os.path.exists('data') else "No data folder")
            return False
    
    def fix_single_column_dataset(self, df):
        """Fix dataset if it's all in one column"""
        # Get the first column name
        first_col = df.columns[0]
        
        # Try to split by comma or other delimiters
        if df[first_col].str.contains(',,').any():
            # Split by double comma if present
            split_df = df[first_col].str.split(',,', expand=True)
        else:
            # Split by comma
            split_df = df[first_col].str.split(',', expand=True)
        
        # Check if we got at least 2 columns
        if split_df.shape[1] >= 2:
            split_df = split_df.iloc[:, :2]  # Take first 2 columns
            split_df.columns = ['Job Title', 'Job Description']
            return split_df
        else:
            # Manual parsing for tricky formats
            titles = []
            descriptions = []
            
            for text in df[first_col].astype(str):
                # Simple heuristic: first 50 chars as title, rest as description
                if len(text) > 50:
                    titles.append(text[:50].strip())
                    descriptions.append(text[50:].strip())
                else:
                    titles.append(text.strip())
                    descriptions.append("")
            
            return pd.DataFrame({'Job Title': titles, 'Job Description': descriptions})
    
    def run(self):
        st.title("🤖 AI-Powered Resume Analyzer")
        st.markdown("Upload your resume and find matching job opportunities!")
        
        # Sidebar for file upload
        with st.sidebar:
            st.header("📤 Upload Resume")
            uploaded_file = st.file_uploader(
                "Choose a PDF or DOCX file",
                type=['pdf', 'docx']
            )
            
            st.header("⚙️ Settings")
            top_matches = st.slider("Number of job matches", 5, 20, 10)
            min_similarity = st.slider("Minimum similarity score", 0.0, 1.0, 0.3)
            
            st.header("📁 Dataset")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Load Dataset", type="primary", use_container_width=True):
                    with st.spinner("Loading dataset..."):
                        if self.load_and_prepare_data():
                            st.success(f"✅ Loaded {len(self.df_processed)} jobs!")
                        else:
                            st.error("Failed to load dataset")
            
            with col2:
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    if os.path.exists('data/job_data_clean.csv'):
                        os.remove('data/job_data_clean.csv')
                    st.success("Cache cleared!")
            
            # Show dataset status
            if self.df_processed is not None:
                st.info(f"📊 Dataset: {len(self.df_processed)} jobs")
            else:
                st.warning("⚠️ Load dataset first")
        
        # Main content area
        if uploaded_file is not None:
            # Check if dataset is loaded
            if self.df_processed is None:
                st.error("Please load the job dataset first!")
                st.info("Click the 'Load Dataset' button in the sidebar")
                return
            
            # Save uploaded file
            file_ext = uploaded_file.name.split('.')[-1].lower()
            temp_path = f"temp_resume.{file_ext}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Parse resume
            with st.spinner("Analyzing your resume..."):
                resume_text = self.parser.extract_text_from_file(temp_path)
                
                if not resume_text or len(resume_text.strip()) < 10:
                    st.error("Could not extract text from resume. Please try a different file.")
                    return
                
                parsed_resume = self.parser.parse_resume(resume_text)
                
                # Display resume insights
                st.subheader("📋 Resume Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Skills Found", len(parsed_resume['skills']))
                with col2:
                    st.metric("Keywords", len(parsed_resume['keywords']))
                with col3:
                    exp = parsed_resume['experience_years']
                    exp_text = f"{exp} years" if exp > 0 else "Not specified"
                    st.metric("Experience", exp_text)
                
                # Show extracted skills
                if parsed_resume['skills']:
                    st.write("**🔍 Extracted Skills:**")
                    cols = st.columns(3)
                    for i, skill in enumerate(parsed_resume['skills']):
                        cols[i % 3].success(f"✓ {skill}")
                
                # Get job matches
                matches = self.matcher.match_resume_to_jobs(
                    parsed_resume['cleaned_text'],
                    top_k=top_matches
                )
                
                # Filter by similarity threshold
                matches = [m for m in matches if m['similarity_score'] >= min_similarity]
                
                # Display matches
                if matches:
                    st.subheader(f"🎯 Top {len(matches)} Job Matches")
                    
                    for i, match in enumerate(matches, 1):
                        # Create a progress bar-like visualization for score
                        score_color = "green" if match['similarity_score'] > 0.7 else "orange" if match['similarity_score'] > 0.4 else "red"
                        
                        with st.expander(f"**{i}. {match['job_title']}** - Score: `{match['similarity_score']:.1%}`", expanded=i==1):
                            st.markdown(f"**Match Score:**")
                            st.progress(match['similarity_score'])
                            
                            st.markdown("**Description Preview:**")
                            st.info(match['description_preview'])
                            
                            # Skill gap analysis
                            gap_analysis = self.matcher.get_skill_gap_analysis(
                                parsed_resume['skills'],
                                match['keywords']
                            )
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**✅ Matching Skills:**")
                                if gap_analysis['matching_skills']:
                                    for skill in gap_analysis['matching_skills'][:8]:
                                        st.success(f"• {skill}")
                                else:
                                    st.info("No matching skills found")
                            
                            with col2:
                                st.markdown("**📚 Skills to Develop:**")
                                if gap_analysis['missing_skills']:
                                    for skill in gap_analysis['missing_skills'][:8]:
                                        st.warning(f"• {skill}")
                                else:
                                    st.success("Perfect match! No missing skills.")
                            
                            st.caption(f"**Overall Skill Match:** {gap_analysis['match_percentage']:.1f}%")
                    
                    # Visualization
                    self.create_visualizations(matches, parsed_resume)
                else:
                    st.warning("""
                    No job matches found above the similarity threshold. 
                    
                    **Try:**
                    1. Lowering the similarity threshold in sidebar
                    2. Adding more skills/details to your resume
                    3. Loading a different resume
                    """)
        
        else:
            # Show instructions when no resume is uploaded
            st.markdown("""
            ## 🚀 Welcome to AI Resume Analyzer
            
            This tool uses **Natural Language Processing** to:
            - 📝 **Parse** your resume and extract key skills
            - 🔍 **Match** your profile with job opportunities
            - 📊 **Analyze** skill gaps and provide insights
            - 📈 **Visualize** your career fit
            
            ### 📋 Quick Start:
            1. **Load Dataset** → Click the button in sidebar
            2. **Upload Resume** → PDF or DOCX format
            3. **Get Insights** → View matches and analysis
            
            ### 📁 Sample Resumes:
            You can use any PDF/DOCX resume. For testing, you can create a simple resume with:
            - Your name and contact info
            - Work experience with years
            - Skills (Python, Java, ML, etc.)
            - Education background
            """)
    
    def create_visualizations(self, matches, parsed_resume):
        """Create interactive visualizations"""
        st.subheader("📊 Match Analysis Dashboard")
        
        # Prepare data for charts
        match_scores = [m['similarity_score'] for m in matches]
        job_titles = [m['job_title'] for m in matches]
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for similarity scores
            fig1 = px.bar(
                x=job_titles,
                y=match_scores,
                title="Job Match Similarity Scores",
                labels={'x': 'Job Title', 'y': 'Similarity'},
                color=match_scores,
                color_continuous_scale='Viridis'
            )
            fig1.update_layout(
                xaxis_tickangle=-45,
                xaxis_title="",
                yaxis_title="Match Score",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Pie chart for skill distribution
            if parsed_resume['skills']:
                skills_count = len(parsed_resume['skills'])
                if matches:
                    top_match = matches[0]
                    gap_analysis = self.matcher.get_skill_gap_analysis(
                        parsed_resume['skills'],
                        top_match['keywords']
                    )
                    
                    labels = ['Matching Skills', 'Missing Skills']
                    values = [
                        len(gap_analysis['matching_skills']),
                        len(gap_analysis['missing_skills'])
                    ]
                    
                    fig2 = px.pie(
                        values=values,
                        names=labels,
                        title=f"Skill Match for: {top_match['job_title'][:30]}...",
                        color=labels,
                        color_discrete_map={'Matching Skills':'green', 'Missing Skills':'orange'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)

# Main execution
if __name__ == "__main__":
    app = ResumeAnalyzerApp()
    app.run()