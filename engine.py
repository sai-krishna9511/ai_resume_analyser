# Import necessary libraries
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import pandas as pd # Using pandas for professional data handling

# Janni's Custom List of Stop Words (to avoid NLTK dependency issues)
# This makes our application much more reliable and professional.
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'job',
    'description', 'resume', 'experience', 'skills', 'duties', 'responsibilities', 'work',
    'company', 'role', 'team', 'project', 'candidate', 'requirements', 'qualifications'
])

# --- Helper Function 1: Text Extraction from PDF ---
def extract_text_from_pdf(pdf_file_stream):
    """
    This function takes a PDF file stream and returns the extracted raw text.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# --- Helper Function 2: Text Preprocessing (Janni's Custom Version) ---
def preprocess_text(text):
    """
    This function cleans raw text using standard Python libraries, avoiding NLTK.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(filtered_tokens)

# --- Main Analysis Function (Upgraded with Advanced Keywords) ---
def analyze_resume(resume_text, job_description_text):
    """
    This function performs the core AI analysis and returns the match score
    and a list of the top 10 most important missing keywords.
    """
    cleaned_resume = preprocess_text(resume_text)
    cleaned_jd = preprocess_text(job_description_text)
    
    # Handle cases where text is empty after cleaning
    if not cleaned_resume or not cleaned_jd:
        return 0, []

    text_corpus = [cleaned_resume, cleaned_jd]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percentage = round(similarity_score * 100, 2)
    
    feature_names = vectorizer.get_feature_names_out()
    jd_vector = tfidf_matrix[1].toarray().flatten()
    resume_words = set(cleaned_resume.split())

    df_keywords = pd.DataFrame({'keyword': feature_names, 'importance': jd_vector})
    df_keywords = df_keywords[df_keywords['importance'] > 0.1] 
    
    df_missing = df_keywords[~df_keywords['keyword'].isin(resume_words)]
    df_missing = df_missing.sort_values(by='importance', ascending=False)
    top_missing_keywords = df_missing['keyword'].head(10).tolist()
    
    return match_percentage, top_missing_keywords


