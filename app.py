import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def calculate_similarity(resume_text, job_description):
    corpus = [resume_text, job_description]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

st.title("📄 AI Resume Screener (Lightweight Version)")

resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
job_description = st.text_area("Paste the Job Description")

if resume_file and job_description:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(resume_file)
        cleaned_resume = preprocess_text(resume_text)
        cleaned_jd = preprocess_text(job_description)
        similarity = calculate_similarity(cleaned_resume, cleaned_jd)
        st.success(f"Match Score: {round(similarity * 100, 2)}%")
        if similarity >= 75:
            st.info("✅ Strong Match")
        elif similarity >= 50:
            st.warning("⚠️ Moderate Match")
        else:
            st.error("❌ Low Match")
