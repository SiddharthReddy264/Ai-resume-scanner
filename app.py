import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def calculate_similarity(resume_text, job_description):
    corpus = [resume_text, job_description]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

st.title("🧠 AI Resume Screener")
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Paste Job Description Here")

if resume_file and job_description:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(resume_file)
        cleaned_resume = preprocess_text(resume_text)
        cleaned_jd = preprocess_text(job_description)
        similarity_score = calculate_similarity(cleaned_resume, cleaned_jd)
        st.success(f"Resume Match Score: {round(similarity_score * 100, 2)}%")
        if similarity_score > 0.75:
            st.info("✅ Strong Match!")
        elif similarity_score > 0.5:
            st.warning("⚠️ Moderate Match.")
        else:
            st.error("❌ Low Match.")
