import streamlit as st
import pickle
import re, string
from PyPDF2 import PdfReader
import docx

import gdown
import os
encoder = pickle.load(open("encoder.pkl", "rb"))
def download_from_drive(file_id, output):
    if not os.path.exists(output): 
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# File IDs from Google Drive
clf_id = "1DW0kRFJUSFHqvwT5b9u3x-4SECFuoly5"
tfidf_id = "1owK81t0COpWhw3QkOBHLBt3sHlb08BD4"
encoder_id = "1BF7iEh_cfOGbkGw6NyWdge3yk_MDKoVn"

# Download files
download_from_drive(clf_id, "clf.pkl")
download_from_drive(tfidf_id, "tfidf.pkl")
download_from_drive(encoder_id, "encoder.pkl")

# Load them
import pickle
model = pickle.load(open("clf.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))

# -------------------
# Cleaning function
# -------------------
def clean_text(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    cleanText = cleanText.strip()
    return cleanText.lower()

def pred(resume_text):
    cleaned_text = clean_text(resume_text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()  # convert to dense
    prediction = model.predict(vectorized_text)[0]
    category = encoder.inverse_transform([prediction])[0]  # convert number → label
    return category



# -------------------
# File text extractor
# -------------------
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return None

# -------------------
# Streamlit App UI
# -------------------
st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="centered")
st.title("📄 Resume Classifier")
st.write("Upload your resume or paste text to predict its category.")

# File upload
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

resume_input = ""
if uploaded_file is not None:
    resume_input = extract_text_from_file(uploaded_file)
    if resume_input:
        st.text_area("Extracted Resume Text:", resume_input[:1000] + "..." if len(resume_input) > 1000 else resume_input, height=200)
    else:
        st.error("Unsupported file format or unable to extract text.")

# Or manual input
resume_text = st.text_area("Or paste resume text manually:", "")

if st.button("Classify Resume"):
    if resume_input:
        result = pred(resume_input)
        st.success(f"Prediction from uploaded file: {result}")
    elif resume_text.strip():
        result = pred(resume_text)
        st.success(f"Prediction from pasted text: {result}")
    else:
        st.warning("Please upload a file or paste text before classifying.")
