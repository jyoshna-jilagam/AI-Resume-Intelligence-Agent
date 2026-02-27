import os
import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------
# 🔐 Load API Key Securely from .env
# ---------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it inside your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------
# Model Configuration
# ---------------------------------------------------
MODEL_NAME = "llama-3.1-8b-instant"

# ---------------------------------------------------
# Streamlit UI Config
# ---------------------------------------------------
st.set_page_config(page_title="AI Resume Intelligence Agent", layout="wide")
st.title("🚀 AI Resume Intelligence Agent")
st.markdown("Powered by RAG + FAISS + Groq LLaMA 3.1")

# ---------------------------------------------------
# Force CPU to Avoid Meta Tensor Error
# ---------------------------------------------------
device = "cpu"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

embedding_model = load_embedding_model()

# ---------------------------------------------------
# Extract Text from PDF
# ---------------------------------------------------
def extract_text(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ---------------------------------------------------
# Chunk Text
# ---------------------------------------------------
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------------------------------------------
# Create FAISS Index
# ---------------------------------------------------
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index

# ---------------------------------------------------
# Retrieve Relevant Chunks
# ---------------------------------------------------
def retrieve_chunks(query, index, chunks, k=3):
    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(np.array(query_vector), k)
    return [chunks[i] for i in indices[0]]

# ---------------------------------------------------
# Analyze Resume using Groq
# ---------------------------------------------------
def analyze_resume(context, job_description):

    prompt = f"""
You are a Senior Technical Recruiter.

Job Description:
{job_description}

Resume Content:
{context}

Evaluate the candidate and provide:

1. Match Score (0-100)
2. Key Strengths
3. Skill Gaps
4. Final Hiring Recommendation
5. Suggested Improvements
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {e}"

# ---------------------------------------------------
# Streamlit Inputs
# ---------------------------------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
job_description = st.text_area("📝 Paste Job Description Here")

# ---------------------------------------------------
# Run Analysis
# ---------------------------------------------------
if uploaded_file and job_description:

    with st.spinner("🔍 Analyzing resume using RAG pipeline..."):

        text = extract_text(uploaded_file)

        if text:
            chunks = chunk_text(text)
            index = create_faiss_index(chunks)
            relevant_chunks = retrieve_chunks(job_description, index, chunks)

            context = " ".join(relevant_chunks)
            result = analyze_resume(context, job_description)

            st.success("✅ Analysis Complete")
            st.subheader("📊 AI Evaluation Result")
            st.write(result)

else:
    st.info("Please upload a resume and enter a job description to begin.")
