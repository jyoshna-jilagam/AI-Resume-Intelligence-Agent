# 🚀 AI Resume Intelligence Agent

An AI-powered Resume Screening System built using:

- RAG (Retrieval Augmented Generation)
- FAISS Vector Database
- Sentence Transformers
- Groq LLaMA 3.1
- Streamlit UI

---

## 🧠 Problem

Recruiters manually review resumes, which is time-consuming.  
This AI system evaluates resumes against job descriptions using semantic retrieval and LLM reasoning.

---

## ⚙️ Tech Stack

- Streamlit
- FAISS
- Sentence Transformers
- Groq API
- PyMuPDF
- Python

---

## 🏗 Architecture

Resume PDF  
↓  
Text Extraction  
↓  
Chunking  
↓  
Embeddings  
↓  
FAISS Retrieval  
↓  
Groq LLM  
↓  
AI Evaluation Output  

---

## 🚀 Run Locally

pip install -r requirements.txt


streamlit run app.py
