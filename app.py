import os
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Page config
st.set_page_config(
    page_title="Medical Literature RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Medical Literature RAG Assistant")
st.markdown("Ask any medical question and get grounded answers from PubMed research papers.")
st.divider()

# Load models and index (cached so they only load once)
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_db/index.faiss")
    with open("faiss_db/metadata.json", "r") as f:
        metadata = json.load(f)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return model, index, metadata, client

model, index, metadata, client = load_models()

def retrieve_chunks(question, k=3):
    """Find the top k most relevant chunks for a question."""
    question_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, k=k)
    results = []
    for idx in indices[0]:
        results.append(metadata[idx])
    return results

def generate_answer(question, chunks):
    """Send question + retrieved chunks to GPT and get a cited answer."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['source']} ---\n"
        context += chunk['text']
        context += "\n"

    prompt = f"""You are a medical research assistant. Answer the question below using ONLY the provided context from PubMed research papers.

For each piece of information you use, cite the source like this: [Source 1], [Source 2], etc.

If the context does not contain enough information to answer the question, say "I don't have enough information in the provided papers to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical research assistant that answers questions based only on provided context and always cites sources."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

# Sidebar info
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This app uses **Retrieval-Augmented Generation (RAG)** to answer medical questions from PubMed papers.
    
    **How it works:**
    1. Your question is converted to an embedding
    2. FAISS finds the most relevant paper chunks
    3. GPT-3.5 generates a cited answer
    
    **Tech Stack:**
    - SentenceTransformers
    - FAISS
    - OpenAI GPT-3.5
    - Streamlit
    """)
    st.divider()
    st.markdown(f"📄 **Papers loaded:** 6")
    st.markdown(f"🔢 **Total chunks:** {index.ntotal}")

# Main interface
question = st.text_input(
    "💬 Ask a medical question:",
    placeholder="e.g. What are the challenges of NLP in clinical notes?"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

if ask_button and question:
    with st.spinner("Searching papers and generating answer..."):
        # Retrieve chunks
        chunks = retrieve_chunks(question, k=3)
        
        # Generate answer
        answer = generate_answer(question, chunks)
    
    # Display answer
    st.subheader("📄 Answer")
    st.markdown(answer)
    
    # Display sources
    st.divider()
    st.subheader("📚 Sources Used")
    for i, chunk in enumerate(chunks):
        with st.expander(f"Source {i+1} — {chunk['source']} (chunk {chunk['chunk_index']})"):
            st.markdown(chunk['text'][:500] + "...")

elif ask_button and not question:
    st.warning("Please enter a question first!")