import os
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import faiss
from pubmed import get_papers_for_question

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
st.markdown("Ask any medical question and get grounded answers from **live PubMed research papers**.")
st.divider()

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = load_model()
client = load_client()

def embed_and_retrieve(question, papers, k=3):
    """Embed papers and retrieve most relevant chunks."""
    if not papers:
        return []
    
    # Use abstracts as chunks
    texts = [p["abstract"] for p in papers]
    
    # Embed all abstracts
    embeddings = model.encode(texts).astype("float32")
    
    # Build temporary FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Embed question and search
    question_embedding = model.encode([question]).astype("float32")
    k = min(k, len(papers))
    distances, indices = index.search(question_embedding, k=k)
    
    return [papers[idx] for idx in indices[0]]

def generate_answer(question, chunks):
    """Generate cited answer using GPT."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['title']} ---\n"
        context += chunk['abstract'] + "\n"

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

# Sidebar
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This app uses **Live RAG** to answer medical questions by searching PubMed in real time.
    
    **How it works:**
    1. Your question is sent to PubMed API
    2. Top 10 most relevant papers are retrieved
    3. Abstracts are embedded with SentenceTransformers
    4. FAISS finds the most relevant chunks
    5. GPT-3.5 generates a cited answer
    
    **Data Source:**
    - 📡 Live PubMed Central
    - 35+ million papers
    - Always up to date
    
    **Tech Stack:**
    - SentenceTransformers
    - FAISS
    - OpenAI GPT-3.5
    - PubMed Entrez API
    - Streamlit
    """)
    st.divider()
    st.markdown("📡 **Data:** Live PubMed API")
    st.markdown("📄 **Papers:** 35+ million available")

# Main interface
question = st.text_input(
    "💬 Ask a medical question:",
    placeholder="e.g. What are the latest treatments for Type 2 diabetes?"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

if ask_button and question:
    
    # Step 1 - Search PubMed
    with st.status("🔍 Searching PubMed for relevant papers...", expanded=True) as status:
        st.write("Querying PubMed API...")
        papers = get_papers_for_question(question, max_results=10)
        
        if not papers:
            st.error("No papers found for this question. Try rephrasing.")
            st.stop()
        
        st.write(f"✅ Found {len(papers)} relevant papers")
        
        # Step 2 - Retrieve most relevant
        st.write("Finding most relevant abstracts...")
        relevant_chunks = embed_and_retrieve(question, papers, k=3)
        st.write(f"✅ Selected top {len(relevant_chunks)} abstracts")
        
        # Step 3 - Generate answer
        st.write("Generating answer with GPT-3.5...")
        answer = generate_answer(question, relevant_chunks)
        st.write("✅ Answer generated!")
        
        status.update(label="✅ Done!", state="complete")
    
    # Display answer
    st.subheader("📄 Answer")
    st.markdown(answer)
    
    # Display sources
    st.divider()
    st.subheader("📚 Sources Used")
    for i, chunk in enumerate(relevant_chunks):
        with st.expander(f"Source {i+1} — {chunk['title'][:80]}..."):
            st.markdown(f"**PMID:** {chunk['pmid']}")
            st.markdown(f"**PubMed Link:** https://pubmed.ncbi.nlm.nih.gov/{chunk['pmid']}/")
            st.markdown(f"**Abstract:** {chunk['abstract'][:500]}...")

elif ask_button and not question:
    st.warning("Please enter a question first!")
