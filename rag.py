import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
DB_FOLDER = "faiss_db"

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded!")

# Load FAISS index and metadata
print("Loading FAISS index...")
index = faiss.read_index(os.path.join(DB_FOLDER, "index.faiss"))
with open(os.path.join(DB_FOLDER, "metadata.json"), "r") as f:
    metadata = json.load(f)
print(f"✅ FAISS index loaded with {index.ntotal} vectors!\n")

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
    
    # Build context from chunks
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['source']} ---\n"
        context += chunk['text']
        context += "\n"
    
    # Build prompt
    prompt = f"""You are a medical research assistant. Answer the question below using ONLY the provided context from PubMed research papers.

For each piece of information you use, cite the source like this: [Source 1], [Source 2], etc.

If the context does not contain enough information to answer the question, say "I don't have enough information in the provided papers to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    # Call OpenAI API
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

def ask(question):
    """Full RAG pipeline: question → retrieve → generate → answer."""
    print(f"\n🔍 Question: {question}")
    print("-" * 60)
    
    # Step 1: Retrieve relevant chunks
    print("Retrieving relevant chunks...")
    chunks = retrieve_chunks(question, k=3)
    
    print(f"✅ Found {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk['source']} (chunk {chunk['chunk_index']})")
    
    # Step 2: Generate answer
    print("\nGenerating answer with GPT-3.5...")
    answer = generate_answer(question, chunks)
    
    print("\n📄 Answer:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    
    return answer

# Test the full pipeline
if __name__ == "__main__":
    # Test questions
    questions = [
        "What are the challenges of natural language processing in clinical notes?",
        "How do electronic health records affect NLP performance?",
        "What methods are used to evaluate clinical NLP systems?"
    ]
    
    for question in questions:
        ask(question)
        print("\n" + "="*60 + "\n")