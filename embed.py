import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Folders
EXTRACTED_FOLDER = "extracted_texts"
DB_FOLDER = "faiss_db"

# Create DB folder if it doesn't exist
os.makedirs(DB_FOLDER, exist_ok=True)

# Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded!\n")

def chunk_text(text, chunk_size=500):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_all_texts():
    txt_files = [f for f in os.listdir(EXTRACTED_FOLDER) if f.endswith(".txt")]
    
    if not txt_files:
        print("No text files found!")
        return
    
    print(f"Found {len(txt_files)} text files. Starting chunking and embedding...\n")
    
    all_embeddings = []
    all_metadata = []
    
    for txt_file in txt_files:
        txt_path = os.path.join(EXTRACTED_FOLDER, txt_file)
        print(f"Processing: {txt_file}")
        
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")
        
        print("  Generating embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        print(f"  ✅ Embeddings generated")
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            all_embeddings.append(embedding)
            all_metadata.append({
                "source": txt_file,
                "chunk_index": i,
                "text": chunk
            })
        
        print(f"  ✅ {len(chunks)} chunks processed\n")
    
    # Convert to numpy array
    embeddings_matrix = np.array(all_embeddings).astype("float32")
    
    # Create FAISS index
    print("Building FAISS index...")
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)
    print(f"✅ FAISS index built with {index.ntotal} vectors!")
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(DB_FOLDER, "index.faiss"))
    
    # Save metadata
    with open(os.path.join(DB_FOLDER, "metadata.json"), "w") as f:
        json.dump(all_metadata, f)
    
    print(f"\n✅ Embedding complete!")
    print(f"Total chunks stored: {len(all_metadata)}")

def test_query(question):
    print(f"\n🔍 Testing query: '{question}'")
    
    # Load FAISS index and metadata
    index = faiss.read_index(os.path.join(DB_FOLDER, "index.faiss"))
    with open(os.path.join(DB_FOLDER, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Embed the question
    question_embedding = model.encode([question]).astype("float32")
    
    # Search for top 3 results
    distances, indices = index.search(question_embedding, k=3)
    
    print("\n📄 Top 3 relevant chunks:\n")
    for i, idx in enumerate(indices[0]):
        chunk_data = metadata[idx]
        print(f"--- Result {i+1} ---")
        print(f"Source: {chunk_data['source']}")
        print(f"Text: {chunk_data['text'][:300]}...")
        print()

if __name__ == "__main__":
    process_all_texts()
    test_query("What are the challenges of natural language processing in clinical notes?")
