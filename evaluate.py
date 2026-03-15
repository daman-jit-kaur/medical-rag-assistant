import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model and FAISS index
print("Loading model and index...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_db/index.faiss")
with open("faiss_db/metadata.json", "r") as f:
    metadata = json.load(f)
print("✅ Ready!\n")

# Test questions with ground truth answers
test_dataset = [
    {
        "question": "What are the challenges of NLP in clinical notes?",
        "ground_truth": "Challenges include semantic and syntactic variation across EHR systems, heterogeneity of language across institutions, and difficulty generalizing NLP models across different medical specialties."
    },
    {
        "question": "How do EHR systems affect NLP model performance?",
        "ground_truth": "EHR systems introduce contextual variation that creates semantic and syntactic differences in clinical notes, hampering the portability and generalizability of NLP models across institutions."
    },
    {
        "question": "What methods are used to evaluate clinical NLP systems?",
        "ground_truth": "Methods include measuring semantic and syntactic similarity, analyzing interoperability levels, comparing medical concept coverage, and examining clinical documentation patterns."
    },
    {
        "question": "What is the impact of EHR migration on clinical documentation?",
        "ground_truth": "EHR migration causes contextual variation in clinical notes, affecting how providers document information and impacting the performance of NLP models trained on previous EHR data."
    },
    {
        "question": "How can NLP models be made more generalizable across institutions?",
        "ground_truth": "By building interoperable clinical NLP systems that account for syntactic, semantic, and pragmatic interoperability levels to reconcile heterogeneity across EHRs and institutions."
    }
]

def retrieve_chunks(question, k=3):
    """Retrieve top k relevant chunks for a question."""
    question_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, k=k)
    return [metadata[idx] for idx in indices[0]]

def generate_answer(question, chunks):
    """Generate answer using GPT with retrieved context."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['source']} ---\n"
        context += chunk['text'] + "\n"

    prompt = f"""You are a medical research assistant. Answer the question using ONLY the provided context.
Cite sources like [Source 1], [Source 2].
If context is insufficient, say "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical research assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

def evaluate_faithfulness(answer, chunks):
    """Check if answer is grounded in retrieved chunks using GPT."""
    context = " ".join([c['text'] for c in chunks])
    
    prompt = f"""Rate how faithful this answer is to the provided context.
Score from 0.0 to 1.0 where:
1.0 = answer is completely grounded in context
0.5 = answer is partially grounded
0.0 = answer contains information not in context

Context: {context[:2000]}
Answer: {answer}

Respond with ONLY a number between 0.0 and 1.0"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5

def evaluate_answer_relevance(question, answer):
    """Check if answer actually addresses the question."""
    prompt = f"""Rate how relevant this answer is to the question.
Score from 0.0 to 1.0 where:
1.0 = answer directly and completely addresses the question
0.5 = answer partially addresses the question
0.0 = answer does not address the question

Question: {question}
Answer: {answer}

Respond with ONLY a number between 0.0 and 1.0"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5

def evaluate_context_precision(question, chunks, ground_truth):
    """Check if retrieved chunks are relevant to the question."""
    prompt = f"""Rate how relevant these retrieved chunks are for answering the question.
Score from 0.0 to 1.0 where:
1.0 = chunks are highly relevant and contain the answer
0.5 = chunks are somewhat relevant
0.0 = chunks are not relevant

Question: {question}
Ground Truth Answer: {ground_truth}
Retrieved chunks: {' '.join([c['text'][:300] for c in chunks])}

Respond with ONLY a number between 0.0 and 1.0"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5

def run_evaluation():
    """Run full RAGAS-style evaluation on all test questions."""
    print("=" * 60)
    print("RAGAS EVALUATION — Medical Literature RAG Assistant")
    print("=" * 60)
    
    results = []
    
    for i, item in enumerate(test_dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"\n📝 Question {i+1}: {question}")
        print("-" * 60)
        
        # Retrieve and generate
        chunks = retrieve_chunks(question, k=3)
        answer = generate_answer(question, chunks)
        
        print(f"Answer: {answer[:200]}...")
        
        # Evaluate
        print("Evaluating...")
        faithfulness = evaluate_faithfulness(answer, chunks)
        relevance = evaluate_answer_relevance(question, answer)
        precision = evaluate_context_precision(question, chunks, ground_truth)
        
        print(f"  Faithfulness:      {faithfulness:.2f}")
        print(f"  Answer Relevance:  {relevance:.2f}")
        print(f"  Context Precision: {precision:.2f}")
        
        results.append({
            "question": question,
            "answer": answer,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision
        })
    
    # Calculate averages
    avg_faithfulness = np.mean([r["faithfulness"] for r in results])
    avg_relevance = np.mean([r["answer_relevance"] for r in results])
    avg_precision = np.mean([r["context_precision"] for r in results])
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Average Faithfulness:      {avg_faithfulness:.2f}")
    print(f"  Average Answer Relevance:  {avg_relevance:.2f}")
    print(f"  Average Context Precision: {avg_precision:.2f}")
    print(f"  Overall Score:             {np.mean([avg_faithfulness, avg_relevance, avg_precision]):.2f}")
    print("=" * 60)
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevance": avg_relevance,
                "avg_context_precision": avg_precision,
                "overall_score": np.mean([avg_faithfulness, avg_relevance, avg_precision])
            }
        }, f, indent=2)
    
    print("\n✅ Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()
