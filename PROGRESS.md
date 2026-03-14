# Medical Literature RAG Assistant — Project Progress

## Project Overview
A RAG (Retrieval-Augmented Generation) system that lets users ask natural language questions over PubMed research papers and receive grounded, cited answers.

**Degree:** MS Computer Science
**Timeline:** 1 month (~40–55 hours total)
**Target:** Complete and deployed on Streamlit Cloud

---

## Tech Stack
| Component | Tool |
|---|---|
| PDF Extraction | PyPDF2 |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | OpenAI API (gpt-3.5-turbo) |
| Orchestration | LangChain |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Deployment | Streamlit Cloud |

---

## Project Structure
```
medical-rag-assistant/
├── data/               ← PubMed PDFs go here
├── notebooks/          ← Experimentation
├── .env                ← OpenAI API key (never commit this)
├── .gitignore
├── PROGRESS.md         ← This file
├── extract.py          ← Week 1: PDF text extraction
├── embed.py            ← Week 2: Chunking + embeddings + ChromaDB
├── rag.py              ← Week 3: Full RAG pipeline
├── app.py              ← Week 3: Streamlit UI
└── evaluate.py         ← Week 4: RAGAS evaluation
```

---

## Weekly Plan
| Week | Focus | Hours | Status |
|---|---|---|---|
| Week 1 | Setup + Data + PDF Extraction | 10–12 hrs | 🔄 In Progress |
| Week 2 | Embeddings + ChromaDB | 10–12 hrs | ⬜ Not Started |
| Week 3 | LLM Integration + Streamlit UI | 12–15 hrs | ⬜ Not Started |
| Week 4 | Evaluation + Write-Up + Polish | 10–12 hrs | ⬜ Not Started |

---

## Milestone Checklist

### ✅ Week 1 — Foundation
- [x] Project folder created (`medical-rag-assistant`)
- [x] Virtual environment set up and activated
- [x] All libraries installed (pypdf2, sentence-transformers, chromadb, streamlit, openai, langchain, ragas)
- [x] OpenAI account created + API key saved in `.env`
- [x] 6 PubMed PDFs downloaded into `/data`
- [x] GitHub repo initialized
- [x] `extract.py` written and tested — extracts text from all PDFs

### ✅ Week 2 — Embeddings & Vector Store
- [x] Text chunked into 500-word segments
- [x] SentenceTransformers embeddings generated
- [x] FAISS index storing and retrieving chunks
- [x] Query test working — medical question returns relevant chunks
- [x] OpenAI API connected to retrieval pipeline
- [x] Full RAG pipeline working — question → retrieve → GPT → cited answer

### 🔄 Week 3 — LLM + UI
- [x] OpenAI API connected to retrieval pipeline
- [x] Full pipeline working: question → retrieve → GPT → cited answer
- [x] Streamlit app built (input box, answer display, sources section)
- [ ] App deployed live on Streamlit Cloud (public URL)

### ⬜ Week 4 — Evaluation & Submission
- [ ] RAGAS evaluation running (Faithfulness, Relevance, Context Precision)
- [ ] Two chunking experiments compared (256 vs 512 words)
- [ ] Medical hallucination analysis written
- [ ] Demo video recorded (3 min, YouTube unlisted)
- [ ] GitHub README finalized with architecture + results
- [ ] 2-page project report written and submitted

---

## Session Log

| Date | What Was Done | Next Step |
|---|---|---|
| Mar 9, 2026 | Project planning complete, roadmap reviewed | Begin Week 1 setup |
| Mar 9, 2026 | Day 1 complete — Python, VS Code, Git, GitHub, venv, libraries, OpenAI API key all set up | Day 2 — download papers |
| Mar 9, 2026 | Day 2 complete — 6 PubMed papers downloaded, extract.py written and working, all texts extracted | Day 3 — chunking and embeddings |
| Mar 14, 2026 | Day 3 complete — embed.py working, 59 chunks embedded with FAISS, query test successful. Switched from ChromaDB to FAISS due to Windows DLL issues | Day 4 — connect OpenAI LLM to pipeline |
| Mar 14, 2026 | Day 4 complete — rag.py working, full pipeline question → retrieve → GPT → cited answer | Day 5 — build Streamlit UI |
| Mar 14, 2026 | Day 5 complete — app.py built, Streamlit UI working locally with cited answers and sources section | Day 6 — deploy to Streamlit Cloud |

## How to Use This File
At the start of any new Claude chat, paste this file and say:
**"Here is my project progress file. Let's continue from where I left off."**
Claude will have full context instantly.
