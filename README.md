---
title: MedRAG AI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
---
# 🏥 MedRAG AI

MedRAG AI is a production-grade, privacy-preserving, local Retrieval-Augmented Generation (RAG) system built to act as a medical report Q&A assistant.

**Important Disclaimer:** *This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. The "Confidence" metrics displayed reflect vector retrieval similarity and not true model medical confidence.*

---

## 🏗 Architecture

```text
PDF Document
     ↓
PyPDFLoader (Error handling & metadata preservation)
     ↓
Chunking (RecursiveCharacterTextSplitter: 1000 size, 200 overlap)
     ↓
BGE Embeddings (BAAI/bge-base-en-v1.5)
     ↓
FAISS Vector Database
     ↓
Retriever (Top-10)
     ↓
CrossEncoder Reranker (ms-marco-MiniLM-L-6-v2) → Top-3 Chunks
     ↓
Prompt Template (Strict adherence to context, no hallucination)
     ↓
Phi-3 Mini (3.8B, 4-bit Quantization via bitsandbytes)
     ↓
Answer + Citations
```

---

## 🚀 Features

- **100% Local and Private:** No data is sent to OpenAI or any cloud provider. Everything runs locally.
- **Hardware Efficient:** Designed to run on consumer hardware (e.g., NVIDIA RTX 3050 4GB/8GB) via 4-bit quantization.
- **Explainability Panel:** View the exact chunks of text retrieved, their source document and page number, and their similarity scores.
- **History-Aware Retrieval:** Remembers previous chat context and rewrites follow-up questions to fetch accurate chunks.
- **Map-Reduce Summarization:** Can ingest long reports and provide structured summaries (Executive Summary, Findings, Abnormalities).
- **Reranking:** Implements a Cross-Encoder to drastically improve the relevancy of retrieved documents before they reach the LLM.

---

## 🛠 Tech Stack

- **Python:** Primary language
- **LangChain:** Orchestration and Prompting
- **FAISS:** Vector Database
- **Hugging Face / bitsandbytes:** Local LLM quantization
- **Sentence Transformers:** Cross-Encoder Reranking
- **Gradio:** Frontend UI
- **FastAPI:** Backend REST API
- **RAGAS:** Evaluation framework

---

## 🧠 Why Phi-3 Mini?

Phi-3 Mini (3.8B) was chosen because it punches far above its weight class in reasoning capabilities. 
It outperforms alternatives like Gemma 2B (weaker reasoning) and TinyLlama (high hallucination rates), while avoiding the complex encoder-decoder architecture of FLAN-T5 Base. 

| Model | Size | Reasoning Quality | Memory Requirement (4-bit) |
|---|---|---|---|
| Phi-3 Mini | 3.8B | High | ~2.5 GB VRAM |
| Gemma 2B | 2.5B | Medium | ~2.0 GB VRAM |
| TinyLlama | 1.1B | Low (High Hallucination) | ~1.5 GB VRAM |
| FLAN-T5 Base| 250M | Low (Rigid structure) | ~1.0 GB VRAM |

---

## 📦 Installation & Usage

1. **Clone and Setup Environment:**
   ```bash
   git clone <repo-url>
   cd medical-rag-assistant
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Backend:**
   ```bash
   python app/fastapi_app.py
   ```

3. **Start the Gradio Frontend:**
   ```bash
   python app/gradio_app.py
   ```
   Open `http://localhost:7860` in your browser.

---

## 📂 Folder Structure

```
medical-rag-assistant/
├── app/
│   ├── gradio_app.py       # Frontend interface
│   └── fastapi_app.py      # Backend API
├── loaders/
│   └── pdf_loader.py       # PyPDF loading and metadata handling
├── chunking/
│   └── splitter.py         # Text chunking logic
├── embeddings/
│   └── embedder.py         # BGE Embeddings initialization
├── vectorstore/
│   └── faiss_store.py      # FAISS indexing and retrieval
├── reranker/
│   └── cross_encoder.py    # Cross-encoder reranking
├── models/
│   └── llm.py              # Phi-3 Mini wrapper with 4-bit quantization
├── rag/
│   ├── retriever.py        # Advanced retrieval pipeline
│   ├── chain.py            # Main RAG orchestration
│   ├── memory.py           # Chat history management
│   └── prompt.py           # System prompts
├── evaluation/
│   └── ragas_eval.py       # Local RAGAS evaluation script
├── requirements.txt
└── README.md
```

---

## 📄 Resume Bullets

- **Architected and Deployed a Private Generative AI Assistant:** Built a local Medical Retrieval-Augmented Generation (RAG) system using **LangChain**, **FastAPI**, and **Gradio**, ensuring zero-data-leakage for sensitive medical PDF documents.
- **Optimized LLM Inference for Consumer Hardware:** Integrated **Hugging Face** transformers and **bitsandbytes** 4-bit quantization to run Microsoft Phi-3 Mini (3.8B parameters) efficiently on an NVIDIA RTX 3050 GPU without compromising reasoning quality.
- **Enhanced Semantic Search Quality:** Engineered an advanced retrieval pipeline combining **FAISS** vector database, **BGE embeddings**, and a **CrossEncoder reranker** to filter and surface the most highly relevant medical contexts, reducing hallucinations.
- **Implemented Advanced Prompt Engineering:** Designed history-aware query rewriting and map-reduce summarization pipelines to overcome context window limitations and support multi-turn conversational interactions.
- **Established AI Evaluation Metrics:** Evaluated system performance locally using the **RAGAS** framework, optimizing for Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

---

## 🎙 Interview Preparation Q&A

**Q: What is RAG?**
A: Retrieval-Augmented Generation. Instead of relying solely on the LLM's internal memory (which might be outdated or hallucinate), we first *retrieve* relevant factual documents based on the user's query, append those documents to the prompt, and ask the LLM to *generate* an answer based exclusively on that retrieved context.

**Q: Why not fine-tune?**
A: Fine-tuning is computationally expensive and difficult to update. If a user uploads a new medical report today, a fine-tuned model would need to be retrained to know about it. RAG provides instant knowledge updates, perfect attribution (citations), and drastically reduces hallucinations by forcing the model to read rather than remember.

**Q: Why FAISS?**
A: FAISS (Facebook AI Similarity Search) is an incredibly fast, efficient library for searching dense vectors in local memory, making it ideal for systems without the need for a complex distributed cloud database.

**Q: Why embeddings?**
A: Embeddings translate human text into high-dimensional math (vectors). This allows the computer to measure semantic similarity.

**Q: What is semantic search?**
A: Searching by meaning rather than exact keyword match. For example, "my head hurts" will match with "patient presents with severe migraine" because their embedding vectors are mathematically close.

**Q: What is reranking and what are cross-encoders?**
A: Standard embedding search (Bi-encoder) is fast but slightly imprecise because the query and document are embedded separately. A Cross-Encoder feeds both the query and document into the transformer together, allowing deep self-attention between the words. It is slow but highly accurate. Reranking uses the fast Bi-encoder to get the top 10 results, and the slow Cross-encoder to score and filter them down to the perfect top 3.

**Q: What is hallucination?**
A: When an LLM generates text that is grammatically correct and sounds plausible but is factually incorrect or unsupported by the provided context.

**Q: Why History-Aware retrieval?**
A: If a user asks "What is my glucose level?" and then follows up with "Is that high?", querying the database for "Is that high?" will fail. History-aware retrieval uses an LLM to rewrite the second query into "Is the glucose level high?" before searching.

**Q: Why Map-Reduce summarization?**
A: LLMs have strict context window limits (e.g., 4k tokens). A 50-page PDF will not fit. Map-Reduce solves this by summarizing chunk 1, chunk 2, etc. (Map), and then combining those mini-summaries into a final summary (Reduce).

**Q: What are Faithfulness, Context Precision, and Answer Relevancy?**
A: 
- *Faithfulness:* Did the model invent anything? (Score drops if answer contains info not in context).
- *Context Precision:* Did the retriever find the exact right paragraphs and put them at the top of the list?
- *Answer Relevancy:* Did the model actually answer the user's question, or did it go off on a tangent?

---

## 🔮 Future Improvements (V2)

- **Multimodal Medical Copilot:** Integrate with ChestViT. A user can upload a Chest X-ray, pass it through a Vision Transformer to generate an Attention Heatmap and Findings, and pass those findings directly into MedRAG to allow conversational Q&A about the image.

## 📸 Example Usage & Screenshots

### Architecture Flow

1. User uploads a PDF (e.g., `blood_test_results.pdf`).
2. The PyPDFLoader extracts the text, preserving page numbers.
3. The RecursiveCharacterTextSplitter cuts the document into 1000-character chunks with 200-character overlap.
4. BGE Embeddings convert these chunks into vector representations, storing them in FAISS.
5. User asks: *"Is diabetes indicated?"*
6. The Retriever pulls the top 10 chunks from FAISS based on semantic similarity.
7. The CrossEncoder reranks those 10 chunks to find the absolute top 3 most relevant paragraphs.
8. The top 3 chunks are injected into the Prompt Template.
9. Phi-3 Mini generates a grounded answer, citing the source document and page number.

### Query Rewriting in Action

**User:** What does the report say about my glucose?
*(System retrieves chunks about glucose and answers).*
**User:** Is that bad?
*(System rewrites query behind the scenes to: "Is the glucose level mentioned in the report considered bad?")*
*(System retrieves chunks again based on the rewritten query).*
**System Answer:** Based on the report on page 2, the glucose level is 110 mg/dL, which is marked as elevated...

### Example Outputs
* **Summarization:** "Executive Summary: The patient underwent a routine blood test. Key Findings: Cholesterol is within normal limits. Abnormalities: Fasting glucose is slightly elevated at 105 mg/dL. Recommendations: Consult with primary care physician regarding dietary adjustments."
* **Explainability:** "Similarity Score: 0.8421. Source: lab_results.pdf, Page 2. Content: Fasting Glucose: 105 mg/dL (High)..."
