import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import shutil
import tempfile
import json

from embeddings.embedder import get_embeddings
from vectorstore.faiss_store import FAISSVectorStore
from reranker.cross_encoder import CrossEncoderReranker
from rag.retriever import AdvancedRetriever
from rag.memory import ConversationMemory
from rag.chain import RAGChain

app = FastAPI(title="MedRAG AI Backend", description="Medical Report Q&A Assistant API")

# ==========================================
# Shared RAG Core Initialization
# ==========================================
embeddings_model = get_embeddings()
vectorstore = FAISSVectorStore(embeddings_model)
vectorstore.load_index()

reranker = CrossEncoderReranker()
memory = ConversationMemory()
retriever = AdvancedRetriever(vectorstore, reranker)

# The same exact RAGChain used in Gradio, exposing identical logic
rag_chain = RAGChain(retriever, memory)
# ==========================================

class AskRequest(BaseModel):
    query: str
    stream: bool = True

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "index_loaded": vectorstore.vectorstore is not None}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Uploads PDFs, processes them, and indexes them into FAISS."""
    try:
        temp_dir = tempfile.mkdtemp()
        filepaths = []
        
        # Save uploaded files temporarily
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            filepaths.append(file_path)
            
        # Delegate document processing/indexing to the shared RAGChain
        result = rag_chain.index_documents(filepaths)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: AskRequest):
    """Answers a question based on uploaded documents."""
    if vectorstore.vectorstore is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Please upload documents first.")
        
    if not request.stream:
        response_data = rag_chain.ask(request.query, stream=False)
        context = [{"source": doc.metadata.get('filename'), "page": doc.metadata.get('page'), "content": doc.page_content, "score": float(score)} for doc, score in response_data["context_docs"]]
        
        return {
            "answer": response_data["answer"],
            "context": context,
            "standalone_query": response_data["standalone_query"],
            "time_taken": response_data["time_taken"]
        }

    # Streaming setup
    response_data = rag_chain.ask(request.query, stream=True)
    streamer = response_data["streamer"]
    
    context = [{"source": doc.metadata.get('filename'), "page": doc.metadata.get('page'), "content": doc.page_content, "score": float(score)} for doc, score in response_data["context_docs"]]
    
    def event_stream():
        # First chunk contains metadata
        metadata = {
            "type": "metadata",
            "context": context,
            "standalone_query": response_data["standalone_query"]
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Subsequent chunks contain the streamed text
        full_answer = ""
        for text in streamer:
            full_answer += text
            yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"
            
        # Update memory after stream completes
        rag_chain.memory.add_user_message(request.query)
        rag_chain.memory.add_assistant_message(full_answer.strip())
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/summarize")
def summarize():
    """Generates a summary of all uploaded documents using Map-Reduce."""
    try:
        summary = rag_chain.summarize()
        return {"summary": summary}
    except Exception as e:
         raise HTTPException(status_code=400, detail=str(e))

@app.post("/clear_history")
def clear_history():
    rag_chain.memory.clear()
    return {"message": "Chat history cleared."}

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
