from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from models.llm import GroqLLM
from rag.retriever import AdvancedRetriever
from rag.memory import ConversationMemory
from rag.prompt import QA_PROMPT, REWRITE_PROMPT, MAP_PROMPT, REDUCE_PROMPT
from loaders.pdf_loader import load_multiple_pdfs
from chunking.splitter import split_documents
import time

class RAGChain:
    """
    Main orchestration class for the RAG pipeline.
    Handles history-aware retrieval, generation, map-reduce summarization,
    and document indexing to unify the backend logic for Gradio and FastAPI.
    """
    def __init__(self, retriever: AdvancedRetriever, memory: ConversationMemory):
        # We lazy-load the LLM so the Gradio UI can boot up quickly.
        self.llm_wrapper = None
        self.langchain_llm = None
        
        self.retriever = retriever
        self.memory = memory
        self.indexed_documents: List[Document] = []
        
    def ensure_llm(self):
        """Lazy loader for the LLM."""
        if self.llm_wrapper is None:
            print("Initializing Groq Llama 3.1 8B...")
            self.llm_wrapper = GroqLLM()
            self.langchain_llm = self.llm_wrapper.get_llm()
            print("Groq Llama 3.1 8B initialized.")

    def index_documents(self, filepaths: List[str]) -> Dict[str, Any]:
        """Loads PDFs, chunks them, and stores them in the FAISS vector database."""
        docs = load_multiple_pdfs(filepaths)
        if not docs:
            raise ValueError("Could not extract text from the provided PDFs.")
            
        self.indexed_documents.extend(docs)
        
        chunks = split_documents(docs)
        
        vectorstore = self.retriever.vectorstore
        if vectorstore.vectorstore is None:
            vectorstore.create_index(chunks)
        else:
            vectorstore.vectorstore.add_documents(chunks)
            
        vectorstore.save_index()
        
        return {
            "message": f"Successfully indexed {len(filepaths)} documents into {len(chunks)} chunks.",
            "num_chunks": len(chunks)
        }

    def rewrite_query(self, query: str) -> str:
        """Rewrites the query into a standalone question using chat history."""
        self.ensure_llm()
        history_str = self.memory.get_history_string()
        if history_str == "No previous history.":
            return query # No rewrite needed for the first turn
            
        prompt_text = REWRITE_PROMPT.format(chat_history=history_str, question=query)
        rewritten = self.langchain_llm.invoke(prompt_text)
        # Clean up output if the model rambles
        content = rewritten.content if hasattr(rewritten, 'content') else str(rewritten)
        return content.strip().split("\n")[0]

    def ask(self, query: str, stream: bool = True) -> Dict[str, Any]:
        """
        Main Q&A function.
        1. Rewrite query if there's history.
        2. Retrieve and rerank context.
        3. Generate answer (streaming or synchronous).
        """
        self.ensure_llm()
        start_time = time.time()
        
        # 1. Query Rewrite
        standalone_query = self.rewrite_query(query)
        
        # 2. Retrieval
        docs_with_scores = self.retriever.retrieve(standalone_query)
        context_str = self.retriever.format_docs(docs_with_scores) if docs_with_scores else ""
        
        # 3. Prompt Construction
        prompt_text = QA_PROMPT.format(context=context_str, question=standalone_query)
        
        response_data = {
            "query": query,
            "standalone_query": standalone_query,
            "context_docs": docs_with_scores,
            "time_taken": 0.0
        }
        
        # 4. Generation
        if stream:
            # Note: For streaming, we return the streamer object. The caller needs to iterate over it
            # and update the memory after streaming finishes.
            streamer = self.llm_wrapper.generate_stream(prompt_text)
            response_data["streamer"] = streamer
        else:
            answer = self.langchain_llm.invoke(prompt_text)
            answer_text = answer.content if hasattr(answer, 'content') else str(answer)
            response_data["answer"] = answer_text.strip()
            self.memory.add_user_message(query)
            self.memory.add_assistant_message(response_data["answer"])
            
        response_data["time_taken"] = time.time() - start_time
        return response_data
        
    def summarize(self) -> str:
        """
        Map-Reduce Summarization:
        1. Map: Summarize each chunk individually.
        2. Reduce: Combine chunk summaries into a final structured summary.
        """
        if not self.indexed_documents:
            raise ValueError("No documents indexed to summarize.")
            
        self.ensure_llm()
        
        # 1. Map phase
        intermediate_summaries = []
        for doc in self.indexed_documents:
            map_prompt_text = MAP_PROMPT.format(text=doc.page_content)
            chunk_summary = self.langchain_llm.invoke(map_prompt_text)
            chunk_text = chunk_summary.content if hasattr(chunk_summary, 'content') else str(chunk_summary)
            intermediate_summaries.append(chunk_text.strip())
            
        # 2. Reduce phase
        combined_summaries = "\n\n".join(intermediate_summaries)
        reduce_prompt_text = REDUCE_PROMPT.format(text=combined_summaries)
        
        final_summary = self.langchain_llm.invoke(reduce_prompt_text)
        final_text = final_summary.content if hasattr(final_summary, 'content') else str(final_summary)
        return final_text.strip()
