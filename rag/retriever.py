from typing import List, Tuple
from langchain_core.documents import Document
from vectorstore.faiss_store import FAISSVectorStore
from reranker.cross_encoder import CrossEncoderReranker

class AdvancedRetriever:
    """
    Bridges FAISS vector store and the Cross-Encoder reranker.
    
    Pipeline:
    Query -> FAISS (Top-10) -> CrossEncoder Reranker -> Top-3 -> Return
    """
    def __init__(self, vectorstore: FAISSVectorStore, reranker: CrossEncoderReranker, initial_k: int = 10, final_k: int = 3):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieves top_k documents based on similarity search from FAISS,
        then reranks them using the Cross-Encoder.
        Returns a list of tuples containing (Document, Similarity Score).
        Note: The score here is from the Cross-Encoder.
        """
        # 1. Initial retrieval from FAISS
        initial_docs = self.vectorstore.similarity_search(query, k=self.initial_k)
        
        if not initial_docs:
            return []
            
        # 2. Rerank with Cross-Encoder
        reranked_docs_with_scores = self.reranker.rerank(query, initial_docs, top_k=self.final_k)
        
        return reranked_docs_with_scores
        
    def format_docs(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """Formats the reranked documents into a single string for the context window."""
        formatted = []
        for i, (doc, score) in enumerate(docs_with_scores):
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)
