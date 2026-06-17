from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class CrossEncoderReranker:
    """
    Reranks documents retrieved from FAISS using a Cross-Encoder.
    
    Explanation:
    - Reranking: Taking a broader set of initially retrieved documents (e.g., top-10 from FAISS)
      and scoring them more precisely to filter down to the absolute best matches (e.g., top-3).
    - Cross-encoders: Unlike Bi-encoders (which embed query and document independently and compare them),
      Cross-encoders pass both the query and document simultaneously through the transformer. This allows
      deep self-attention between the query and document tokens, yielding significantly more accurate relevance scores.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Takes a query and a list of Documents, scores them, and returns the top_k
        documents sorted by relevance score, along with their scores.
        """
        if not documents:
            return []

        # Prepare pairs of (Query, Document Text) for the CrossEncoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Combine documents with their scores and sort descending
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return doc_score_pairs[:top_k]
