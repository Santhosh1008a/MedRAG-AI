import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class FAISSVectorStore:
    """
    Manages the FAISS vector database for nearest-neighbor search.
    
    Explanation:
    - Vector Databases: Specialized databases optimized for storing and querying high-dimensional vectors.
    - Nearest-neighbor search: A mathematical search method to find vectors mathematically closest 
      (e.g., using cosine similarity or L2 distance) to a query vector.
    """
    def __init__(self, embeddings: Embeddings, index_path: str = "./data/faiss_index"):
        self.embeddings = embeddings
        self.index_path = index_path
        self.vectorstore: Optional[FAISS] = None

    def create_index(self, documents: List[Document]):
        """Creates a new FAISS index from documents."""
        if not documents:
            raise ValueError("No documents provided to create index.")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def save_index(self):
        """Saves the FAISS index to disk."""
        if self.vectorstore is not None:
            os.makedirs(self.index_path, exist_ok=True)
            self.vectorstore.save_local(self.index_path)
        else:
            raise ValueError("No vector store initialized to save.")

    def load_index(self) -> bool:
        """
        Loads the FAISS index from disk. 
        Returns True if successful, False if the index doesn't exist.
        """
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            self.vectorstore = FAISS.load_local(
                self.index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True # Required when loading local FAISS indices
            )
            return True
        return False

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a top-k similarity search.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Please create or load an index first.")
        return self.vectorstore.similarity_search(query, k=k)
        
    def similarity_search_with_score(self, query: str, k: int = 4):
        """
        Performs a top-k similarity search returning documents and their L2 distance scores.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized.")
        return self.vectorstore.similarity_search_with_score(query, k=k)
        
    def get_retriever(self, k: int = 4):
        """Returns the base LangChain retriever for this vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
