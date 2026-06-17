from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embeddings(model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu") -> HuggingFaceBgeEmbeddings:
    """
    Initializes and returns the BGE embeddings model.
    
    Explanation:
    - Embeddings: Dense vector representations of text where mathematically similar vectors represent semantically similar text.
    - Vector representations: Text mapped to high-dimensional floating-point arrays.
    - Semantic similarity: The ability to find answers not by exact keyword match, but by meaning. 
      (e.g., "elevated glucose" vs "high blood sugar").
    
    We use BAAI/bge-base-en-v1.5 as it provides highly competitive retrieval performance for RAG.
    """
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True} # Normalizing helps with cosine similarity
    
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
