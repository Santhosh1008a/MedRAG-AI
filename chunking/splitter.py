from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter configured for medical documents.
    
    Why chunking is required:
    1. Context Windows: LLMs have a fixed context window limit (e.g., 4k tokens for Phi-3 Mini). 
       We cannot pass an entire 50-page PDF at once.
    2. Precision vs Recall: 
       - Smaller chunks (precision) allow the retriever to find exact answers and reduce distraction/hallucination.
       - Overlap (recall) ensures context is not broken across chunks (e.g., a diagnosis spanning two paragraphs).
       
    The RecursiveCharacterTextSplitter splits by paragraphs, then sentences, then words,
    preserving semantic meaning better than a simple character splitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks.
    Metadata (like filename and page number) from the parent document is preserved in all child chunks.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)
