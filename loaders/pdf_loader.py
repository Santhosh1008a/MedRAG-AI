import os
from typing import List, Optional
import sys
import types

# Workaround: langchain_community.document_loaders.__init__ imports PebbloSafeLoader
# which requires the Unix-only 'pwd' module. Provide a stub on Windows.
if "pwd" not in sys.modules:
    pwd_stub = types.ModuleType("pwd")
    pwd_stub.getpwuid = lambda uid: types.SimpleNamespace(pw_name="user")  # type: ignore
    sys.modules["pwd"] = pwd_stub

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(filepath: str) -> List[Document]:
    """
    Loads a PDF file and returns a list of LangChain Document objects.
    
    Handles:
    - Missing files
    - Corrupt PDFs
    - Empty documents
    
    Preserves metadata like page number and filename.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
        
    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        
        if not docs:
            print(f"Warning: The document at {filepath} appears to be empty.")
            return []
            
        # Ensure filename is in metadata (PyPDFLoader adds 'source' and 'page' by default)
        filename = os.path.basename(filepath)
        for doc in docs:
            doc.metadata["filename"] = filename
            
        return docs
        
    except Exception as e:
        print(f"Error: Failed to load PDF at {filepath}. It might be corrupt. Details: {e}")
        return []

def load_multiple_pdfs(filepaths: List[str]) -> List[Document]:
    """
    Loads multiple PDF files and aggregates their documents.
    """
    all_docs = []
    for fp in filepaths:
        docs = load_pdf(fp)
        all_docs.extend(docs)
    return all_docs
