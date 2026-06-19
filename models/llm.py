import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

class GroqLLM:
    """
    Wrapper for Groq API using llama-3.1-8b-instant.
    """
    def __init__(self, model_id: str = "llama-3.1-8b-instant"):
        self.model_id = model_id
        
        print(f"Initializing Groq LLM: {self.model_id}")
        self.llm = ChatGroq(
            model=self.model_id,
            temperature=0.1,     # Low temperature for factual medical RAG to prevent hallucination
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def get_llm(self):
        """Returns the LangChain-compatible LLM."""
        return self.llm

    def generate_stream(self, prompt: str):
        """
        Generates text while natively streaming chunks from ChatGroq.
        Yields plain string chunks to maintain compatibility with the UI.
        """
        for chunk in self.llm.stream(prompt):
            if chunk.content:
                yield chunk.content
