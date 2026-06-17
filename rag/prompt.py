from langchain_core.prompts import PromptTemplate

# Prompt to prevent medical hallucinations
QA_PROMPT_TEMPLATE = """You are MedRAG AI, a highly accurate and professional medical report assistant.
Use the following context to answer the user's question. 

CRITICAL INSTRUCTIONS:
1. Ground your answer strictly in the provided context.
2. If the answer is not contained in the context, respond EXACTLY with: "I could not find sufficient information in the uploaded documents."
3. Never invent, guess, or hallucinate medical facts, diagnoses, or numbers.
4. When you provide an answer, try to cite the source file or page if available in the context.

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Prompt for history-aware query rewriting
REWRITE_PROMPT_TEMPLATE = """You are an AI assistant. Your task is to rephrase the user's follow-up question into a standalone question, using the chat history for context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

REWRITE_PROMPT = PromptTemplate(
    template=REWRITE_PROMPT_TEMPLATE,
    input_variables=["chat_history", "question"]
)

# Prompt for Map-Reduce Summarization: Chunk-level (Map)
MAP_PROMPT_TEMPLATE = """You are a medical assistant. Summarize the following extracted text from a medical document. Focus on key findings, abnormal values, and recommendations.

Text:
{text}

Summary:"""

MAP_PROMPT = PromptTemplate(
    template=MAP_PROMPT_TEMPLATE,
    input_variables=["text"]
)

# Prompt for Map-Reduce Summarization: Final (Reduce)
REDUCE_PROMPT_TEMPLATE = """You are a medical assistant. Combine the following intermediate summaries into a final comprehensive summary of the medical report.
Structure your summary with the following sections:
1. Executive Summary
2. Key Findings
3. Abnormalities
4. Recommendations

Intermediate Summaries:
{text}

Final Comprehensive Summary:"""

REDUCE_PROMPT = PromptTemplate(
    template=REDUCE_PROMPT_TEMPLATE,
    input_variables=["text"]
)
