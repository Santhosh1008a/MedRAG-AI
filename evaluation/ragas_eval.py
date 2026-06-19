import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from models.llm import GroqLLM
from embeddings.embedder import get_embeddings

def evaluate_rag_offline(evaluation_data_path: str):
    """
    Runs RAGAS evaluation locally using Groq Llama 3.1 8B.
    
    Explanation of Metrics:
    - Faithfulness: Measures if the answer can be inferred purely from the provided context (checks for hallucination).
    - Answer Relevancy: Measures how directly the answer addresses the initial question.
    - Context Precision: Measures if the retrieved chunks contain the exact answer, and if they are ranked highly.
    - Context Recall: Measures if the retrieval managed to fetch all necessary context to answer the question.
    """
    print("Loading evaluation dataset...")
    try:
        with open(evaluation_data_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return
        
    dataset = Dataset.from_dict(data)
    
    print("Initializing Judge (Groq)...")
    llm_wrapper = GroqLLM()
    local_judge = llm_wrapper.get_llm()
    
    print("Initializing Local Embeddings (BGE Base)...")
    embeddings = get_embeddings()
    
    print("Running RAGAS evaluation...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    # RAGAS requires explicit setting of the LLM and Embeddings to bypass the OpenAI defaults
    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = local_judge
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    
    print("\n=== Evaluation Results ===")
    print(result)
    
    return result

if __name__ == "__main__":
    # Example usage:
    # evaluation_data.json should be a list of dicts with:
    # "question", "answer", "contexts", and "ground_truths"
    
    eval_file = "./data/evaluation_data.json"
    import os
    if not os.path.exists(eval_file):
        print(f"Please create an evaluation dataset at {eval_file}")
        print("Format: {'question': [...], 'answer': [...], 'contexts': [[...]], 'ground_truths': [[...]]}")
    else:
        evaluate_rag_offline(eval_file)
