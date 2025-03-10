import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,         # Measures if AI-generated response is factual
    answer_relevancy,     # Measures how relevant the generated response is
    context_precision,    # Measures how precise the retrieved documents are
    context_recall        # Measures how many relevant documents were retrieved
)

from backend.models.hyprid_retrival import hybrid_node_retrieval, bert_extract_answer, client

# Example dataset of questions and ground-truth answers
EVALUATION_DATA = [
    {"query": "What are the benefits of Python?", "ground_truth": "Python is easy to learn, versatile, and widely used in data science, web development, and AI."},
    {"query": "Explain Newton's First Law.", "ground_truth": "Newton's First Law states that an object remains at rest or in uniform motion unless acted upon by an external force."},
    {"query": "What is the capital of France?", "ground_truth": "Paris is the capital of France."},
]

# 🔥 Function to Run RAG Evaluation
def evaluate_rag():
    evaluation_results = []

    for data in EVALUATION_DATA:
        query = data["query"]
        ground_truth = data["ground_truth"]

        # Retrieve relevant nodes
        top_nodes = hybrid_node_retrieval(query, alpha=0.6, top_k=3)
        retrieved_docs = [node.text for node, _ in top_nodes]

        # Generate answer using AI model
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Answer the question: {query} using retrieved context: {retrieved_docs}"}
            ],
        )
        generated_answer = response.choices[0].message.content

        # Store for evaluation
        evaluation_results.append({
            "question": query,
            "ground_truth": ground_truth,
            "retrieved_docs": retrieved_docs,
            "generated_answer": generated_answer,
        })

    # Convert to DataFrame
    df = pd.DataFrame(evaluation_results)

    # 🔥 Compute RAG Metrics
    scores = evaluate(
        dataset=df,
        metrics=[
            faithfulness,  # Checks if the response is supported by retrieved docs
            answer_relevancy,  # Checks if the generated answer is relevant to the query
            context_precision  # Measures how precise retrieved docs are
        ]
    )

    print("\n✅ RAG Evaluation Results:")
    print(scores)
    return scores

if __name__ == "__main__":
    evaluate_rag()

