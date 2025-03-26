import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings
from datasets import Dataset

# Load env variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
os.environ["RAGAS_APP_TOKEN"] = os.getenv("RAGAS_APP_TOKEN")

def safe_mean(values):
    """Safely calculate mean handling empty lists and None values"""
    if not values:
        return 0.0
    return float(np.mean([v for v in values if v is not None]))

def evaluate_answers(input_path, metrics_path):
    """Evaluate generated answers using RAGAS"""
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"❌ Failed to load answers file: {e}")
        return

    # Prepare evaluation data
    eval_data = []
    for _, row in df.iterrows():
        if row["Status"] == "Success":
            eval_data.append({
                "question": row["Question"],
                "answer": row["ModelAnswer"],
                "contexts": [row["RelevantChunks"]],
                "ground_truth": row["ExpectedAnswer"]
            })

    if not eval_data:
        print("⚠️ No successful answers to evaluate")
        return

    # Initialize models
    llm = Together(
        model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        temperature=0.7,
        max_tokens=3000,
        top_k=1,
        together_api_key=api_key
    )

    embeddings = TogetherEmbeddings(
        model="togethercomputer/m2-bert-80M-8k-retrieval"
    )

    # Run evaluation
    try:
        print("🧪 Running RAGAS Evaluation...")
        ragas_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
        
        result = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ],
            llm=llm,
            embeddings=embeddings
        )
        
        # Save metrics
        result_df = result.to_pandas()
        result_df.to_excel(metrics_path, index=False)
        
        # Print summary
        print("\n📊 Evaluation Summary:")
        for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']:
            if metric in result:
                values = result[metric]
                if isinstance(values, list):
                    mean_val = safe_mean(values)
                    print(f"{metric.capitalize()}: {mean_val:.2f}")
                else:
                    print(f"{metric.capitalize()}: {values:.2f}")
        
        print(f"✅ Evaluation metrics saved to {metrics_path}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")

if __name__ == "__main__":
    input_path = "backend/tests/evaluation/generated_answers.xlsx"
    metrics_path = "backend/tests/evaluation/evaluation_metrics.xlsx"
    evaluate_answers(input_path, metrics_path)