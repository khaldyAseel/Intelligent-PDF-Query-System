# import os
# import pandas as pd
# import asyncio
# from dotenv import load_dotenv
# from datasets import Dataset
# from tqdm import tqdm

# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# from langchain_together import Together
# from langchain_together.embeddings import TogetherEmbeddings

# from backend.models.routing_agent import route_query_with_book_context, client

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("TOGETHER_API_KEY")

# # Init Together LLM + Embeddings
# llm = Together(
#     model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
#     temperature=0.7,
#     max_tokens=4000,
#     top_k=1,
#     together_api_key=api_key
# )

# embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# # Load Excel
# df = pd.read_excel("backend/tests/evaluation/evaluation_data.xlsx", sheet_name="q&a pt2")

# # Prepare outputs
# scores_list = []

# # Loop one-by-one
# for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating each question"):

#     question = str(row.get("Question", "")).strip()
#     expected_answer = str(row.get("Expected answer", "")).strip()
#     ground_truth = str(row.get("ground truth", "")).strip()

#     if not question or not expected_answer:
#         print(f"⚠️ Skipping row {i}: missing question or expected answer")
#         scores_list.append({
#             "Question": question,
#             "Faithfulness": None,
#             "Answer Relevancy": None,
#             "Context Recall": None,
#             "Context Precision": None
#         })
#         continue

#     # Step 1: If model answer is missing, generate it
#     if not ground_truth or ground_truth.lower() == "nan":
#         try:
#             print(f"🤖 No model answer in row {i}, generating from model...")
#             ground_truth = route_query_with_book_context(client, question)
#         except Exception as e:
#             print(f"❌ Row {i} model answer generation failed: {e}")
#             scores_list.append({
#                 "Question": question,
#                 "Faithfulness": None,
#                 "Answer Relevancy": None,
#                 "Context Recall": None,
#                 "Context Precision": None
#             })
#             continue

#     # Step 2: Get context again (for clarity, you can skip this if already done above)
#     try:
#         context = route_query_with_book_context(client, question)
#         if isinstance(context, str):
#             context = [context]
#         elif not isinstance(context, list):
#             context = [str(context)]

#         if not context or all(c.strip() == "" for c in context):
#             raise ValueError("Empty or invalid context retrieved.")

#     except Exception as e:
#         print(f"❌ Row {i} context retrieval failed: {e}")
#         scores_list.append({
#             "Question": question,
#             "Faithfulness": None,
#             "Answer Relevancy": None,
#             "Context Recall": None,
#             "Context Precision": None
#         })
#         continue

#     # Step 3: Evaluate with RAGAS
#     print(f"\n🧪 Evaluating Question {i}: {question}")
#     print(f"📚 Context: {context}")
#     print(f"✅ Expected: {expected_answer}")
#     print(f"🤖 Model Answer: {ground_truth}")

#     try:
#         result = evaluate(
#             Dataset.from_list([sample]),
#             metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
#             llm=llm,
#             embeddings=embeddings
#         )

#         score = result.scores[0]
#         print(f"✅ Scores: {score}")
#     except Exception as e:
#         error_msg = str(e)

#         # Attempt to extract text from Together ClientResponse if applicable
#         if hasattr(e, "response"):
#             try:
#                 # You might need to do this asynchronously depending on the client
#                 if asyncio.iscoroutinefunction(e.response.text):
#                     error_msg = asyncio.run(e.response.text())
#                 elif callable(e.response.text):
#                     error_msg = e.response.text()
#             except Exception:
#                 pass

#         print(f"❌ Evaluation failed on row {i}: {error_msg}")
#         score = {
#             "faithfulness": None,
#             "answer_relevancy": None,
#             "context_recall": None,
#             "context_precision": None
#         }

#     scores_list.append({
#         "Question": question,
#         "Faithfulness": score.get("faithfulness"),
#         "Answer Relevancy": score.get("answer_relevancy"),
#         "Context Recall": score.get("context_recall"),
#         "Context Precision": score.get("context_precision")
#     })

# # Save results
# results_df = pd.DataFrame(scores_list)
# results_df.to_csv("ragas_debug_results.csv", index=False)
# print("📁 Results saved to 'ragas_debug_results.csv'")


# import os
# import pandas as pd
# import asyncio
# from dotenv import load_dotenv
# from datasets import Dataset
# from tqdm import tqdm

# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# from langchain_together import Together
# from langchain_together.embeddings import TogetherEmbeddings

# from backend.models.routing_agent import route_query_with_book_context, client

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("TOGETHER_API_KEY")

# # Init Together LLM + Embeddings
# llm = Together(
#     model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
#     temperature=0.7,
#     max_tokens=4000,
#     top_k=1,
#     together_api_key=api_key
# )

# embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# # Load Excel
# df = pd.read_excel("backend/tests/evaluation/evaluation_data.xlsx", sheet_name="q&a pt2")

# # Store model answers separately
# model_answers_output = []
# scores_list = []

# for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating each question"):
#     question = str(row.get("Question", "")).strip()
#     expected_answer = str(row.get("Expected answer", "")).strip()

#     if not all([question, expected_answer]):
#         print(f"⚠️ Skipping row {i}: missing required fields")
#         continue

#     try:
#         # Retrieve full LLM response (context + answer in text)
#         full_response = route_query_with_book_context(client, question)

#         # Split out context metadata from the model answer if present
#         if "\n\n📚 Metadata:" in full_response:
#             model_answer, context_metadata = full_response.split("\n\n📚 Metadata:", 1)
#             context = [context_metadata.strip()]
#         else:
#             model_answer = full_response
#             context = [""]  # No context info returned, still evaluate

#         if not model_answer or model_answer.strip().lower() == "nan":
#             raise ValueError("Model answer generation failed or empty.")

#         ground_truth = model_answer.strip()

#     except Exception as e:
#         print(f"❌ Row {i} model answer generation failed: {e}")
#         scores_list.append({
#             "Question": question,
#             "Faithfulness": None,
#             "Answer Relevancy": None,
#             "Context Recall": None,
#             "Context Precision": None
#         })
#         model_answers_output.append({
#             "Question": question,
#             "Model Answer": None
#         })
#         continue

#     print(f"\n🧪 Evaluating Question {i}: {question}")
#     print(f"📚 Context: {context}")
#     print(f"✅ Expected: {expected_answer}")
#     print(f"🤖 Model Answer: {ground_truth}")

#     model_answers_output.append({
#         "Question": question,
#         "Model Answer": ground_truth
#     })

#     sample = {
#         "question": question,
#         "contexts": context,
#         "answer": ground_truth,
#         "ground_truth": expected_answer,
#     }

#     try:
#         result = evaluate(
#             Dataset.from_list([sample]),
#             metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
#             llm=llm,
#             embeddings=embeddings
#         )
#         score = result.scores[0]
#         print(f"✅ Scores: {score}")
#     except Exception as e:
#         error_msg = str(e)
#         if hasattr(e, "response"):
#             try:
#                 if asyncio.iscoroutinefunction(e.response.text):
#                     error_msg = asyncio.run(e.response.text())
#                 elif callable(e.response.text):
#                     error_msg = e.response.text()
#             except Exception:
#                 pass
#         print(f"❌ Evaluation failed on row {i}: {error_msg}")
#         score = {
#             "faithfulness": None,
#             "answer_relevancy": None,
#             "context_recall": None,
#             "context_precision": None
#         }

#     scores_list.append({
#         "Question": question,
#         "Faithfulness": score.get("faithfulness"),
#         "Answer Relevancy": score.get("answer_relevancy"),
#         "Context Recall": score.get("context_recall"),
#         "Context Precision": score.get("context_precision")
#     })

# # Save results
# pd.DataFrame(scores_list).to_csv("ragas_debug_results.csv", index=False)
# pd.DataFrame(model_answers_output).to_csv("model_answers.csv", index=False)
# print("📁 Results saved to 'ragas_debug_results.csv'")
# print("📁 Model answers saved to 'model_answers.csv'")

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings
from backend.models.routing_agent import route_query_with_book_context, client
from dotenv import load_dotenv
import nltk

# Load env variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
os.environ["RAGAS_APP_TOKEN"] = os.getenv("RAGAS_APP_TOKEN")

# Configuration
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
MAX_TOKENS = 3000  # Reduced from 4000 to avoid token limits

def download_nltk_data():
    """Handle NLTK data download with retries"""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            nltk.download('punkt', quiet=True)
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"⚠️ Failed to download NLTK data after {max_attempts} attempts: {e}")
            time.sleep(2)

# Initialize NLTK
download_nltk_data()

# Initialize LLM
llm = Together(
    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
    temperature=0.7,
    max_tokens=MAX_TOKENS,
    top_k=1,
    together_api_key=api_key
)

embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval"
)

def handle_rate_limiting():
    """Handle rate limiting with exponential backoff"""
    delay = RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        print(f"⚠️ Rate limited, waiting {delay} seconds (attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(delay)
        delay *= 2  # Exponential backoff
    raise Exception("Max retries exceeded for rate limiting")

def safe_mean(values):
    """Safely calculate mean handling empty lists and None values"""
    if not values:
        return 0.0
    return float(np.mean([v for v in values if v is not None]))

def process_evaluation():
    # Load data
    input_path = "backend/tests/evaluation/evaluation_data.xlsx"
    try:
        df = pd.read_excel(input_path, sheet_name="q&a pt2")
    except Exception as e:
        print(f"❌ Failed to load Excel file: {e}")
        return

    # Evaluation data
    eval_data = []
    output_rows = []
    
    # Process questions with rate limit handling
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = str(row["Question"]).strip()
        expected_answer = str(row["ExpectedAnswer"]).strip()
        relevant_chunks = [str(row["RelevantChunks"]).strip()] if pd.notna(row["RelevantChunks"]) else [""]
        
        for attempt in range(MAX_RETRIES):
            try:
                # Get model response
                full_response = route_query_with_book_context(client, question)
                
                if "\n\n📚 Metadata:" in full_response:
                    model_answer, _ = full_response.split("\n\n📚 Metadata:", 1)
                    model_answer = model_answer.strip()
                else:
                    model_answer = full_response.strip()
                
                if not model_answer:
                    model_answer = "[No answer generated]"
                
                # Store results
                eval_data.append({
                    "question": question,
                    "answer": model_answer,
                    "contexts": relevant_chunks,
                    "ground_truth": expected_answer
                })
                
                output_rows.append({
                    "Question": question,
                    "ExpectedAnswer": expected_answer,
                    "ModelAnswer": model_answer,
                    "Status": "Success"
                })
                break
                
            except Exception as e:
                if "400 Bad Request" in str(e) or "rate limit" in str(e).lower():
                    if attempt == MAX_RETRIES - 1:
                        output_rows.append({
                            "Question": question,
                            "ExpectedAnswer": expected_answer,
                            "ModelAnswer": f"Error: Rate limited after {MAX_RETRIES} attempts",
                            "Status": "Failed"
                        })
                        break
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    output_rows.append({
                        "Question": question,
                        "ExpectedAnswer": expected_answer,
                        "ModelAnswer": f"Error: {str(e)}",
                        "Status": "Failed"
                    })
                    break
    
    # Save model answers
    output_path = input_path.replace(".xlsx", "_results.xlsx")
    try:
        pd.DataFrame(output_rows).to_excel(output_path, index=False)
        print(f"✅ Saved results to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # Run RAGAS evaluation if we have successful samples
    if eval_data:
        try:
            from datasets import Dataset
            ragas_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
            
            print("🧪 Running RAGAS Evaluation...")
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
            
            # Convert results to DataFrame
            result_df = result.to_pandas()
            metrics_path = input_path.replace(".xlsx", "_metrics.xlsx")
            result_df.to_excel(metrics_path, index=False)
            
            # Calculate and print metrics safely
            print("\n📊 Evaluation Summary:")
            for metric in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']:
                if metric in result:
                    values = result[metric]
                    if isinstance(values, list):
                        mean_val = safe_mean(values)
                        print(f"{metric.capitalize()}: {mean_val:.2f}")
                    else:
                        print(f"{metric.capitalize()}: {values:.2f}")
            
        except Exception as e:
            print(f"❌ RAGAS evaluation failed: {str(e)}")
            # Save partial results if available
            if 'result_df' in locals():
                result_df.to_excel(metrics_path, index=False)

if __name__ == "__main__":
    process_evaluation()