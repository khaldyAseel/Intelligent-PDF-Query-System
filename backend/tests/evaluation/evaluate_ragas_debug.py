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
import pandas as pd
from tqdm import tqdm
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings
from backend.models.routing_agent import route_query_with_book_context, client
from dotenv import load_dotenv

# Load env variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
os.environ["RAGAS_APP_TOKEN"] = os.getenv("RAGAS_APP_TOKEN")

# LLM and embedding setup
llm = Together(
    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
    temperature=0.7,
    max_tokens=4000,
    top_k=1,
    together_api_key=api_key,
)
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# Load Excel file
df = pd.read_excel("backend/tests/evaluation/evaluation_data.xlsx", sheet_name="q&a pt2")

# Prepare evaluation data
eval_data = []
model_answers = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
    question = str(row["Question"]).strip()
    ground_truth = str(row["Expected answer"]).strip()
    relevant_chunks = eval(row["Relevant chunks"]) if pd.notna(row["Relevant chunks"]) else [""]
    
    if not question or not ground_truth:
        print(f"⚠️ Skipping row {i}: missing question or expected answer")
        continue

    try:
        # Get model's answer
        full_response = route_query_with_book_context(client, question)
        
        if "\n\n📚 Metadata:" in full_response:
            model_answer, _ = full_response.split("\n\n📚 Metadata:", 1)
            model_answer = model_answer.strip()
        else:
            model_answer = full_response.strip()

        if not model_answer or model_answer.lower() == "nan":
            raise ValueError("Empty model answer")

        # Store for evaluation
        eval_data.append({
            "question": question,
            "answer": model_answer,
            "contexts": relevant_chunks,  # Using relevant chunks from Excel
            "ground_truth": ground_truth
        })
        
        # Store model answers for reference
        model_answers.append({
            "Question": question,
            "Model Answer": model_answer,
            "Expected Answer": ground_truth
        })

    except Exception as e:
        print(f"❌ Error processing row {i} ({question}): {str(e)}")
        model_answers.append({
            "Question": question,
            "Model Answer": f"Error: {str(e)}",
            "Expected Answer": ground_truth
        })
        continue

# Convert to RAGAS dataset
from datasets import Dataset
ragas_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

# Run evaluation
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

# Save results
result_df = result.to_pandas()
result_df.to_csv("evaluation_results.csv", index=False)

# Save model answers
pd.DataFrame(model_answers).to_csv("model_answers.csv", index=False)

# Print summary
print("\n📊 Evaluation Summary:")
print(f"Faithfulness: {result['faithfulness'].mean():.2f}")
print(f"Answer Relevancy: {result['answer_relevancy'].mean():.2f}")
print(f"Context Recall: {result['context_recall'].mean():.2f}")
print(f"Context Precision: {result['context_precision'].mean():.2f}")

print("\n✅ Evaluation complete. Results saved to evaluation_results.csv and model_answers.csv")