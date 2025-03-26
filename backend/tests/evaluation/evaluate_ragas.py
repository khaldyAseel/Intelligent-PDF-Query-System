# import pandas as pd
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, answer_similarity
# from backend.models.routing_agent import route_query_with_book_context, client
# from ragas.schema import QuestionAnswer

# # Load Excel Data
# file_path = r"backend\tests\evaluation\evaluation_data.xlsx"
# sheet_name = "q&a pt2"
# df = pd.read_excel(file_path, sheet_name=sheet_name)

# print("✅ Sheet loaded successfully")

# # ✅ Function to get the actual model answer from your routing agent
# def get_model_answer(question):
#     try:
#         answer = route_query_with_book_context(client, question, threshold=0.4)
#         return answer
#     except Exception as e:
#         print(f"❌ Error getting model answer for question: {question}")
#         print(f"   {e}")
#         return "ERROR"

# # 🔄 Apply model to generate answers and store them in Column E (ground truth)
# print("🚀 Generating answers using the model...")
# df["ground truth"] = df["Question"].apply(get_model_answer)

# # 🧠 Extract evaluation inputs
# questions = df["Question"].tolist()
# expected_answers = df["Expected answer"].tolist()
# generated_answers = df["ground truth"].tolist()
# retrieved_chunks = df["Relevant chunks"].tolist()

# # ✅ Define Ragas evaluation metrics
# metrics = [faithfulness, answer_relevancy, answer_correctness, answer_similarity]

# # 📊 Evaluate the answers
# print("🔍 Evaluating responses with Ragas...")

# qa_pairs = [
#     QuestionAnswer(
#         question=q,
#         answer=a,
#         contexts=[c] if isinstance(c, str) else [],
#         ground_truth=e
#     )
#     for q, a, c, e in zip(questions, generated_answers, retrieved_chunks, expected_answers)
# ]

# evaluation_results = evaluate(qa_pairs, metrics)

# # 📝 Add results to DataFrame
# df["Faithfulness"] = evaluation_results["faithfulness"]
# df["Relevance"] = evaluation_results["answer_relevancy"]
# df["Correctness"] = evaluation_results["answer_correctness"]
# df["Answer Similarity"] = evaluation_results["answer_similarity"]

# # 💾 Save results to new Excel file
# output_path = r"backend\tests\evaluation\evaluation_results.xlsx"
# df.to_excel(output_path, index=False)

# print(f"✅ Evaluation completed! Results saved to {output_path}")

# import os
# import pandas as pd
# from dotenv import load_dotenv  # ✅ Load .env variables
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     answer_correctness,
#     answer_similarity,
# )
# from langchain_community.llms import Together
# from backend.models.routing_agent import route_query_with_book_context, client
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_community.embeddings import TogetherEmbeddings  # ✅ supports embeddings via Together



# # 🔐 Load TOGETHER_API_KEY from .env
# load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# # 🚨 Safety check
# if not TOGETHER_API_KEY:
#     raise EnvironmentError("TOGETHER_API_KEY not found in .env")

# # 🧠 Setup Together LLM
# llm = Together(
#     model="togethercomputer/llama-2-70b-chat",  # You can change model if needed
#     temperature=0.0,
#     max_tokens=512,
#     together_api_key=TOGETHER_API_KEY,
# )

# # 📖 Load evaluation data
# file_path = r"backend\tests\evaluation\evaluation_data.xlsx"
# sheet_name = "q&a pt2"
# df = pd.read_excel(file_path, sheet_name=sheet_name)
# print("✅ Sheet loaded successfully")

# # 🤖 Answer generation
# def get_model_answer(question):
#     try:
#         answer = route_query_with_book_context(client, question, threshold=0.4)
#         if not answer or "ERROR" in str(answer) or isinstance(answer, Exception):
#             raise ValueError("Empty or invalid answer")
#         return answer
#     except Exception as e:
#         print(f"❌ Error getting model answer for question: {question}")
#         print(f"   {e}")
#         return f"ERROR: {str(e)}"

# print("🚀 Generating answers using the model...")
# df["ground truth"] = df["Question"].apply(get_model_answer)

# # 📦 Create examples for Ragas
# examples = []
# for i in range(len(df)):
#     examples.append({
#         "question": df.loc[i, "Question"],
#         "contexts": [df.loc[i, "Relevant chunks"]] if pd.notna(df.loc[i, "Relevant chunks"]) else [""],
#         "answer": df.loc[i, "ground truth"],
#         "ground_truth": df.loc[i, "Expected answer"]
#     })

# # 📊 Define Ragas evaluation metrics
# metrics = [faithfulness, answer_relevancy, answer_correctness, answer_similarity]
# # ✅ Setup embeddings (instead of default OpenAI)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # 📊 Evaluate with Ragas
# print("🔍 Evaluating responses with Ragas...")
# results = evaluate(examples, metrics, llm=llm, embeddings=embeddings, use_default_embedding=False)

# # 📝 Store results
# df["Faithfulness"] = results["faithfulness"]
# df["Relevance"] = results["answer_relevancy"]
# df["Correctness"] = results["answer_correctness"]
# df["Answer Similarity"] = results["answer_similarity"]

# # 💾 Save to Excel
# output_path = r"backend\tests\evaluation\evaluation_results.xlsx"
# df.to_excel(output_path, index=False)

# print(f"✅ Evaluation completed! Results saved to {output_path}")


import os
import pandas as pd
import ast
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

def safe_convert_to_list(text):
    """Safely convert string to list, handling various formats"""
    if pd.isna(text):
        return [""]
    
    text = str(text).strip()
    if not text:
        return [""]
    
    # If it's already a properly formatted list string
    if text.startswith('[') and text.endswith(']'):
        try:
            return ast.literal_eval(text)
        except:
            pass
    
    # If it's a single string, wrap it in a list
    return [text]

# Load Excel file
df = pd.read_excel("backend/tests/evaluation/evaluation_data.xlsx", sheet_name="q&a pt2")

# Prepare evaluation data
eval_data = []
model_answers = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
    try:
        question = str(row["Question"]).strip()
        ground_truth = str(row["Expected answer"]).strip()
        relevant_chunks = safe_convert_to_list(row["Relevant chunks"])
        
        if not question or not ground_truth:
            print(f"⚠️ Skipping row {i}: missing question or expected answer")
            continue

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
            "contexts": relevant_chunks,
            "ground_truth": ground_truth
        })
        
        # Store model answers for reference
        model_answers.append({
            "Question": question,
            "Model Answer": model_answer,
            "Expected Answer": ground_truth,
            "Context Used": relevant_chunks
        })

    except Exception as e:
        print(f"❌ Error processing row {i}: {str(e)}")
        model_answers.append({
            "Question": row.get("Question", ""),
            "Model Answer": f"Error: {str(e)}",
            "Expected Answer": row.get("Expected answer", ""),
            "Context Used": ""
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
pd.DataFrame(model_answers).to_excel("model_answers.xlsx", index=False)

print("\n✅ Evaluation complete. Results saved to evaluation_results.csv and model_answers.xlsx")