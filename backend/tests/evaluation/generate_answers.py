import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_together import Together
from backend.models.routing_agent import route_query_with_book_context, client
import nltk

# Load env variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

# Configuration
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
MAX_TOKENS = 3000

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

def initialize_llm():
    """Initialize the LLM with rate limit handling"""
    return Together(
        model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        top_k=1,
        together_api_key=api_key
    )

def generate_answers(input_path, output_path):
    """Generate answers for questions and save to new file"""
    try:
        df = pd.read_excel(input_path, sheet_name="q&a pt2")
    except Exception as e:
        print(f"❌ Failed to load Excel file: {e}")
        return

    output_data = []
    successful_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
        question = str(row["Question"]).strip()
        expected_answer = str(row["ExpectedAnswer"]).strip()
        relevant_chunks = [str(row["RelevantChunks"]).strip()] if pd.notna(row["RelevantChunks"]) else [""]
        
        if not question:
            continue

        for attempt in range(MAX_RETRIES):
            try:
                full_response = route_query_with_book_context(client, question)
                
                if "\n\n📚 Metadata:" in full_response:
                    model_answer, _ = full_response.split("\n\n📚 Metadata:", 1)
                    model_answer = model_answer.strip()
                else:
                    model_answer = full_response.strip()
                
                if not model_answer:
                    model_answer = "[No answer generated]"
                
                output_data.append({
                    "Question": question,
                    "ExpectedAnswer": expected_answer,
                    "ModelAnswer": model_answer,
                    "RelevantChunks": "\n".join(relevant_chunks),
                    "Status": "Success"
                })
                successful_count += 1
                break
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    output_data.append({
                        "Question": question,
                        "ExpectedAnswer": expected_answer,
                        "ModelAnswer": f"Error: {str(e)}",
                        "RelevantChunks": "\n".join(relevant_chunks),
                        "Status": "Failed"
                    })
                time.sleep(RETRY_DELAY * (attempt + 1))

    # Save results
    results_df = pd.DataFrame(output_data)
    results_df.to_excel(output_path, index=False)
    print(f"\n✅ Saved {successful_count}/{len(df)} answers to {output_path}")

if __name__ == "__main__":
    download_nltk_data()
    input_path = "backend/tests/evaluation/evaluation_data.xlsx"
    output_path = "backend/tests/evaluation/generated_answers.xlsx"
    generate_answers(input_path, output_path)