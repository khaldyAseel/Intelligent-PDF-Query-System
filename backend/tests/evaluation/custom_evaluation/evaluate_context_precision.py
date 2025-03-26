import pandas as pd
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum()]  # keep only alphanumeric tokens

def evaluate_context_precision(input_file):
    # Load the data
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Ensure NLTK tokens are available
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        import nltk
        nltk.download("punkt")

    # Filter successful rows
    df_success = df[df["Status"] == "Success"].copy()

    # Add dummy context column if not already present (update this if you use real context)
    if "Context" not in df_success.columns:
        print("⚠️ No context column found! Creating dummy context from ExpectedAnswer.")
        df_success["Context"] = df_success["ExpectedAnswer"]

    precision_scores = []
    for _, row in df_success.iterrows():
        try:
            model_tokens = set(tokenize(row["ModelAnswer"]))
            context_tokens = set(tokenize(row["Context"]))

            if len(model_tokens) == 0:
                precision_scores.append(0.0)
            else:
                precision = len(model_tokens & context_tokens) / len(model_tokens)
                precision_scores.append(precision)
        except Exception as e:
            print(f"⚠️ Skipping row due to error: {e}")
            precision_scores.append(0.0)

    df_success["Context_Precision"] = precision_scores

    # Save output
    output_file = input_file.replace(".xlsx", "_context_precision.xlsx")
    df_success.to_excel(output_file, index=False)
    print(f"✅ Context Precision results saved to {output_file}")

if __name__ == "__main__":
    input_path = "backend/tests/evaluation/evaluation_data_results.xlsx"
    evaluate_context_precision(input_path)
