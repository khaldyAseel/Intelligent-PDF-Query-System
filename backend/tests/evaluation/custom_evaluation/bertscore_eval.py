import os
import pandas as pd
from bert_score import score

# def evaluate_bertscore(input_file):
#     # Load the data
#     try:
#         df = pd.read_excel(input_file, sheet_name="Sheet1")
#     except Exception as e:
#         print(f"❌ Error loading file: {e}")
#         return

#     # Prepare candidates and references
#     candidates = df["ModelAnswer"].fillna("").astype(str).tolist()
#     references = df["ExpectedAnswer"].fillna("").astype(str).tolist()

#     # Compute BERTScore (F1)
#     print("🔍 Computing BERTScore...")
#     P, R, F1 = score(candidates, references, lang="en", verbose=True)

#     # Store scores
#     df["BERTScore_F1"] = F1.numpy()
#     df["BERTScore_Precision"] = P.numpy()
#     df["BERTScore_Recall"] = R.numpy()

#     # Save results
#     output_file = input_file.replace(".xlsx", "_bertscore.xlsx")
#     df.to_excel(output_file, index=False)
#     print(f"✅ BERTScore results saved to {output_file}")

def evaluate_bertscore(input_file):
    # Load the data
    try:
        df = pd.read_excel(input_file, sheet_name="Sheet1")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Filter successful rows only
    df_success = df[df["Status"] == "Success"].copy()

    # Prepare candidates and references
    candidates = df_success["ModelAnswer"].fillna("").astype(str).tolist()
    references = df_success["ExpectedAnswer"].fillna("").astype(str).tolist()

    # Compute BERTScore
    print("🔍 Computing BERTScore...")
    P, R, F1 = score(candidates, references, lang="en", verbose=True)

    # Store scores in the filtered DataFrame
    df_success["BERTScore_F1"] = F1.numpy()
    df_success["BERTScore_Precision"] = P.numpy()
    df_success["BERTScore_Recall"] = R.numpy()

    # Merge back with the original DataFrame
    df.update(df_success)

    # Save updated results
    output_file = input_file.replace(".xlsx", "_bertscore.xlsx")
    df.to_excel(output_file, index=False)
    print(f"✅ BERTScore results saved to {output_file}")


if __name__ == "__main__":
    input_path = "backend/tests/evaluation/evaluation_data_results.xlsx"
    evaluate_bertscore(input_path)
