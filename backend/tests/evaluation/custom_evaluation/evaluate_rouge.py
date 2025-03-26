import pandas as pd
from rouge_score import rouge_scorer

def evaluate_rouge(input_file):
    # Load the data
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Prepare score list
    rouge_l_scores = []

    df_success = df[df["Status"] == "Success"].copy()

    for ref, cand in zip(df_success["ExpectedAnswer"], df_success["ModelAnswer"]):
        try:
            scores = scorer.score(str(ref), str(cand))
            rouge_l_scores.append(scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"⚠️ Skipping a row due to error: {e}")
            rouge_l_scores.append(0.0)

    # Add to DataFrame
    df_success["ROUGE_L"] = rouge_l_scores

    # Save results
    output_file = input_file.replace(".xlsx", "_rouge.xlsx")
    df_success.to_excel(output_file, index=False)
    print(f"✅ ROUGE-L results saved to {output_file}")

if __name__ == "__main__":
    input_path = "backend/tests/evaluation/evaluation_data_results.xlsx"
    evaluate_rouge(input_path)
