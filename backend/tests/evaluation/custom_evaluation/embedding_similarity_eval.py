import pandas as pd
from sentence_transformers import SentenceTransformer, util

def evaluate_embedding_similarity(input_file):
    # Load the data
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Filter successful rows
    df_success = df[df["Status"] == "Success"].copy()

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode model and expected answers
    try:
        model_embeds = model.encode(df_success["ModelAnswer"].astype(str).tolist(), convert_to_tensor=True)
        expected_embeds = model.encode(df_success["ExpectedAnswer"].astype(str).tolist(), convert_to_tensor=True)
    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        return

    # Compute cosine similarities
    try:
        similarities = util.cos_sim(model_embeds, expected_embeds).diagonal().cpu().numpy()
        df_success["Embedding_Similarity"] = similarities
    except Exception as e:
        print(f"❌ Error during similarity computation: {e}")
        return

    # Save results
    output_file = input_file.replace(".xlsx", "_embedding_similarity.xlsx")
    df_success.to_excel(output_file, index=False)
    print(f"✅ Embedding similarity results saved to {output_file}")

if __name__ == "__main__":
    input_path = "backend/tests/evaluation/evaluation_data_results.xlsx"
    evaluate_embedding_similarity(input_path)
