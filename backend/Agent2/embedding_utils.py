import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model globally
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index
def create_faiss_index(text_list):
    embeddings = np.array([model.encode(text) for text in text_list]).astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # Map indices to text
    text_map = {i: text for i, text in enumerate(text_list)}

    return faiss_index, text_map

# Query FAISS index
def query_faiss_index(query, faiss_index, text_map):
    query_embedding = model.encode(query).astype("float32")
    distances, indices = faiss_index.search(np.array([query_embedding]), k=1)
    return text_map[indices[0][0]] if indices[0].size > 0 else None
