import sqlite3
import numpy as np
from dotenv import load_dotenv
import os
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.retrievers.bm25 import BM25Retriever
from together import Together
# For splitting text into nodes (chunks)
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from transformers import pipeline
import Stemmer
nltk.download("punkt")

# Load your SentenceTransformer model (adjust model as needed)
model = SentenceTransformer('BAAI/bge-large-en')

# Load a BERT-based Question Answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

load_dotenv(dotenv_path="../api/.env")
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)


def get_docs_from_db(db_path="../../backend/database/text_database.db"):
    """
    Retrieve chunks from the database.
    Each chunk should have columns: chunk_id, chunk_content, metadata, and chunk_embedding.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all chunks with embeddings
    cursor.execute("SELECT chunk_id, chunk_content, subchapter_title, parent, page, chunk_embeddings FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for row in rows:
        chunk_id, content, subchapter, parent, page_value, embedding_blob = row

        # Create metadata as a dictionary with subchapter, parent, and page
        metadata = {"subchapter": subchapter, "parent": parent, "page": page_value}

        # Convert the embedding from BLOB to a numpy array
        if embedding_blob:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
        else:
            embedding = None  # Handle missing embeddings

        docs.append({
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        })

    return docs

def get_nodes_from_db():
    """
    For each document from the database, use SentenceSplitter to split the text
    into smaller nodes (chunks). This allows you to perform more granular retrieval.
    """
    docs = get_docs_from_db()
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    all_nodes = []

    for doc in docs:
        # Create a Document object required by the splitter.
        document = Document(
            text=doc["content"],
            metadata=doc["metadata"],
        )
        nodes = splitter.get_nodes_from_documents([document])

        # Optionally, include the original document ID in the node's metadata.
        for node in nodes:
            node.metadata["doc_id"] = doc["id"]
        all_nodes.extend(nodes)
    return all_nodes

def embeddings_from_docs(db_path="../../backend/database/text_database.db", bm25_nodes=None):
    """
    Fetch embeddings from the database for given BM25 nodes.

    :param db_path: Path to the SQLite database.
    :param bm25_nodes: List of BM25 retrieved nodes.
    :return: List of embeddings for the nodes.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    embeddings_lst = []
    if bm25_nodes:  # If BM25 nodes are provided
        subchapters = list(set(node.metadata["subchapter"] for node in bm25_nodes))  # Unique subchapter titles
        print(f"Retrieving embeddings for subchapters: {subchapters}")

        query = f"SELECT chunk_embeddings FROM chunks WHERE subchapter_title IN ({','.join(['?'] * len(subchapters))})"
        cursor.execute(query, subchapters)
    else:
        cursor.execute("SELECT chunk_embeddings FROM chunks")

    rows = cursor.fetchall()
    conn.close()

    # Extract embeddings from the returned tuples and deserialize
    for row in rows:
        embedding_bytes = row[0]  # Retrieve the binary string from the database
        if embedding_bytes is not None:  # Check if the embedding is not None
            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)  # Convert bytes back to numpy array
            embeddings_lst.append(embedding_array)
        else:
            print("Embedding not found for one of the nodes")

    embeddings_2d_array = np.array(embeddings_lst)
    return embeddings_2d_array


def hybrid_node_retrieval(query, alpha=0.6, top_k=5):
    """
    Perform hybrid retrieval over nodes (chunks) using both BM25 (sparse) and
    dense embedding similarity, but compute dense similarity only for the top-k BM25 results.

    Parameters:
      query   : The search query.
      alpha   : Weight for the dense score (0 <= alpha <= 1). (1 - alpha) is for BM25.
      top_k   : Number of top nodes to return.

    Returns:
      A list of tuples: (node, hybrid_score)
    """
    # Get nodes from the database (split documents into chunks)
    nodes = get_nodes_from_db()

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    bm25_results = bm25_retriever.retrieve(query)

    # Select top-k nodes and corresponding BM25 scores
    top_k_nodes = [res.node for res in bm25_results[:top_k]]
    top_k_bm25_scores = [res.score for res in bm25_results[:top_k]]

    # --- Step 2: Dense Retrieval ---
    query_embedding = model.encode(query)
    dense_scores = []
    embeddings = embeddings_from_docs(bm25_nodes=top_k_nodes)
    # Compute cosine similarity between query and each embedding
    for embedding in embeddings:
        sim = cosine_similarity([query_embedding], [embedding])[0][0]
        dense_scores.append(sim)
    if len(dense_scores)>5:
        # Convert scores to a NumPy array for easy sorting
        dense_scores = np.array(dense_scores)
        # Get the indices of the top 5 scores
        top_indices = np.argsort(dense_scores)[-5:]  # Sort and take the highest 5
        top_dense_scores = dense_scores[top_indices]
        top_bm25_scores = top_k_bm25_scores
    elif len(dense_scores)<len(top_k_bm25_scores):
        top_dense_scores = dense_scores
        top_k_bm25_scores = np.array(top_k_bm25_scores)
        top_indices_bm25 = np.argsort(top_k_bm25_scores)[-len(dense_scores):]
        top_bm25_scores = top_k_bm25_scores[top_indices_bm25]

    # --- Normalize scores to [0, 1] ---
    def normalize(scores):
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    bm25_norm = normalize(top_bm25_scores)
    dense_norm = normalize(top_dense_scores)

    # --- Step 3: Combine Scores ---
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

    # --- Step 4: Sort and Return Top-K Results ---
    node_scores = list(zip(top_k_nodes, hybrid_scores))
    node_scores.sort(key=lambda x: x[1], reverse=True)

    return node_scores  # Return top-k results after hybrid scoring
