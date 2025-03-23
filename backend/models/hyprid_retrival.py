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
import sqlite3
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
    cursor.execute("SELECT chunk_id, chunk_content, subchapter_title, parent, page,type, chunk_embeddings FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for row in rows:
        chunk_id, content, subchapter, parent, page_value,type, embedding_blob = row

        # Create metadata as a dictionary with subchapter, parent, and page
        metadata = {"subchapter": subchapter, "parent": parent, "page": page_value,"type": type }

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

    # Process Content Chunks Normally
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

def subchapters_and_nodes(nodes):
    """
    This function takes the nodes from the database and creates a dictionary in which the keys are the subchapters'
    titles, and the values are a list of the nodes found in the subchapter.
    :param nodes: the nodes from the database docs.
    :return: a dictionary {"subchapter's title" : [nodes in the subchapter]}
    """
    subchapters = list(set([node.metadata["subchapter"] for node in nodes]))
    subchapters_and_nodes_dict = {}

    for subchapter in subchapters:
        subchapter_nodes = []
        for node in nodes:
            if (subchapter == node.metadata["subchapter"]):
                subchapter_nodes.append(node)

        subchapters_and_nodes_dict[subchapter] = subchapter_nodes
    return subchapters_and_nodes_dict

def keywords_retriever(query, nodes, top_k=5):
    """
    This function uses BM25 retriever to retrieve top_k relevant nodes based on keywords approach.
    :param query: the user input.
    :param nodes: the nodes from the database docs.
    :param top_k: number of returned nodes from the retriever.
    :return: top_k nodes, top_k nodes' scores. (by keyword).
    """
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    bm25_results = bm25_retriever.retrieve(query)

    # Select top_k nodes and corresponding BM25 scores
    top_k_bm25_nodes = [res.node for res in bm25_results[:top_k]]
    top_k_bm25_scores = [res.score for res in bm25_results[:top_k]]
    return top_k_bm25_nodes, top_k_bm25_scores

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

def embedding_retriever(query, top_k_bm25_nodes, subchapters_nodes_dict, top_k=5):
    """
    This function performs embedding retrieval, and retrieve top_k nodes from the subchapters of the
    previously retrieved BM25 nodes.
    :param query: the user input.
    :param top_k_bm25_nodes: top_k BM25 retrieved nodes.
    :param subchapters_nodes_dict: dictionary of the subchapters and their nodes.
    :param top_k: number of returned nodes from the retriever.
    :return: top_k nodes, top_k nodes' scores. (by embedding).
    """
    query_embedding = model.encode(query)
    dense_scores = []
    subchapters_nodes = []
    embeddings_lst = []
    nodes_with_embeddings = []

    subchapters = [node.metadata["subchapter"] for node in top_k_bm25_nodes]
    subchapters_nodes_lists = [subchapters_nodes_dict[subchapter] for subchapter in subchapters]

    for lst in subchapters_nodes_lists:
        subchapters_nodes.extend(lst)

    for node in subchapters_nodes:
        embedding = node.metadata["embedding"]
        if embedding is not None:  # Check if the embedding is not None
            embeddings_lst.append(embedding)
            nodes_with_embeddings.append(node)
        else:
            print("Embedding not found for one of the nodes")

    embeddings_2d_array = np.array(embeddings_lst)
    for embedding in embeddings_2d_array:
        sim = cosine_similarity([query_embedding], [embedding])[0][0]
        dense_scores.append(sim)

    if len(dense_scores) > top_k:
        # Convert scores to a NumPy array for easy sorting
        dense_scores = np.array(dense_scores)
        nodes_with_embeddings = np.array(nodes_with_embeddings)
        # Get the indices of the top k scores
        top_indices = np.argsort(dense_scores)[-top_k:]  # Sort and take the highest k
        top_dense_scores = list(dense_scores[top_indices])
        top_emb_nodes = list(nodes_with_embeddings[top_indices])
    else:
        top_dense_scores = dense_scores
        top_emb_nodes = nodes_with_embeddings

    return top_emb_nodes, top_dense_scores

def normalize(scores):
    """
    Normalize scores to a [0,1] range in min-max normalization approach.
    :param scores: list of scores
    :return: normalizes scores.
    """
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)

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

    # Separate metadata chunks from other content
    metadata_chunks = [node for node in nodes if node.metadata.get("type") == "metadata"]
    subchapter_chunks = [node for node in nodes if node.metadata.get("type") == "subchapter"]

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
    else:
        top_bm25_scores = top_k_bm25_scores
        top_dense_scores = dense_scores
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

    # --- Step 3: Combine Scores with Metadata Boost ---
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

    # --- Step 4: Sort and Return Top-K Results ---
    node_scores = list(zip(top_k_nodes, hybrid_scores))
    node_scores.sort(key=lambda x: x[1], reverse=True)

    # --- Metadata Priority: If metadata exists, return only that ---
    best_metadata = None
    for node, score in node_scores:
        if node in metadata_chunks:
            best_metadata = (node, score)
            break  # Stop searching as we only need the best metadata chunk

    if best_metadata:
        return [best_metadata]  # Return only the best metadata chunk

    # If no metadata found, return top-k subchapter chunks
    return node_scores[:top_k]

