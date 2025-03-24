import pickle

import sqlite3
import numpy as np

# For splitting text into nodes (chunks)
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
import time

db_path="../backend/database/text_database.db"
pickle_db_path="../backend/database/pickle_database.pkl"


def get_docs_from_db(db_path=db_path):
    """
    Retrieve chunks from the database.
    Each chunk should have columns: chunk_id, chunk_content, metadata, and chunk_embedding.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all chunks with embeddings
    cursor.execute("SELECT chunk_id, chunk_content, subchapter_title, parent, page, type, chunk_embeddings FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for row in rows:
        chunk_id, content, subchapter, parent, page_value,type, embedding_blob = row

        # Create metadata as a dictionary with subchapter, parent, and page
        metadata = {"subchapter": subchapter, "parent": parent, "page": page_value,"type":type}

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
    docs = get_docs_from_db(db_path)
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
            node.metadata["embedding"] = doc["embedding"]
        all_nodes.extend(nodes)

    return all_nodes

def save_nodes_to_pickle(filepath=pickle_db_path):
    nodes = get_nodes_from_db()
    with open(filepath, "wb") as f:
        pickle.dump(nodes, f)
    print("✅ Nodes saved to", filepath)



start = time.perf_counter()
save_nodes_to_pickle()

end = time.perf_counter()
elapsed_time = end - start
print(f"Elapsed time nodes pickle db: {elapsed_time:.6f} seconds")