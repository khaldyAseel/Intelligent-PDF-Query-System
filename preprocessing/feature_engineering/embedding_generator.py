import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np

# Connect to the existing database
conn = sqlite3.connect("../../backend/database/text_database.db")
cursor = conn.cursor()

# Load the pre-trained embedding model
model = SentenceTransformer('BAAI/bge-large-en')

# Add a column for chunk embeddings in the chunks table if it doesn't exist
try:
    cursor.execute("ALTER TABLE chunks ADD COLUMN chunk_embeddings BLOB")
except sqlite3.OperationalError:
    pass


def generate_and_save_chunk_embeddings():
    """
    Generate embeddings for each chunk and store them in the database.
    """
    cursor.execute("SELECT chunk_id, chunk_content FROM chunks")
    rows = cursor.fetchall()

    for chunk_id, content in rows:
        # Generate embedding for the chunk
        embedding = model.encode(content, show_progress_bar=False)
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        # Store the embedding in the database
        cursor.execute("UPDATE chunks SET chunk_embeddings = ? WHERE chunk_id = ?", (embedding_blob, chunk_id))

    conn.commit()


def view_database():
    """
    Queries and displays all records from the chunks table.
    """
    cursor.execute("SELECT * FROM chunks LIMIT 5")
    rows = cursor.fetchall()

    for row in rows:
        print("\n".join(map(str, row)))
        print()  # Adds an extra newline between rows

# Run the embedding generation and update the database
# generate_and_save_chunk_embeddings()
view_database()

# Commit and close the connection
conn.commit()
conn.close()
