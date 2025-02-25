import sqlite3
from sentence_transformers import SentenceTransformer
import ast
from transformers import AutoTokenizer

# Connect to the existing database
conn = sqlite3.connect("../../backend/database/text_database.db")
cursor = conn.cursor()

# Load the pre-trained embedding model
model = SentenceTransformer('BAAI/bge-large-en')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en')
max_tokens = model.max_seq_length  # Ensure we respect model constraints
print(max_tokens)

# Add a column for chunk-wise embeddings if it doesn't exist
try:
    cursor.execute("ALTER TABLE documents ADD COLUMN chunk_embeddings BLOB")
except sqlite3.OperationalError:
    pass

def generate_and_save_chunk_embeddings():
    """
    Generate embeddings for each chunk of document content and store them in a new column.
    """
    # Retrieve document IDs and their content
    cursor.execute("SELECT id, content FROM documents")
    rows = cursor.fetchall()

    for doc_id, content in rows:
        # Convert the stored content into a list of chunks.
        try:
            chunks = ast.literal_eval(content)
            if not isinstance(chunks, list):
                chunks = [content]
        except Exception:
            chunks = [content]

        # Generate embeddings for each chunk.
        chunk_embeddings = model.encode(chunks, show_progress_bar=False)
        chunk_embeddings_list = [embedding.tolist() for embedding in chunk_embeddings]

        # Convert the list of chunk embeddings to a string for storage
        chunk_embeddings_blob = str(chunk_embeddings_list).encode("utf-8")
        cursor.execute("UPDATE documents SET chunk_embeddings = ? WHERE id = ?", (chunk_embeddings_blob, doc_id))

    conn.commit()

def view_database():
    """
    Queries and displays all records from the SQLite database.
    """
    cursor.execute("SELECT * FROM documents LIMIT 10")
    rows = cursor.fetchall()
    print("\nViewing Database Contents:")
    for row in rows:
        print("\n".join(map(str, row)))
        print()  # Adds an extra newline between rows

# Run the embedding generation and update the database.
generate_and_save_chunk_embeddings()
# Now view the updated database
view_database()

# Commit all changes and close the connection.
conn.commit()
conn.close()
