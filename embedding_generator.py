import sqlite3
from sentence_transformers import SentenceTransformer
import sqlite3

# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to SQLite database
conn = sqlite3.connect("text_database.db")
cursor = conn.cursor()

# Add new column 'embedding' to the table if it doesn't already exist
try:
    cursor.execute("""
    ALTER TABLE documents
    ADD COLUMN embedding BLOB
    """)
    conn.commit()
except sqlite3.OperationalError:
    print("Column 'embedding' already exists. Skipping creation.")

# Query all rows from the 'documents' table
cursor.execute("SELECT id, content FROM documents")
rows = cursor.fetchall()

# Process each row, generate the embedding, and update the row
for row in rows:
    doc_id, content = row

    if content:
        # Generate the embedding for the content
        embedding = model.encode(content).tolist()  # Convert to list for saving

        # Convert the embedding to a format suitable for SQLite (using BLOB for storage)
        embedding_blob = sqlite3.Binary(bytes(str(embedding), 'utf-8'))

        # Update the 'embedding' column in the database for the corresponding document
        cursor.execute("""
        UPDATE documents
        SET embedding = ?
        WHERE id = ?
        """, (embedding_blob, doc_id))

        # Commit the changes after each update
        conn.commit()

    print(f"Processed: {doc_id}")

# Close the database connection
conn.close()

print("Embeddings for all documents have been generated and stored.")
