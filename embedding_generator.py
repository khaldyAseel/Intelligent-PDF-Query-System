import sqlite3
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

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

print("Embeddings for all documents have been generated and stored.")

# Close the database connection
conn.close()

# # Connect to SQLite database
# conn = sqlite3.connect("text_database.db")
# cursor = conn.cursor()
# # Query all rows from the 'documents' table
# cursor.execute("SELECT id, content, embedding FROM documents")
# rows = cursor.fetchall()

# # Print the results
# print("Viewing Database Contents:")
# print("-" * 50)
# for row in rows:
#     doc_id, content, embedding = row
#
#     # Print document ID and content
#     print(f"ID: {doc_id}")
#     print(f"Content: {content[:200]}...")  # Preview first 200 characters of content
#
#     # Print the embedding (you can convert it from binary format if needed)
#     print(f"Embedding: {embedding[:50]}...")  # Preview first 50 bytes of the embedding
#
#     print("-" * 50)
#
# # Close the database connection
# conn.close()

def get_most_relevant_embeddings(context_text, top_n=5):
    """
    Takes a context text and finds the most relevant embeddings from the database using cosine similarity.
    Returns the IDs of the most relevant documents and their similarities.
    """
    # Connect to SQLite database
    conn = sqlite3.connect("text_database.db")
    cursor = conn.cursor()

    # Generate the embedding for the context text
    context_embedding = model.encode(context_text)

    # Query all rows from the 'documents' table
    cursor.execute("SELECT id, embedding FROM documents")
    rows = cursor.fetchall()

    similarities = []

    # Process each row and calculate cosine similarity
    for row in rows:
        doc_id, embedding_blob = row

        # Convert the embedding from binary format to list
        embedding = ast.literal_eval(embedding_blob.decode('utf-8'))

        # Calculate cosine similarity
        sim = cosine_similarity([context_embedding], [embedding])[0][0]
        similarities.append((doc_id, sim))

    # Sort the results by similarity (in descending order)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the top_n most relevant documents
    most_relevant = similarities[:top_n]

    # Close the database connection
    conn.close()

    return most_relevant


# Example usage
context_text = "What are the health benefits of cacao?"
top_n = 5
relevant_documents = get_most_relevant_embeddings(context_text, top_n)

print(f"Top {top_n} most relevant documents based on cosine similarity:")
for doc_id, sim in relevant_documents:
    print(f"ID: {doc_id}, Similarity: {sim}")
