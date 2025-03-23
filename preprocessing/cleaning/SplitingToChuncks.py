import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import json

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Connect using the absolute path
db_path = "../../backend/database/text_database.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch subchapter content from the documents table
cursor.execute("SELECT id, content, parent, page,type FROM documents")
subchapters = cursor.fetchall()  # List of (id, content, parent, page)


# Initialize LangChain's RecursiveTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Target chunk size
    chunk_overlap=50,  # Overlap to maintain context
    length_function=len,  # Use length of text for measuring
    separators=["\n\n", ". ", "? ", "! "]  # Split at sentences
)

# Create a new table for storing text chunks
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-incrementing ID
    chunk_content TEXT,  -- The chunk itself
    subchapter_title TEXT,  -- The subchapter title (from documents.id)
    parent TEXT,  -- Parent metadata
    page INTEGER, -- Page number
    type TEXT     -- type metadata
)
""")
conn.commit()


# Insert chunks into the chunks table
for sub_id, content, parent, page,type in subchapters:
    # Split the subchapter into chunks
    chunks = text_splitter.split_text(content)

    # Store each chunk in the database
    for chunk in chunks:
        cursor.execute("""
        INSERT INTO chunks (chunk_content, subchapter_title, parent, page, type)
        VALUES (?, ?, ?, ?, ?)
        """, (chunk, sub_id, parent, page,type))

conn.commit()

# Fetch and display stored chunks
cursor.execute("SELECT * FROM chunks")
rows = cursor.fetchall()

# Print sample data
for row in rows:
    print(f"Chunk ID: {row[0]}")
    print(f"Content: {row[1][:100]}...")  # Display only first 100 chars
    print(f"Subchapter: {row[2]}, Parent: {row[3]}, Page: {row[4]}")
    print(f"type:{row[5]}")
    print("-" * 50)

# Close database connection
conn.close()
