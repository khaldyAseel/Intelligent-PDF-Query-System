import sqlite3
from sentence_transformers import SentenceTransformer
import ast
import numpy as np

# Load the pre-trained embedding model
model = SentenceTransformer('BAAI/bge-large-en')


def generate_and_save_embeddings():
	"""
	Generate embeddings for document content stored in chunks and save them to a new column.

	The function:
	  - Connects to the SQLite database.
	  - Adds an 'embedding' column if it doesn't exist.
	  - Reads the content (assumed to be split into chunks) for each document.
	  - Generates an embedding for each chunk.
	  - Averages the chunk embeddings to produce a single document embedding.
	  - Saves the document embedding to the 'embedding' column.

	Note: The model "all-MiniLM-L6-v2" has a token limit (around 256 tokens),
		  so splitting the text into smaller chunks helps avoid truncation issues.
	"""
	# Connect to the database
	conn = sqlite3.connect("../../text_database.db")
	cursor = conn.cursor()

	# Create the 'embedding' column if it does not exist
	try:
		cursor.execute("ALTER TABLE documents ADD COLUMN embedding BLOB")
	except sqlite3.OperationalError:
		# Likely the column already exists, so we can pass.
		pass

	# Retrieve document IDs and their content
	cursor.execute("SELECT id, content FROM documents")
	rows = cursor.fetchall()

	for doc_id, content in rows:
		# Convert the stored content into a list of chunks.
		# If the content isn't in list format, treat it as a single chunk.
		try:
			chunks = ast.literal_eval(content)
			if not isinstance(chunks, list):
				chunks = [content]
		except Exception:
			chunks = [content]

		# Generate embeddings for each chunk.
		# The model will internally handle token limits (truncating if necessary).
		chunk_embeddings = model.encode(chunks, show_progress_bar=False)

		# If there are multiple chunks, average the embeddings.
		if len(chunk_embeddings) > 1:
			avg_embedding = np.mean(chunk_embeddings, axis=0)
			embedding_to_save = avg_embedding.tolist()
		else:
			# If there's only one chunk, use its embedding directly.
			embedding_to_save = (chunk_embeddings[0].tolist()
								 if hasattr(chunk_embeddings[0], 'tolist')
								 else list(chunk_embeddings[0]))

		# Convert the embedding to a string and then to a UTF-8 encoded BLOB for storage.
		embedding_blob = str(embedding_to_save).encode("utf-8")
		cursor.execute("UPDATE documents SET embedding = ? WHERE id = ?", (embedding_blob, doc_id))

	# Commit all changes and close the connection.
	conn.commit()
	conn.close()
