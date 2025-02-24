import os
import sqlite3
import ast
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# For splitting text into nodes (chunks)
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter

nltk.download("punkt")

# Load your SentenceTransformer model (adjust model as needed)
model = SentenceTransformer('BAAI/bge-large-en')


def get_docs_from_db(db_path="../../text_database.db"):
	"""
	Retrieve documents from the database.
	Each document should have columns: id, content, metadata, and embedding.
	"""
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	cursor.execute("SELECT id, content, page, embedding FROM documents")
	rows = cursor.fetchall()
	conn.close()

	docs = []
	for row in rows:
		doc_id, content, metadata_blob, embedding_blob = row

		# Convert metadata (if stored as a string or BLOB) into a Python object.
		try:
			if isinstance(metadata_blob, bytes):
				metadata = ast.literal_eval(metadata_blob.decode("utf-8"))
			else:
				metadata = ast.literal_eval(metadata_blob)
		except Exception:
			metadata = metadata_blob

		# Convert the embedding from BLOB to a Python list.
		try:
			embedding = ast.literal_eval(embedding_blob.decode("utf-8"))
		except Exception:
			embedding = embedding_blob

		docs.append({
			"id": doc_id,
			"content": content,
			"page": metadata,
			"parent":metadata,
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
			metadata=doc["metadata"]
		)
		nodes = splitter.get_nodes_from_documents([document])

		# Optionally, include the original document ID in the node's metadata.
		for node in nodes:
			node.metadata["doc_id"] = doc["id"]
		all_nodes.extend(nodes)

	# (Optional) Print out the nodes for inspection.
	for node in all_nodes:
		print("NODE TEXT:", node.text)
		print("METADATA:", node.metadata, "\n")

	return all_nodes


def hybrid_node_retrieval(query, alpha=0.6, top_k=5):
	"""
	Perform hybrid retrieval over nodes (chunks) using both BM25 (sparse) and
	dense embedding similarity.

	Parameters:
	  query   : The search query.
	  alpha   : Weight for the dense score (0 <= alpha <= 1). (1 - alpha) is for BM25.
	  top_k   : Number of top nodes to return.

	Returns:
	  A list of tuples: (node, hybrid_score)
	"""
	# Get nodes from the database (split documents into chunks)
	nodes = get_nodes_from_db()

	# Extract node texts for retrieval.
	texts = [node.text for node in nodes]

	# --- Sparse Retrieval: BM25 ---
	tokenized_texts = [word_tokenize(text.lower()) for text in texts]
	bm25 = BM25Okapi(tokenized_texts)
	tokenized_query = word_tokenize(query.lower())
	bm25_scores = bm25.get_scores(tokenized_query)

	# --- Dense Retrieval: Compute embeddings for each node on the fly ---
	query_embedding = model.encode(query)
	dense_scores = []
	for text in texts:
		node_embedding = model.encode(text)
		sim = cosine_similarity([query_embedding], [node_embedding])[0][0]
		dense_scores.append(sim)

	# --- Normalize scores to [0, 1] ---
	def normalize(scores):
		scores = np.array(scores)
		min_score = scores.min()
		max_score = scores.max()
		if max_score - min_score == 0:
			return np.ones_like(scores)
		return (scores - min_score) / (max_score - min_score)

	bm25_norm = normalize(bm25_scores)
	dense_norm = normalize(dense_scores)

	# --- Combine scores using weighted sum ---
	hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

	# --- Pair each node with its hybrid score and sort ---
	node_scores = list(zip(nodes, hybrid_scores))
	node_scores.sort(key=lambda x: x[1], reverse=True)

	return node_scores[:top_k]


# Example usage:
if __name__ == "__main__":
	query = "What factors contribute to price volatility in the cocoa market?"
	top_nodes = hybrid_node_retrieval(query, alpha=0.6, top_k=5)

	# Combine the top nodes' texts to create context for the LLM.
	context_lst = [node.text for node, score in top_nodes]
	all_context = "\n".join(context_lst)

	print("\n--- Combined Context for LLM ---\n", all_context)

# At this point, you could pass 'all_context' to your LLM for generating an answer.
# For example:
# response = client.chat.completions.create(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#     messages=[
#         {"role": "system", "content": "You are a helpful chatbot."},
#         {"role": "user", "content": f"Answer the question: {query}\nUsing only the information provided here:\n{all_context}"}
#     ]
# )
# print("\n--- LLM Response ---\n", response.choices[0].message.content)
