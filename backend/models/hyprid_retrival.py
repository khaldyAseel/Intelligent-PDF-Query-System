import sqlite3
import ast
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
	Retrieve documents from the database.
	Each document should have columns: id, content, metadata, and embedding.
	"""
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	cursor.execute("SELECT id, content, parent, page, chunk_embeddings FROM documents")
	rows = cursor.fetchall()
	conn.close()

	docs = []
	for row in rows:
		doc_id, content, parent, page_value, embedding_blob = row

		# Create metadata as a dictionary with both parent and page
		metadata = {"parent": parent, "page": page_value}

		# Convert the embedding from BLOB to a Python list.
		try:
			embedding = ast.literal_eval(embedding_blob.decode("utf-8"))
		except Exception:
			embedding = embedding_blob

		docs.append({
			"id": doc_id,
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

	# --- Step 2: Dense Retrieval (Only for top-k BM25 results) ---
	query_embedding = model.encode(query)
	dense_scores = []

	for node in top_k_nodes:
		node_embedding = model.encode(node.text)
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

	bm25_norm = normalize(top_k_bm25_scores)
	dense_norm = normalize(dense_scores)

	# --- Step 3: Combine Scores ---
	hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

	# --- Step 4: Sort and Return Top-K Results ---
	node_scores = list(zip(top_k_nodes, hybrid_scores))
	node_scores.sort(key=lambda x: x[1], reverse=True)

	return node_scores  # Return top-k results after hybrid scoring

def bert_extract_answer(query, retrieved_nodes):
	"""
	Uses BERT-QA to extract a direct answer from the top retrieved nodes.
	"""
	best_answer = ""
	best_score = -float("inf")

	for node, _ in retrieved_nodes:  # Iterate through the top nodes
		context = node.text
		result = qa_pipeline(question=query, context=context)

		# Compare confidence scores and keep the best answer
		if result["score"] > best_score:
			best_score = result["score"]
			best_answer = result["answer"]

	return best_answer

if __name__ == "__main__":
	query = "What is the optimal roasting time for cocoa?"
	# Retrieve top nodes using hybrid retrieval
	top_nodes = hybrid_node_retrieval(query, alpha=0.6, top_k=5)

	# (Optional) Print out the nodes for inspection.
	for node in top_nodes:
		print("NODE TEXT:", node)

	# Extract best answer using BERT
	bert_answer = bert_extract_answer(query, top_nodes)

	# Combine BERT answer + retrieved context for LLaMA
	all_context = "\n".join([node.text for node, _ in top_nodes])
	response = client.chat.completions.create(
		model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
		messages=[
			{"role": "system", "content": "You are a helpful chatbot."},
			{"role": "user", "content": f"Answer the question: {query}. Here is an extracted answer: {bert_answer}. You can also use additional context: {all_context}"},
		],
	)
	print(response.choices[0].message.content)
