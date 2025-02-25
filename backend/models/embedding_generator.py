import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import ast
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings_from_db():
	"""
    Retrieve document IDs, embeddings, and content from the database.
    Returns lists of document IDs, embeddings, and texts.
    """
	conn = sqlite3.connect("text_database.db")
	cursor = conn.cursor()

	cursor.execute("SELECT id, content, embedding FROM documents")
	rows = cursor.fetchall()
	conn.close()

	doc_ids = []
	texts = []
	embeddings = []

	for row in rows:
		doc_id, text, embedding_blob = row
		embedding = ast.literal_eval(embedding_blob.decode("utf-8"))  # Convert BLOB back to list
		doc_ids.append(doc_id)
		texts.append(text)
		embeddings.append(embedding)

	return doc_ids, texts, embeddings


def dense_retrieval(query, doc_ids, embeddings, texts, top_n=5):
	"""
    Find top N relevant passages using cosine similarity on embeddings.
    """
	query_embedding = model.encode(query)

	similarities = []
	for doc_id, embedding, text in zip(doc_ids, embeddings, texts):
		sim = cosine_similarity([query_embedding], [embedding])[0][0]
		similarities.append((doc_id, text, sim))

	# Sort by similarity and return top_n results
	similarities.sort(key=lambda x: x[2], reverse=True)
	return similarities[:top_n]


def sparse_retrieval(query, texts, doc_ids, top_m=5):
	"""
    Find top M relevant passages using BM25 (keyword-based retrieval).
    """
	tokenized_texts = [word_tokenize(text.lower()) for text in texts]
	bm25 = BM25Okapi(tokenized_texts)

	tokenized_query = word_tokenize(query.lower())
	bm25_scores = bm25.get_scores(tokenized_query)

	# Pair document IDs with scores and sort
	scores = list(zip(doc_ids, texts, bm25_scores))
	scores.sort(key=lambda x: x[2], reverse=True)

	return scores[:top_m]


def normalize_scores(scores):
	"""
    Normalize a list of (doc_id, text, score) pairs to a range of [0, 1].
    """
	if not scores:
		return []

	min_score = min(score for _, _, score in scores)
	max_score = max(score for _, _, score in scores)

	if max_score - min_score == 0:  # Avoid division by zero
		return [(doc_id, text, 1.0) for doc_id, text, _ in scores]

	return [(doc_id, text, (score - min_score) / (max_score - min_score)) for doc_id, text, score in scores]


def hybrid_retrieval(query, top_n=5, top_m=5, alpha=0.5):
	"""
    Perform hybrid retrieval and return the most relevant passages.
    """
	doc_ids, texts, embeddings = get_embeddings_from_db()

	# Step 1: Get top results from both retrieval methods
	dense_results = dense_retrieval(query, doc_ids, embeddings, texts, top_n)
	sparse_results = sparse_retrieval(query, texts, doc_ids, top_m)

	# Step 2: Normalize scores
	dense_results = normalize_scores(dense_results)
	sparse_results = normalize_scores(sparse_results)

	# Step 3: Combine scores using weighted sum
	score_dict = {}

	for doc_id, text, score in dense_results:
		score_dict[doc_id] = (text, score * alpha)  # Weight for dense retrieval

	for doc_id, text, score in sparse_results:
		if doc_id in score_dict:
			score_dict[doc_id] = (
			text, score_dict[doc_id][1] + score * (1 - alpha))  # Add weighted sparse retrieval score
		else:
			score_dict[doc_id] = (text, score * (1 - alpha))

	# Step 4: Sort final scores
	ranked_results = sorted(score_dict.items(), key=lambda x: x[1][1], reverse=True)

	# Step 5: Return the top-ranked passages
	return [(doc_id, text, score) for doc_id, (text, score) in ranked_results]


# Example usage
query_text = "What is the optimal roasting time for cocoa?"
results = hybrid_retrieval(query_text, top_n=5, top_m=5, alpha=0.6)

print("\nTop relevant passages:")
for doc_id, passage, score in results:
	print(
		f"\nDocument ID: {doc_id} (Score: {score:.3f})\nPassage: {passage}...")  # Display only the first 500 chars
