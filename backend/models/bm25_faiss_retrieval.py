# import faiss
# import numpy as np
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer

# # Sample paragraphs (Replace with your actual text data)
# sample_paragraphs = [
#     "The earliest evidence of consumption of cocoa is from 1900 to 1750 BC by the Mokaya people, a pre-Olmec culture from what is nowadays the southern part of Mexico and Guatemala (Powis, 2007). Later, cocoa was first cultivated and domesticated by the Mayan and Aztec peoples.",
#     "Cocoa is grown commercially between 20° north and 20° south of the equator, in areas with a suitable environment for cocoa (e.g. rainfall, soil type). The seven largest cocoa-producing countries are Côte d'Ivoire, Ghana, Indonesia, Nigeria, Cameroon, Brazil, and Ecuador, accounting for 90% of the world crop.",
#     "The main type of cocoa is called Forastero. It was introduced into Bahia, Brazil, and later spread to West Africa. Another type, Trinitario, is a hybrid of Criollo and Forastero, known for its special flavors such as dried fruits or molasses.",
#     "The crop does not all ripen at the same time, so harvesting has to be carried out over several months. Pods are normally harvested every 2–4 weeks. Fermentation is an essential step where natural yeasts and bacteria multiply, breaking down the sugars in the pulp and helping develop cocoa flavors.",
#     "Cocoa beans have to get from the many small farmers, who are often in remote areas of developing countries, to the cocoa processing factories that may be located in temperate countries. The international cocoa markets function as intermediaries between producers and users.",
# ]

# ### 1️⃣ BM25 Retrieval (Keyword-Based) ###
# # Tokenize paragraphs
# tokenized_paragraphs = [para.lower().split() for para in sample_paragraphs]
# bm25 = BM25Okapi(tokenized_paragraphs)

# # Function to retrieve using BM25
# def retrieve_with_bm25(query, top_k=2):
#     tokenized_query = query.lower().split()
#     scores = bm25.get_scores(tokenized_query)
#     top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
#     return [sample_paragraphs[i] for i in top_indices]

# ### 2️⃣ FAISS Vector Storage (Embedding-Based) ###
# # Load a free embedding model
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Encode paragraphs
# paragraph_embeddings = embedding_model.encode(sample_paragraphs)
# dimension = paragraph_embeddings.shape[1]

# # Create FAISS index
# faiss_index = faiss.IndexFlatL2(dimension)
# faiss_index.add(np.array(paragraph_embeddings))

# # Function to retrieve using FAISS
# def retrieve_with_faiss(query, top_k=2):
#     query_embedding = embedding_model.encode([query])
#     distances, indices = faiss_index.search(np.array(query_embedding), top_k)
#     return [sample_paragraphs[i] for i in indices[0]]

# ### 3️⃣ Test Hybrid Retrieval ###
# query_text = "What are the different types of cocoa?"
# # query_text= "How does cocoa fermentation work?"
# # query_text = "Where is cocoa grown?"
# bm25_results = retrieve_with_bm25(query_text, top_k=2)
# faiss_results = retrieve_with_faiss(query_text, top_k=2)

# # Display results
# print("BM25 (Keyword-Based) Results:", bm25_results)
# print("FAISS (Embedding-Based) Results:", faiss_results)


# import faiss
# import numpy as np
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # Sample paragraphs 
# sample_paragraphs = [
#     "The earliest evidence of consumption of cocoa is from 1900 to 1750 BC by the Mokaya people, a pre-Olmec culture from what is nowadays the southern part of Mexico and Guatemala (Powis, 2007). Later, cocoa was first cultivated and domesticated by the Mayan and Aztec peoples.",
#     "Cocoa is grown commercially between 20° north and 20° south of the equator, in areas with a suitable environment for cocoa (e.g. rainfall, soil type). The seven largest cocoa-producing countries are Côte d'Ivoire, Ghana, Indonesia, Nigeria, Cameroon, Brazil, and Ecuador, accounting for 90% of the world crop.",
#     "The main type of cocoa is called Forastero. It was introduced into Bahia, Brazil, and later spread to West Africa. Another type, Trinitario, is a hybrid of Criollo and Forastero, known for its special flavors such as dried fruits or molasses.",
#     "The crop does not all ripen at the same time, so harvesting has to be carried out over several months. Pods are normally harvested every 2–4 weeks. Fermentation is an essential step where natural yeasts and bacteria multiply, breaking down the sugars in the pulp and helping develop cocoa flavors.",
#     "Cocoa beans have to get from the many small farmers, who are often in remote areas of developing countries, to the cocoa processing factories that may be located in temperate countries. The international cocoa markets function as intermediaries between producers and users.",
# ]

# ### 1️⃣ BM25 Retrieval (Keyword-Based) ###
# # Tokenize paragraphs
# tokenized_paragraphs = [para.lower().split() for para in sample_paragraphs]
# bm25 = BM25Okapi(tokenized_paragraphs)

# # Function to retrieve using BM25
# def retrieve_with_bm25(query, top_k=2):
#     tokenized_query = query.lower().split()
#     scores = bm25.get_scores(tokenized_query)
#     top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
#     return [sample_paragraphs[i] for i in top_indices]

# ### 2️⃣ FAISS Vector Storage (Embedding-Based) ###
# # Load embedding model
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Encode paragraphs
# paragraph_embeddings = embedding_model.encode(sample_paragraphs)
# dimension = paragraph_embeddings.shape[1]

# # Create FAISS index
# faiss_index = faiss.IndexFlatL2(dimension)
# faiss_index.add(np.array(paragraph_embeddings))

# # Function to retrieve using FAISS
# def retrieve_with_faiss(query, top_k=2):
#     query_embedding = embedding_model.encode([query])
#     distances, indices = faiss_index.search(np.array(query_embedding), top_k)
#     return [sample_paragraphs[i] for i in indices[0]]

# ### 3️⃣ Hybrid Retrieval (BM25 + FAISS) ###
# def hybrid_retrieval(query, top_k=5):
#     bm25_results = retrieve_with_bm25(query, top_k)
#     faiss_results = retrieve_with_faiss(query, top_k)

#     # Combine results, remove duplicates
#     all_results = list(set(bm25_results + faiss_results))
#     return all_results

# ### 4️⃣ Re-Ranking Using Cross-Encoder ###
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# def rerank_results(query, results):
#     query_pairs = [(query, passage) for passage in results]
#     scores = cross_encoder.predict(query_pairs)

#     # Sort results by score (higher = better match)
#     ranked_results = [x for _, x in sorted(zip(scores, results), reverse=True)]
#     return ranked_results

# ### 5️⃣ Testing the Full Pipeline ###
# # query_text = "What are the different types of cocoa?" 
# # query_text = "How does cocoa fermentation work?"
# query_text = "Where is cocoa grown?"

# # Retrieve results
# retrieved_results = hybrid_retrieval(query_text, top_k=5)

# # Re-rank results
# ranked_results = rerank_results(query_text, retrieved_results)

# # Get the best answer
# best_answer = ranked_results[0] if ranked_results else "No relevant answer found."

# # Display final results
# print("\n🔹 Query:", query_text)
# print("\n✅ Best Retrieved Answer:", best_answer)
# print("\n📌 Other Relevant Results:", ranked_results[1:])  # Show other top-ranked results


import faiss
import numpy as np
import sqlite3
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

### 1️⃣ Load Documents from SQLite Database ###
def load_documents():
    conn = sqlite3.connect("../../text_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM documents")
    retrieved_docs = cursor.fetchall()
    conn.close()
    return [doc[0] for doc in retrieved_docs]

# Load real book data
sample_paragraphs = load_documents()

### 2️⃣ BM25 Retrieval (Keyword-Based) ###
tokenized_paragraphs = [para.lower().split() for para in sample_paragraphs]
bm25 = BM25Okapi(tokenized_paragraphs)

def retrieve_with_bm25(query, top_k=2):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [sample_paragraphs[i] for i in top_indices]

### 3️⃣ FAISS Vector Storage (Embedding-Based) ###
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
paragraph_embeddings = embedding_model.encode(sample_paragraphs)
dimension = paragraph_embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(paragraph_embeddings))

def retrieve_with_faiss(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    return [sample_paragraphs[i] for i in indices[0]]

### 4️⃣ Hybrid Retrieval (BM25 + FAISS) ###
def hybrid_retrieval(query, top_k=5):
    bm25_results = retrieve_with_bm25(query, top_k)
    faiss_results = retrieve_with_faiss(query, top_k)
    return list(set(bm25_results + faiss_results))  # Remove duplicates

### 5️⃣ Re-Ranking Using Cross-Encoder ###
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, results):
    query_pairs = [(query, passage) for passage in results]
    scores = cross_encoder.predict(query_pairs)
    return [x for _, x in sorted(zip(scores, results), reverse=True)]  # Sort by score

### 6️⃣ Testing the Full Pipeline ###
# query_text = "What are the different types of cocoa?"
query_text = "describe the differences between production of regular chocolate, milk chocolate and white chocolate"

retrieved_results = hybrid_retrieval(query_text, top_k=5)
ranked_results = rerank_results(query_text, retrieved_results)
best_answer = ranked_results[0] if ranked_results else "No relevant answer found."

# Display final results
print("\n🔹 Query:", query_text)
print("\n✅ Best Retrieved Answer:", best_answer)
print("\n📌 Other Relevant Results:", ranked_results[1:])  # Show other top-ranked results
