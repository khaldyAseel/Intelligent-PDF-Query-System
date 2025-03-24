import numpy as np
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.retrievers.bm25 import BM25Retriever
from together import Together
import Stemmer
import pickle
db_path=r"../database/text_database.db"
pickle_db_path=r"../database/pickle_database.pkl"

# Load your SentenceTransformer model (adjust model as needed)
model = SentenceTransformer('BAAI/bge-large-en')

load_dotenv(dotenv_path="../api/.env")
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

def load_nodes_from_pickle(filepath=pickle_db_path):
    with open(filepath, "rb") as f:
        nodes = pickle.load(f)
    return nodes


def subchapters_and_nodes(nodes):
    """
    This function takes the nodes from the database and creates a dictionary in which the keys are the subchapters'
    titles, and the values are a list of the nodes found in the subchapter.
    :param nodes: the nodes from the database docs.
    :return: a dictionary {"subchapter's title" : [nodes in the subchapter]}
    """
    subchapters = list(set([node.metadata["subchapter"] for node in nodes]))
    subchapters_and_nodes_dict = {}

    for subchapter in subchapters:
        subchapter_nodes = []
        for node in nodes:
            if (subchapter == node.metadata["subchapter"]):
                subchapter_nodes.append(node)

        subchapters_and_nodes_dict[subchapter] = subchapter_nodes
    return subchapters_and_nodes_dict


def keywords_retriever(query, nodes, top_k):
    """
    This function uses BM25 retriever to retrieve top_k relevant nodes based on keywords approach.
    :param query: the user input.
    :param nodes: the nodes from the database docs.
    :param top_k: number of returned nodes from the retriever.
    :return: top_k nodes, top_k nodes' scores. (by keyword).
    """
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    bm25_results = bm25_retriever.retrieve(query)

    # Select top_k nodes and corresponding BM25 scores
    top_k_bm25_nodes = [res.node for res in bm25_results[:top_k]]
    top_k_bm25_scores = [res.score for res in bm25_results[:top_k]]
    return top_k_bm25_nodes, top_k_bm25_scores


def embedding_retriever(query, top_k_bm25_nodes, subchapters_nodes_dict, top_k):
    """
    This function performs embedding retrieval, and retrieve top_k nodes from the subchapters of the
    previously retrieved BM25 nodes.
    :param query: the user input.
    :param top_k_bm25_nodes: top_k BM25 retrieved nodes.
    :param subchapters_nodes_dict: dictionary of the subchapters and their nodes.
    :param top_k: number of returned nodes from the retriever.
    :return: top_k nodes, top_k nodes' scores. (by embedding).
    """
    query_embedding = model.encode(query)
    dense_scores = []
    subchapters_nodes = []
    embeddings_lst = []
    nodes_with_embeddings = []

    subchapters = [node.metadata["subchapter"] for node in top_k_bm25_nodes]
    subchapters_nodes_lists = [subchapters_nodes_dict[subchapter] for subchapter in subchapters]

    for lst in subchapters_nodes_lists:
        subchapters_nodes.extend(lst)

    for node in subchapters_nodes:
        embedding = node.metadata["embedding"]
        if embedding is not None:  # Check if the embedding is not None
            embeddings_lst.append(embedding)
            nodes_with_embeddings.append(node)
        else:
            print("Embedding not found for one of the nodes")

    embeddings_2d_array = np.array(embeddings_lst)
    for embedding in embeddings_2d_array:
        sim = cosine_similarity([query_embedding], [embedding])[0][0]
        dense_scores.append(sim)

    if len(dense_scores) > top_k:
        # Convert scores to a NumPy array for easy sorting
        dense_scores = np.array(dense_scores)
        nodes_with_embeddings = np.array(nodes_with_embeddings)
        # Get the indices of the top k scores
        top_indices = np.argsort(dense_scores)[-top_k:]  # Sort and take the highest k
        top_dense_scores = list(dense_scores[top_indices])
        top_emb_nodes = list(nodes_with_embeddings[top_indices])
    else:
        top_dense_scores = dense_scores
        top_emb_nodes = nodes_with_embeddings

    return top_emb_nodes, top_dense_scores


def normalize(scores):
    """
    Normalize scores to a [0,1] range in min-max normalization approach.
    :param scores: list of scores
    :return: normalizes scores.
    """
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def hybrid_retriever(top_k_bm25_nodes, top_k_bm25_scores ,top_emb_nodes, top_dense_scores, alpha, top_k):
    """
    This function returns the top_k nodes using a hybrid retriever, it gets both retrievers' nodes, normalize
    their scores, and re-rank the nodes to choose the top_k using weighted sum re-ranking method, where alpha
    is the weighting factor which applies on the dense scores.
    :param top_k_bm25_nodes: top_k BM25 retrieved nodes.
    :param top_k_bm25_scores: top_k BM25 retrieved nodes' scores.
    :param top_emb_nodes: top_k by embedding retrieved nodes.
    :param top_dense_scores: top_k by embedding retrieved nodes' scores.
    :param alpha: the weighted sum re-ranking factor, applied on the dense or embedding scores.
    :return: top_k nodes, top_k nodes' scores. (by hybrid).
    """
    bm25_hybrid_scores = []
    dense_hybrid_scores = []
    nodes_and_scores = []
    bm25_norm = normalize(top_k_bm25_scores)
    dense_norm = normalize(top_dense_scores)

    for i in range(len(bm25_norm)):
        s1 = (1 - alpha) * bm25_norm[i]
        bm25_hybrid_scores.append(s1)
        d1 = {"node": top_k_bm25_nodes[i], "retriever": "bm25", "hybrid score": s1}
        nodes_and_scores.append(d1)

    for i in range(len(dense_norm)):
        s2 = alpha * dense_norm[i]
        dense_hybrid_scores.append(s2)
        d2 = {"node": top_emb_nodes[i], "retriever": "embedding", "hybrid score": s2}
        nodes_and_scores.append(d2)

    nodes_and_scores_sorted = sorted(nodes_and_scores, key=lambda x: x["hybrid score"], reverse=True)
    final_nodes = [doc["node"] for doc in nodes_and_scores_sorted[:top_k]]
    final_scores = [doc["hybrid score"] for doc in nodes_and_scores_sorted[:top_k]]
    return final_nodes, final_scores



def retrieved_nodes_and_scores(query,alpha=0.6, top_k=5):
    """
    This function gets the query and runs the pipeline to return the top nodes and their scores.
    :param query: the user's input.
    :return: top_k nodes, top_k nodes scores. by running the whole pipeline, these nodes are the hybrid
    retriever nodes.
    """
    nodes = load_nodes_from_pickle()
    subchapters_and_nodes_dict = subchapters_and_nodes(nodes)
    kw_nodes, kw_nodes_scores = keywords_retriever(query, nodes, top_k)
    emb_nodes, emb_nodes_scores = embedding_retriever(query, kw_nodes, subchapters_and_nodes_dict,top_k)
    hyb_nodes, hyb_nodes_scores = hybrid_retriever(kw_nodes, kw_nodes_scores, emb_nodes, emb_nodes_scores,alpha,top_k)
    return hyb_nodes, hyb_nodes_scores