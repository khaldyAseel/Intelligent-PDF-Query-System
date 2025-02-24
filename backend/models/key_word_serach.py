#implementing keyword retriever
#chunking the text using llama-index sentence splitter
#retrieve top k nodes (chunks) using m25 retriever
#pass the top k nodes to llama3.3 llm model as the context (for example)


import os
from dotenv import load_dotenv
from together import Together
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
import Stemmer


load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

#top k nodes parameter
K=10

file_path = r"book.pdf"
documents = SimpleDirectoryReader(file_path).load_data()
print(len(documents))

splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
print(len(nodes))
print(nodes[0].metadata)
print(nodes[0].text)


question = "What factors contribute to price volatility in the cocoa market?"

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=K,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

bm25_retrieved_nodes = bm25_retriever.retrieve(question)
print(bm25_retrieved_nodes[0].text)

query = question

#using the retrieved nodes as the context
context_lst = []
for i in range(K):
    context_lst.append(bm25_retrieved_nodes[i].text)

responses = []  # List to store responses

for context in context_lst:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": f"Answer the question: {query}. Use only information provided here: {context}"},
        ],
    )

    responses.append(response.choices[0].message.content)  # Store each response

# Print all responses
for i, resp in enumerate(responses):
    print(f"Response {i + 1}: {resp}\n")
