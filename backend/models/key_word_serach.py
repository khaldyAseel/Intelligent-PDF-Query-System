#implementing keyword retriever
#chunking the text using llama-index sentence splitter
#retrieve top k nodes (chunks) using m25 retriever
#pass the top k nodes to llama3.3 llm model as the context (for example)


import os
from dotenv import load_dotenv
from together import Together

from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
import Stemmer


load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

#top k nodes parameter
K=5


#text data as a parameter, replace with the text from the database (the sub chapter)
#also replace the metadata from the database
text = ""
metadata = {"page_number": 5}

#convert to a Document object
document = Document(
    text=text,
    metadata=metadata
)

splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)  # Adjust chunk size as needed
nodes = splitter.get_nodes_from_documents([document])

# Print nodes
for node in nodes:
    print(node.text)
    print(node.metadata)



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

#using the retrieved nodes as the context
context_lst = []
for i in range(K):
    context_lst.append(bm25_retrieved_nodes[i].text)

all_context = "\n".join(context_lst)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
      {"role": "system", "content": "You are a helpful chatbot."},
      {"role": "user", "content": f"Answer the question: {query}. Use only information provided here: {all_context}"},
    ],
)
print(response.choices[0].message.content)