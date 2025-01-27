from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, VectorStoreIndex
from langchain_pinecone  import PineconeVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Step 1: Set up Pinecone API key and environment
pc = Pinecone(
    api_key="pcsk_Jhg12_6pdd8fhxkVouchXp7i1Ev8y3JffU7VnRjZzuqbazwe3XhxMTKecLVDJKB3hy9aJ")
INDEX_NAME = "pdf-vector-index"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # model dimensions
        metric="cosine",  # model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
pinecone_index = pinecone.Index(INDEX_NAME,"https://pdf-vector-index-wckyov5.svc.aped-4627-b74a.pinecone.io")

# Step 2: Use SentenceTransformers for Embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
embedding_fn = LangchainEmbedding(hf_embedding_model)

# Step 3: Set Up LlamaIndex with Pinecone
vector_store = PineconeVectorStore(pinecone_index)
index = GPTVectorStoreIndex([], vector_store=vector_store, embed_model=embedding_fn)


# Step 4: Extract Text from PDF and Build Index
def extract_text_from_pdf(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def build_index_from_pdf(pdf_path):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Save text to a temporary file
    temp_file = "temp_text.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)

    # Load text and create the index
    documents = SimpleDirectoryReader(input_files=[temp_file]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, embed_model=embedding_fn, vector_store=vector_store)
    return index


# Step 5: Query the Index
def query_index(index, query, top_k=3):
    response = index.query(query, similarity_top_k=top_k)
    return response


# Main workflow
if __name__ == "__main__":
    pdf_path = "book.pdf"
    query_text = "What are the key points of this document?"

    # Build index from PDF
    index = build_index_from_pdf(pdf_path)

    # Query the index
    response = query_index(index, query_text)
    print("Query Results:")
    for result in response.response['matches']:
        print(f"Score: {result.score}, Content: {result.text}")

