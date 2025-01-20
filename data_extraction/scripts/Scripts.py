# data extraction
from llama_index.readers import PyMuPDFReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import os

def extract_text_from_pdf(pdf_file_path):
    """
    Extracts text from a single PDF file using PyMuPDFReader.

    Args:
        pdf_file_path (str): Path to the PDF file.

    Returns:
        list: A list of Document objects, where each Document contains the extracted text.
    """
    # Initialize the PyMuPDF reader
    pdf_reader = PyMuPDFReader()

    # Load data from the PDF file
    documents = pdf_reader.load_data(pdf_file_path)

    return documents

# Example usage:
pdf_file_path = "Becketts_Industrial_Chocolate_Manufacture_and_Use.pdf"
documents = extract_text_from_pdf(pdf_file_path)

# Print the content of each document
for idx, doc in enumerate(documents):
    print(f"Document {idx + 1}:")
    print(doc.get_text())  # Access the text content of each document
    print("\n" + "="*50 + "\n")


def parse_layout(documents):
    """Parse document into hierarchical layout (chapters, subchapters)."""
    parsed_data = []
    for doc in documents:
        text = doc.text
        # Example logic for layout parsing
        if "Chapter" in text:
            parsed_data.append({"type": "chapter", "content": text, "page_number": doc.metadata.get("page_number", 0)})
        elif "Section" in text:
            parsed_data.append({"type": "subchapter", "content": text, "page_number": doc.metadata.get("page_number", 0)})
        else:
            parsed_data.append({"type": "content", "content": text, "page_number": doc.metadata.get("page_number", 0)})
    return parsed_data

def preprocess_and_clean(parsed_data):
    """Normalize and clean text."""
    clean_data = []
    for item in parsed_data:
        cleaned_text = item["content"].replace("\n", " ").replace("\t", " ").strip().lower()
        clean_data.append({
            "type": item["type"],
            "content": cleaned_text,
            "page_number": item["page_number"]
        })
    return clean_data

def save_to_faiss(clean_data, faiss_index_path="faiss_indexes/faiss_index"):
    """Save cleaned data into a FAISS vector database."""
    # Create embeddings
    embedding_model = OpenAIEmbeddings()

    # Prepare data for FAISS
    texts = [data["content"] for data in clean_data]
    metadatas = [{"type": data["type"], "page_number": data["page_number"]} for data in clean_data]

    # Create FAISS vector store
    vector_store = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

    # Save FAISS index to disk
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    vector_store.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")

def main():
    # Path to your PDF file
    pdf_path = "your_pdf_book.pdf"

    # Step 1: Extract text
    documents = extract_text_from_pdf(pdf_path)

    # Step 2: Parse layout
    parsed_data = parse_layout(documents)

    # Step 3: Preprocess and clean
    clean_data = preprocess_and_clean(parsed_data)

    # Step 4: Save to FAISS
    save_to_faiss(clean_data)

if __name__ == "__main__":
    main()
