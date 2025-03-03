from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document
import pinecone
from PyPDF2 import PdfReader
import os

print("Current working directory:", os.getcwd())

# Step 1: Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Function to parse the layout of the PDF using LlamaIndex
def parse_pdf_layout(text):
    """
    Parse the extracted text into a Document object for LlamaIndex.
    """
    # Wrap the text in a LlamaIndex Document object
    document = Document(text)
    return document

# Step 3: Function to save parsed data into a structured format (chapters, subchapters, etc.)
def save_parsed_data(document):
    """
    Extract and structure data from the Document object.
    This function is currently a placeholder and assumes the document is flat text.
    """
    # In this example, we are treating the document as a single chapter
    structured_data = {
        "Chapter 1": {
            "content": document.text,
            "subchapters": {}
        }
    }
    return structured_data

# Step 4: Function to save the parsed and structured data to Pinecone database
def save_to_pinecone(structured_data, pinecone_index):
    # For each section, store the text in Pinecone with a unique ID (e.g., chapter ID)
    for chapter_title, chapter_data in structured_data.items():
        # Store the chapter
        pinecone_index.upsert(vectors=[(chapter_title, [0.1] * 384, {"content": chapter_data["content"]})])
        for subchapter_title, subchapter_data in chapter_data["subchapters"].items():
            # Store the subchapter
            pinecone_index.upsert(vectors=[(subchapter_title, [0.1] * 384, {"content": subchapter_data["content"]})])

# Step 5: Main function to process the PDF, parse it, save structured data, and upload to Pinecone
def process_pdf_and_save_to_db(pdf_path, pinecone_index):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Parse the text into a Document object
    document = parse_pdf_layout(text)

    # Save the parsed data in a structured format
    structured_data = save_parsed_data(document)

    # Save the structured data to Pinecone
    save_to_pinecone(structured_data, pinecone_index)

    return structured_data

# Main workflow
if __name__ == "__main__":
    pdf_path = "/Users/cleopypavlou/PycharmProjects/Intelligent-PDF-Query-System/data_extraction/scripts/book.pdf"

    # Step 1: Set up Pinecone API key and environment
    pinecone.init(
        api_key="your-pinecone-api-key",  # Replace with your Pinecone API key
        environment="your-environment"  # Replace with your environment (e.g., "us-east1-gcp")
    )
    INDEX_NAME = "pdf-vector-index"

    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=384,  # Embedding vector dimension
            metric="cosine"  # Metric for similarity
        )
    pinecone_index = pinecone.Index(INDEX_NAME)

    # Step 2: Process PDF and save to Pinecone
    structured_data = process_pdf_and_save_to_db(pdf_path, pinecone_index)

    # Print the structured data (for visualization)
    print("Extracted and Structured Data:")
    for chapter_title, chapter_data in structured_data.items():
        print(f"Chapter: {chapter_title}")
        print(f"Content: {chapter_data['content'][:200]}...")  # Print the first 200 characters
