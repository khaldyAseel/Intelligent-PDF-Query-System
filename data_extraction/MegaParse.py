import os
import fitz  # PyMuPDF
import re
from dotenv import load_dotenv
from megaparse.parser.megaparse_vision import MegaParseVision
from langchain_openai import ChatOpenAI


def split_pdf(input_pdf, output_dir, chunk_size=2):  # Reduced to 2 pages per chunk
    """Splits the input PDF into smaller chunks."""
    doc = fitz.open(input_pdf)

    # Ensure chunks directory is cleared before writing new files
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    for i in range(0, len(doc), chunk_size):
        chunk = fitz.open()
        chunk.insert_pdf(doc, from_page=i, to_page=min(i + chunk_size - 1, len(doc) - 1))

        chunk_filename = os.path.join(output_dir, f"chunk_{i // chunk_size + 1}.pdf")

        # Ensure the file is writable
        if os.path.exists(chunk_filename):
            os.remove(chunk_filename)

        chunk.save(chunk_filename)
        chunk.close()

    print(f"✅ PDF split into smaller chunks and saved in: {output_dir}")


def extract_text_with_pymupdf(pdf_path):
    """Alternative text extraction using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()


def numeric_sort(filename):
    """Ensures chunks are processed in correct order."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


if __name__ == "__main__":
    # Load OpenAI API key
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Ensure the API key is available
    if not openai_api_key:
        raise ValueError("❌ ERROR: OpenAI API key is missing! Please add it to the .env file.")

    # Initialize MegaParse Vision with OpenAI GPT-4o
    model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
    parser = MegaParseVision(model=model)

    # File paths
    input_pdf = r"data_extraction\scripts\book.pdf"
    output_dir = "./chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Split the PDF
    split_pdf(input_pdf, output_dir)

    all_responses = []

    with open("./Data.md", "w", encoding="utf-8") as file:
        sorted_files = sorted(os.listdir(output_dir), key=numeric_sort)  # Ensure correct order

        for chunk_file in sorted_files:
            chunk_path = os.path.join(output_dir, chunk_file)
            print(f"🔄 Processing: {chunk_path}")

            try:
                response = parser.convert(chunk_path)  # Use MegaParse Vision

                if response and response.strip():
                    print(f"✅ Extracted text from {chunk_file} (MegaParse Vision):\n{response[:500]}...")
                    file.write(response + "\n\n")
                else:
                    print(f"⚠️ Warning: MegaParse Vision extracted no text from {chunk_file}. Trying PyMuPDF...")

                    # Try extracting with PyMuPDF
                    fallback_response = extract_text_with_pymupdf(chunk_path)
                    if fallback_response:
                        print(f"✅ Extracted text using PyMuPDF for {chunk_file}:\n{fallback_response[:500]}...")
                        file.write(fallback_response + "\n\n")
                    else:
                        print(f"❌ No text found in {chunk_file} even with PyMuPDF.")

            except Exception as e:
                print(f"❌ Error processing {chunk_file}: {e}")
