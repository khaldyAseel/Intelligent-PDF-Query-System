from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, Document
import pinecone
from llama_index.readers.llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec
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


# Step 2: Function to parse the layout of the PDF using LlamaParser
def parse_pdf_layout(text):
# Instantiate the LlamaParser
    llama_parser = LlamaParse(api_key="llx-r1MMT3h23EfJxwhDsigRJaWSYLfj6yDP7p84wAxZ0KfjiaWf")
# Parse the extracted text (assumes the text has some identifiable layout structure)
    parsed_data = llama_parser.load_data(text)
# The parsed data will be structured into chapters, subchapters, etc.
    return parsed_data

# Step 3: Function to save parsed data into a structured format (chapters, subchapters, etc.)
def save_parsed_data(parsed_data):
	structured_data = {}
	# Assuming parsed_data is in a nested structure, we can loop through and save it
	for chapter in parsed_data:
		chapter_title = chapter['title']
		chapter_content = chapter['content']
		# Save chapters, subchapters, and sub-subchapters in a nested dictionary
		structured_data[chapter_title] = {
			"content": chapter_content,
			"subchapters": {}
		}
		for subchapter in chapter.get('subchapters', []):
			subchapter_title = subchapter['title']
			subchapter_content = subchapter['content']
			structured_data[chapter_title]["subchapters"][subchapter_title] = {
				"content": subchapter_content,
				"subsubchapters": {}
			}
			for subsubchapter in subchapter.get('subsubchapters', []):
				subsubchapter_title = subsubchapter['title']
				subsubchapter_content = subsubchapter['content']
				structured_data[chapter_title]["subchapters"][subchapter_title]["subsubchapters"][
					subsubchapter_title] = subsubchapter_content

	return structured_data


# Step 4: Function to save the parsed and structured data to Pinecone database
def save_to_pinecone(structured_data, pinecone_index):
    # For each section, store the text in Pinecone with a unique ID (e.g., chapter ID)
    for chapter_title, chapter_data in structured_data.items():
        # Store the chapter
        pinecone_index.upsert(vectors=[(chapter_title, chapter_data['content'])])
        for subchapter_title, subchapter_data in chapter_data["subchapters"].items():
            # Store the subchapter
            pinecone_index.upsert(vectors=[(subchapter_title, subchapter_data['content'])])
            for subsubchapter_title, subsubchapter_content in subchapter_data["subsubchapters"].items():
                # Store the sub-subchapter
                pinecone_index.upsert(vectors=[(subsubchapter_title, subsubchapter_content)])


# Step 5: Main function to process the PDF, parse it, save structured data, and upload to Pinecone
def process_pdf_and_save_to_db(pdf_path, pinecone_index):
	# Extract text from the PDF
	text = extract_text_from_pdf(pdf_path)
	# Parse the text using layout parsing method
	parsed_data = parse_pdf_layout(text)
	# Save the parsed data in a structured format
	structured_data = save_parsed_data(parsed_data)
	# Save the structured data to Pinecone
	save_to_pinecone(structured_data, pinecone_index)

	return structured_data


# Main workflow
if __name__ == "__main__":
	pdf_path = "data_extraction/Scripts/book.pdf"
# Step 1: Set up Pinecone API key and environment
	pc = Pinecone(
		api_key="pcsk_Jhg12_6pdd8fhxkVouchXp7i1Ev8y3JffU7VnRjZzuqbazwe3XhxMTKecLVDJKB3hy9aJ")
	INDEX_NAME = "pdf-vector-index"

	if INDEX_NAME not in pc.list_indexes().names():
		pc.create_index(
			name=INDEX_NAME,
			dimension=384,  # Model dimensions (though not used in this case)
			metric="cosine",  # Model metric
			spec=ServerlessSpec(
				cloud="aws",
				region="us-east-1"
			)
		)
	pinecone_index = pinecone.Index(INDEX_NAME, "https://pdf-vector-index-wckyov5.svc.aped-4627-b74a.pinecone.io")

	# Step 2: Process PDF and save to Pinecone
	structured_data = process_pdf_and_save_to_db(pdf_path, pinecone_index)

	# Print the structured data (for visualization)
    # Print the first chapter, subchapter, and sub-subchapter

	print("Extracted and Structured Data:")
   
	# for chapter_title, chapter_data in structured_data.items():
	# 	print(f"Chapter: {chapter_title}")
	# 	print(f"Content: {chapter_data['content']}")
    #
	# 	for subchapter_title, subchapter_data in chapter_data["subchapters"].items():
	# 		print(f"  Subchapter: {subchapter_title}")
	# 		print(f"  Content: {subchapter_data['content']}")
    #
	# 		for subsubchapter_title, subsubchapter_content in subchapter_data["subsubchapters"].items():
	# 			print(f"    Sub-subchapter: {subsubchapter_title}")
	# 			print(f"    Content: {subsubchapter_content}")
