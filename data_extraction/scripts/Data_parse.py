# bring in deps
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
import  os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../backend/api/.env")

api_key = os.getenv("lLAMA_CLOUD_API")
# set up parser
parser = LlamaParse(
    api_key= api_key,
    result_type="markdown"  # "markdown" and "text" are available
)
# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['book.pdf'], file_extractor=file_extractor).load_data()
# Extract text from documents
extracted_content = "\n".join([doc.text for doc in documents])
# Save extracted content to a Markdown file
with open("book.md", "w", encoding="utf-8") as file:
    file.write(extracted_content)
print("PDF parsed and saved as 'book.md'.")
