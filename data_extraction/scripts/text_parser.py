# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

# bring in deps
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader

LLAMA_CLOUD_API_KEY = "llx-HXPXlm5mea4nEbeJarG7oHBWecoBZst3bWWgEgSWbmzsuWdH"

# set up parser
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown"  # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['book.pdf'], file_extractor=file_extractor).load_data()
print(documents)