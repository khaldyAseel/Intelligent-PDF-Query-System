# from megaparse.megaparse import MegaParse
# from langchain_openai import ChatOpenAI
# from megaparse.parser.megaparse_vision import MegaParseVision
#
# OPENAI_API = "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"
# model = ChatOpenAI(model="gpt-4", api_key=OPENAI_API)
# parser = MegaParseVision(model=model)
# megaparse = MegaParse(parser)
#
# response = megaparse.load("book.pdf")
# print(response)
# #megaparse.save("./test.md")

from langchain_community.chat_models import ChatOpenAI
from megaparse.parser.unstructured_parser import UnstructuredParser
from megaparse import MegaParse
from megaparse.parser.megaparse_vision import MegaParseVision
import os

# Set your Together AI API key
os.environ["TOGETHER_API_KEY"] = "339d6aae5dd24ebfff4eb075e953efa0c8708bf7aeae13ba455b06f1b879753b"

# Use LLaMA on Together AI
model = ChatOpenAI(
	model_name="togethercomputer/llama-3-70b-instruct",
    openai_api_base="https://api.together.xyz/v1/chat/completions",
    openai_api_key=os.getenv("TOGETHER_API_KEY")
)

# Initialize parser
parser = UnstructuredParser(model=model)
megaparse = MegaParse(parser)

# for chunk_file in os.listdir("./chunks"):
# 	try:
# 		response = megaparse.load(f"./chunks/{chunk_file}")
# 		print(response)
# 	except Exception as e:
# 		print(f"Error processing {chunk_file}: {e}")

# Load and process the document
response = megaparse.load("./chunks/chunk_1.pdf")
print(response)

# Save processed content
megaparse.save("./test.json")
