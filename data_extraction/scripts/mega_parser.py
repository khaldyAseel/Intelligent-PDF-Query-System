# from langchain_community.chat_models import ChatOpenAI
# from megaparse.parser.unstructured_parser import UnstructuredParser
# from megaparse import MegaParse
# import os
# # Set your Together AI API key
# os.environ["TOGETHER_API_KEY"] = "339d6aae5dd24ebfff4eb075e953efa0c8708bf7aeae13ba455b06f1b879753b"
#
# # Use LLaMA on Together AI
# model = ChatOpenAI(
# 	model_name="togethercomputer/llama-3-70b-instruct",
#     openai_api_base="https://api.together.xyz/v1/chat/completions",
#     openai_api_key=os.getenv("TOGETHER_API_KEY")
# )
#
# # Initialize parser
# parser = UnstructuredParser(model=model)
# megaparse = MegaParse(parser)
#
# # Create the output directory if it doesn't exist
# output_dir = "./chunks"
# os.makedirs(output_dir, exist_ok=True)
#
# # Split the PDF into smaller chunks
# def split_pdf(input_pdf, output_dir, chunk_size=10):
#     import fitz  # PyMuPDF
#     doc = fitz.open(input_pdf)
#     for i in range(0, len(doc), chunk_size):
#         chunk = fitz.open()
#         chunk.insert_pdf(doc, from_page=i, to_page=min(i + chunk_size - 1, len(doc) - 1))
#         chunk.save(f"{output_dir}/chunk_{i // chunk_size + 1}.pdf")
#         chunk.close()
#
# # Process each chunk
# split_pdf("book.pdf", output_dir)
# for chunk_file in os.listdir(output_dir):
#     try:
#         print(f"Processing {chunk_file}...")
#         # Use the parser's `convert` method to process the document
#         response = parser.convert(f"{output_dir}/{chunk_file}")
#         print(response)
#     except Exception as e:
#         print(f"Error processing {chunk_file}: {e}")
#         continue  # Skip to the next chunk

# Save the final output (if needed)
# megaparse.save("./test.md")

import os
from langchain_openai import ChatOpenAI
from megaparse.parser.megaparse_vision import MegaParseVision
from langchain import OpenAI

# Initialize the MegaParseVision model for text and visual content

# Initialize MegaParseVision for parsing the document (you could use MegaParse if you're not using vision)
parser = MegaParseVision(model=model)

# Load the content from the PDF and convert it to Markdown
response = parser.convert("./book.pdf")

# Save the extracted content into a Markdown file
with open("book.md", "w") as file:
    file.write(response)

# Now, pass the extracted content to GPT-4 (or another model) to identify chapters/sub-chapters
chat_model = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API"))

# Ask the model to recognize and structure the content into chapters and sub-chapters
prompt = """
The content below contains chapters, sub-chapters, and text. Please help me identify and organize the content by splitting it into distinct sub-chapters, and return the result.

Content:
{extracted_content}
"""

# Load the content from the markdown file
with open("book.md", "r") as file:
    extracted_content = file.read()

# Generate a structured version of the content
response = chat_model.chat([{"role": "system", "content": prompt.format(extracted_content=extracted_content)}])

# Save the structured content into sub-chapters (or print it)
structured_content = response['message']['content']
with open("structured_book.md", "w") as file:
    file.write(structured_content)

