# import os
# import re
# from openai import OpenAI
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv(dotenv_path="../../backend/api/.env")
# openai_api_key = os.getenv("OPENAI_API_KEY")
#
# # Set up OpenAI client
# client = OpenAI(api_key=openai_api_key)
#
# # Directory for parsed subchapters
# output_dir = "parsed_book_md"
# os.makedirs(output_dir, exist_ok=True)
#
# # Read parsed book content from book.md
# with open("book.md", "r", encoding="utf-8") as file:
# 	extracted_content = file.read()
#
#
# # Function to chunk text (if needed)
# def chunk_text(text, max_length=4000):
# 	"""Splits long text into smaller chunks to fit token limits."""
# 	return [text[i: i + max_length] for i in range(0, len(text), max_length)]
#
#
# chunks = chunk_text(extracted_content)
#
# # GPT-4 prompt to extract structured subchapters
# system_message = {
# 	"role": "system",
# 	"content": "Extract and organize the book into subchapters only. "
# 			   "Each subchapter should have a title, body text, start page, and parent chapter."
# }
#
# structured_content = []
# for chunk in chunks:
# 	response = client.chat.completions.create(
# 		model="gpt-4-turbo",
# 		messages=[system_message, {"role": "user", "content": chunk}]
# 	)
#
# 	structured_content.extend(response.choices[0].message.content)
#
#
# # Function to sanitize filenames
# def sanitize_filename(title):
# 	"""Removes invalid characters and formats filenames."""
# 	title = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove forbidden characters
# 	title = title.replace(' ', '_')  # Replace spaces with underscores
# 	return title + ".md"
#
#
# # Process structured content and save subchapters
# for subchapter in structured_content:
# 	subchapter_title = subchapter["title"]
# 	subchapter_text = subchapter["text"]
# 	start_page = subchapter["start_page"]
# 	parent_chapter = subchapter["parent_chapter"]
#
# 	# Prepare markdown content
# 	md_content = f"# {subchapter_title}\n\n{subchapter_text}\n\n---\n**Start Page:** {start_page}\n\n**Parent Chapter:** {parent_chapter}\n"
#
# 	# Generate valid filename
# 	filename = sanitize_filename(subchapter_title)
# 	filepath = os.path.join(output_dir, filename)
#
# 	# Save to Markdown file
# 	with open(filepath, "w", encoding="utf-8") as file:
# 		file.write(md_content)
#
# 	print(f"Saved: {filepath}")
#
# print("✅ All subchapters saved successfully!")

from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="book_1.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="chunks",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()