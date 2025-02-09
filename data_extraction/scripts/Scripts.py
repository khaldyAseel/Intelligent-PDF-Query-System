from llama_index.readers.llama_parse import LlamaParse
from PyPDF2 import PdfReader
import re
import json
import os
import unicodedata


# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
	reader = PdfReader(pdf_path)
	text = ""
	for page in reader.pages:
		page_text = page.extract_text()
		if page_text:
			text += page_text + "\n"
	return text.strip()


# Step 2: Parse TOC Manually (Use Provided TOC Text)
def parse_toc(toc_text):
	toc = []
	current_chapter = None
	current_subchapter = None

	lines = toc_text.strip().split("\n")
	chapter_pattern = re.compile(r'^(\d+) (.+), (\d+)$')
	subchapter_pattern = re.compile(r'^(\d+\.\d+) (.+), (\d+)$')
	subsubchapter_pattern = re.compile(r'^(\d+\.\d+\.\d+) (.+), (\d+)$')

	for line in lines:
		line = line.strip()
		chapter_match = chapter_pattern.match(line)
		subchapter_match = subchapter_pattern.match(line)
		subsubchapter_match = subsubchapter_pattern.match(line)

		if chapter_match:
			chapter_number, chapter_title, page = chapter_match.groups()
			current_chapter = {
				"title": f"Chapter {chapter_number}: {chapter_title}",
				"page": int(page),
				"subchapters": []
			}
			toc.append(current_chapter)

		elif subchapter_match and current_chapter:
			sub_number, sub_title, page = subchapter_match.groups()
			current_subchapter = {
				"title": f"Subchapter {sub_number}: {sub_title}",
				"page": int(page),
				"subsubchapters": []
			}
			current_chapter["subchapters"].append(current_subchapter)

		elif subsubchapter_match and current_subchapter:
			subsub_number, subsub_title, page = subsubchapter_match.groups()
			subsubchapter = {
				"title": f"Sub-subchapter {subsub_number}: {subsub_title}",
				"page": int(page)
			}
			current_subchapter["subsubchapters"].append(subsubchapter)

	return toc


# Step 3: Extract content for each section
def extract_text_from_pdf_by_range(pdf_path, start_page, end_page):
	reader = PdfReader(pdf_path)
	extracted_text = ""

	for page_num in range(start_page - 1, min(end_page, len(reader.pages))):
		extracted_text += reader.pages[page_num].extract_text() + "\n"

	return extracted_text.strip()


# function to extract the title of chapter/subchapter
def sanitize_filename(title, max_length=30):
	title = title.replace("\xa0", " ")  # Fix non-breaking spaces
	title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode(
		"ascii")  # Remove accents & symbols
	title = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid characters
	title = title.replace(" ", "_")[:max_length]  # Replace spaces & limit length
	return title


# Step 4: Save structured data
def save_chapters_to_json(pdf_path, toc, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for i, chapter in enumerate(toc):
		start_page = chapter["page"]
		end_page = toc[i + 1]["page"] if i + 1 < len(toc) else len(PdfReader(pdf_path).pages)

		chapter_data = {
			"title": chapter["title"],
			"content": extract_text_from_pdf_by_range(pdf_path, start_page, end_page),
			"metadata": {
				"page": start_page,
				"type": "chapter"
			}
		}

		chapter_filename = os.path.join(output_dir, f"{sanitize_filename(chapter['title'])}.json")
		with open(chapter_filename, "w", encoding="utf-8") as f:
			json.dump(chapter_data, f, indent=4)

		for subchapter in chapter["subchapters"]:
			sub_start_page = subchapter["page"]
			sub_end_page = end_page

			sub_data = {
				"title": subchapter["title"],
				"content": extract_text_from_pdf_by_range(pdf_path, sub_start_page, sub_end_page),
				"metadata": {
					"page": sub_start_page,
					"parent": chapter["title"],
					"type": "subchapter"
				}
			}

			sub_filename = os.path.join(output_dir, f"{sanitize_filename(subchapter['title'])}.json")
			with open(sub_filename, "w", encoding="utf-8") as f:
				json.dump(sub_data, f, indent=4)

			for subsubchapter in subchapter["subsubchapters"]:
				subsub_start_page = subsubchapter["page"]
				subsub_data = {
					"title": subsubchapter["title"],
					"content": extract_text_from_pdf_by_range(pdf_path, subsub_start_page, sub_end_page),
					"metadata": {
						"page": subsub_start_page,
						"parent": subchapter["title"],
						"type": "sub-subchapter"
					}
				}

				subsub_filename = os.path.join(output_dir, f"{sanitize_filename(subsubchapter['title'])}.json")
				with open(subsub_filename, "w", encoding="utf-8") as f:
					json.dump(subsub_data, f, indent=4)


# Main execution
if __name__ == "__main__":
	pdf_path = "data_extraction/Scripts/book.pdf"
	output_dir = "data_extraction/parsed_text_output"
	toc_text = extract_text_from_pdf(pdf_path)
	parsed_toc = parse_toc(toc_text)
	save_chapters_to_json(pdf_path, parsed_toc, output_dir)
	print(f"Extracted data saved in: {output_dir}")
