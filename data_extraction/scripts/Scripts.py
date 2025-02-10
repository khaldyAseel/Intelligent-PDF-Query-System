from PyPDF2 import PdfReader
import re
import json
import os
import fitz  # PyMuPDF



# Step 1: Parse TOC
def extract_toc(pdf_path):
    """Extracts the Table of Contents from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()

    if not toc:
        return "No structured Table of Contents found."

    toc_list = []
    for level, title, page in toc:
        toc_list.append({"level": level, "title": title, "page": page})

    return toc_list


# Step 2: Extract content for each section
def extract_text_from_pdf_by_range(pdf_path, start_page, end_page):
    reader = PdfReader(pdf_path)
    extracted_text = ""

    for page_num in range(start_page - 1, min(end_page, len(reader.pages))):
        extracted_text += reader.pages[page_num].extract_text() + "\n"

    return extracted_text.strip()


# Step 3: function to extract the title of chapter/subchapter
def sanitize_filename(title):
    """Sanitize filename by replacing invalid characters."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in title)


# Step 4: Save structured data
def save_chapters_to_json(pdf_path, toc, output_dir):
    """Saves chapters, subchapters, and sub-subchapters with proper hierarchy tracking."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parent_stack = []  # Stack to track parent-child relationships

    for i, entry in enumerate(toc):
        start_page = entry["page"]
        end_page = toc[i + 1]["page"] if i + 1 < len(toc) else len(PdfReader(pdf_path).pages)

        # Determine hierarchy level
        level = entry.get("level", 1)

        # Extract text for the section
        content = extract_text_from_pdf_by_range(pdf_path, start_page, end_page)

        # Find the correct parent
        while parent_stack and parent_stack[-1]["level"] >= level:
            parent_stack.pop()  # Remove higher or equal-level parents

        parent_title = parent_stack[-1]["title"] if parent_stack else None

        # Build structured data
        entry_data = {
            "title": entry["title"],
            "content": content,
            "metadata": {
                "page": start_page,
                "type": "chapter" if level == 1 else "subchapter" if level == 2 else "sub-subchapter",
                "parent": parent_title
            }
        }

        # Save to JSON
        filename = os.path.join(output_dir, f"{sanitize_filename(entry['title'])}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entry_data, f, indent=4)

        # Add to parent stack
        parent_stack.append({"title": entry["title"], "level": level})

# Main execution
if __name__ == "__main__":
    pdf_path = "book.pdf"
    output_dir = "parsed_text_output"
    toc_text = extract_toc(pdf_path)
    #print(toc_text)
    save_chapters_to_json(pdf_path, toc_text, output_dir)
    print(f"Extracted data saved in: {output_dir}")
