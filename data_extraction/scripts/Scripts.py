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
def extract_text_from_pdf_by_range(pdf_path, start_page, end_page, current_title=None, next_title=None):
    reader = PdfReader(pdf_path)
    extracted_text = ""

    for page_num in range(start_page - 1, min(end_page, len(reader.pages))):
        page_text = reader.pages[page_num].extract_text()

        if not page_text:
            continue  # Skip empty pages

        # If this is the first page, start from the title
        if page_num == start_page - 1 and current_title:
            title_match = re.search(re.escape(current_title), page_text)
            if title_match:
                page_text = page_text[title_match.end():]  # Remove everything before the title

        # Stop extracting when the next title appears
        if next_title:
            next_match = re.search(re.escape(next_title), page_text)
            if next_match:
                extracted_text += page_text[:next_match.start()]  # Keep only text before next title
                break  # Stop processing further pages
            else:
                extracted_text += page_text + "\n"
        else:
            extracted_text += page_text + "\n"

    return extracted_text.strip()

# Step 3: function to extract the title of chapter/subchapter
def sanitize_filename(title):
    """Sanitize filename by replacing invalid characters."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in title)

def extract_chapters_from_pdf_by_range(pdf_path, start_page, end_page, current_title=None, next_title=None):
    """Extracts text starting from a chapter title and stops at the next chapter title."""
    reader = PdfReader(pdf_path)
    extracted_text = ""

    for page_num in range(start_page - 1, min(end_page, len(reader.pages))):
        page_text = reader.pages[page_num].extract_text()

        if not page_text:
            continue  # Skip empty pages

        # Start from chapter title
        if page_num == start_page - 1 and current_title:
            title_match = re.search(re.escape(current_title), page_text)
            if title_match:
                page_text = page_text[title_match.end():]  # Remove everything before the title

        # Stop at next chapter title
        if next_title:
            next_match = re.search(re.escape(next_title), page_text)
            if next_match:
                extracted_text += page_text[:next_match.start()]
                break
            else:
                extracted_text += page_text + "\n"
        else:
            extracted_text += page_text + "\n"

    return extracted_text.strip()

# extract chapters only
def extract_chapters_from_pdf(pdf_path, toc, output_dir):
    """Extracts and saves only level 1 chapters (ignores subchapters) and stops at 'Glossary'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter only level 1 chapters
    chapters = [entry for entry in toc if entry["level"] == 1]

    # Find the "Glossary" section in TOC
    glossary_page = next((entry["page"] for entry in toc if "glossary" in entry["title"].lower()), None)

    for i, chapter in enumerate(chapters):
        start_page = chapter["page"]

        if i + 1 < len(chapters):
            end_page = chapters[i + 1]["page"]  # Stop at next chapter
        elif glossary_page:
            end_page = glossary_page  # Stop at Glossary if it exists
        else:
            end_page = len(PdfReader(pdf_path).pages)  # Fallback to last page

        next_chapter_title = chapters[i + 1]["title"] if i + 1 < len(chapters) else None

        # Extract text for this chapter
        content = extract_text_from_pdf_by_range(pdf_path, start_page, end_page, chapter["title"], next_chapter_title)

        # Save chapter to JSON
        chapter_data = {
            "title": chapter["title"],
            "content": content,
            "metadata": {
                "page": start_page,
                "type": "chapter"
            }
        }

        filename = os.path.join(output_dir, f"{sanitize_filename(chapter['title'])}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chapter_data, f, indent=4)

    print(f"Extracted {len(chapters)} chapters successfully!")



# Step 4: Save structured data
def save_chapters_to_json(pdf_path, toc, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parent_stack = []

    # Find the "Glossary" section page number
    glossary_page = next((entry["page"] for entry in toc if "glossary" in entry["title"].lower()), None)

    for i, entry in enumerate(toc):
        start_page = entry["page"]

        # Determine the end page
        if i + 1 < len(toc):
            end_page = toc[i + 1]["page"]  # Stop at the next chapter
        elif glossary_page:
            end_page = glossary_page  # Stop at "Glossary"
        else:
            end_page = len(PdfReader(pdf_path).pages)  # Fallback: Last page

        next_title = toc[i + 1]["title"] if i + 1 < len(toc) else None
        level = entry.get("level", 1)

        # Handle hierarchy levels
        while parent_stack and parent_stack[-1]["level"] >= level:
            parent_stack.pop()

        parent_title = parent_stack[-1]["title"] if parent_stack else None

        # Extract chapter content
        content = extract_text_from_pdf_by_range(pdf_path, start_page, end_page, current_title=entry["title"], next_title=next_title)

        # Structure chapter data
        entry_data = {
            "title": entry["title"],
            "content": content,
            "metadata": {
                "page": start_page,
                "type": "chapter" if level == 1 else "subchapter" if level == 2 else "sub-subchapter",
                "parent": parent_title
            }
        }

        # Save as JSON
        filename = os.path.join(output_dir, f"{sanitize_filename(entry['title'])}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entry_data, f, indent=4)

        parent_stack.append({"title": entry["title"], "level": level})

    print("Chapters saved successfully!")



# Main execution
if __name__ == "__main__":
    pdf_path = "book.pdf"
    output_dir = "parsed_text_output"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    toc_text = extract_toc(pdf_path)
    # extract_chapters_from_pdf(pdf_path,toc_text,output_dir)
    save_chapters_to_json(pdf_path, toc_text, output_dir)
    print(f"Extracted data saved in: {output_dir}")
