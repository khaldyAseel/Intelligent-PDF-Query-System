from PyPDF2 import PdfReader
import re
import json
import os
import fitz  # PyMuPDF
import datetime

# Step 1: Extract Metadata
def extract_pdf_metadata(pdf_path):
    """Extract metadata from a PDF file."""
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    title = metadata.get("title", "Unknown Title")
    editors = metadata.get("author", "Unknown Editors").split(", ")
    total_pages = len(doc)

    # Read the first few pages for edition, publisher, and publication year
    text_content = ""
    for page_num in range(min(10, total_pages)):  # Read first 10 pages max
        text_content += doc[page_num].get_text() + "\n"

        # Search for edition
        edition_match = re.search(
            r"(\bFirst|\bSecond|\bThird|\bFourth|\bFifth|\bSixth|\bSeventh|\bEighth|\bNinth|\bTenth) Edition",
            text_content, re.IGNORECASE)
        edition = edition_match.group(0) if edition_match else "Unknown Edition"

        # Search for publisher (John Wiley & Sons is common for academic books)
        publisher_match = re.search(r"by ([A-Za-z\s&]+)", text_content)
        publisher = publisher_match.group(1).split("\n")[0] if publisher_match else "Unknown Publisher"

        # Search for publication year (e.g., "© 2017" or "First published in 2017")
        year_match = re.search(r"© (\d{4})|First published in (\d{4})", text_content)
        publication_year = year_match.group(1) if year_match else "Unknown Year"

    return {
        "title": title,
        "editors": editors,
        "edition": edition,
        "publisher": publisher,
        "publication year": publication_year,
        "total pages": total_pages,
        "copyright": "© 2017 John Wiley & Sons Ltd"
    }


# Step 2: Save metadata to JSON
def save_metadata(pdf_path, output_dir):
    """Extracts and saves metadata to a JSON file."""
    metadata = extract_pdf_metadata(pdf_path)

    metadata_file = os.path.join(output_dir, "book_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("📜 Metadata extracted and saved successfully!")
    return metadata

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

# Step 4: Save structured data
def save_chapters_to_json(pdf_path, toc, output_dir,book_metadata):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Structure book-level metadata like a chapter
    book_metadata_entry = {
        "title": book_metadata.get("title", "Unknown Book"),
        "content":book_metadata,
        "metadata": {
            "page": 1,
            "type": "metadata",
            "parent": None
        }
    }

    # Save book metadata as a JSON file
    metadata_filename = os.path.join(output_dir, "book_metadata.json")
    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(book_metadata_entry, f, indent=4)

    # Find the "Glossary" section page number
    glossary_page = next((entry["page"] for entry in toc if "glossary" in entry["title"].lower()), None)

    parent_chapter = None  # Track the last seen chapter (Level 1)

    for i, entry in enumerate(toc):
        level = entry.get("level", 1)
        if level == 1:
            parent_chapter = entry["title"]  # Store the main chapter title
            continue  # Skip saving level 1 chapters

        if level > 2:
            continue  # Skip sub-subchapters (level 3+)

        start_page = entry["page"]
        next_title = toc[i + 1]["title"] if i + 1 < len(toc) else None
        end_page = toc[i + 1]["page"] if i + 1 < len(toc) else glossary_page or len(PdfReader(pdf_path).pages)

        # Extract content for subchapter
        content = extract_text_from_pdf_by_range(pdf_path, start_page, end_page, current_title=entry["title"],
                                                 next_title=next_title)

        # Structure subchapter data
        entry_data = {
            "title": entry["title"],
            "content": content,
            "metadata": {
                "page": start_page,
                "type": "subchapter",
                "parent": parent_chapter , # Attach the last seen chapter
            }
        }

        # Save as JSON
        filename = os.path.join(output_dir, f"{sanitize_filename(entry['title'])}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entry_data, f, indent=4)

    print("Subchapters saved successfully!")


# Main execution
if __name__ == "__main__":
    pdf_path = "book.pdf"
    output_dir = "parsed_text_output"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Extract and save metadata
    metadata = save_metadata(pdf_path, output_dir)
    print("Extracted Metadata:", metadata)
    toc_text = extract_toc(pdf_path)
    save_chapters_to_json(pdf_path, toc_text, output_dir,metadata)
    print(f"Extracted data saved in: {output_dir}")


