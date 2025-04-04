import os
import string
import sqlite3
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the folder path containing your JSON files
data_folder = "../../data_extraction/scripts/parsed_text_output"
if not os.path.exists(data_folder):
    print("Error: Folder does not exist!")

# Abbreviation dictionary
abbreviation_dict = {
    "temp": "temperature",
    "visc": "viscosity",
    "haccp": "Hazard Analysis Critical Control Point",
    "cfu": "colony-forming units",
    "ppm": "parts per million",
    "rpm": "revolutions per minute",
    "ph": "potential of hydrogen"
}

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("../../backend/database/text_database.db")
cursor = conn.cursor()

# Create a table to store the cleaned text and metadata
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,  -- Unique ID for each document (title from JSON)
    content TEXT,        -- Cleaned text content
    parent TEXT,         -- Parent metadata
    page INTEGER,        -- Page metadata
    type TEXT            -- type metadata 
)
""")

cursor.execute("""
        CREATE TABLE IF NOT EXISTS BookMetadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            edition TEXT,
            editors TEXT,
            publication_year TEXT,
            publisher TEXT,
            total_pages INTEGER,
            copyright TEXT
        )
    """)
conn.commit()

def expand_abbreviations(text, abbreviation_dict):
    """
    Expands abbreviations in the text using the provided dictionary.
    """
    words = text.split()
    expanded_words = [abbreviation_dict.get(word, word) for word in words]
    expanded_text = " ".join(expanded_words)
    return expanded_text

def clean_text(paragraph):
    """
    Cleans the input paragraph by:
    1. Removing figure and diagram references
    2. Converting text to lowercase
    3. Removing punctuation
    4. Stripping extra whitespace
    5. Expanding abbreviations
    6. Removing stopwords
    7. Lemmatizing words
    8. Removing unwanted Unicode characters
    """
    # Step 1: Remove unwanted references (e.g., "see Figure 2.5", "see diagram")
    paragraph = re.sub(r'\(see\s(?:Figure|Diagram|Table)?\s?\d*\.?\d*\)', '', paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r'\bsee\s(chapter|section|table)?\s?\d+\.?\d*\b', '', paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r'\b(given|refer to|as shown in|as seen in)\s(table|figure|diagram|section)?\s?\d+\.?\d*\b', '',
                       paragraph, flags=re.IGNORECASE)

    # Step 2: Convert to lowercase
    paragraph = paragraph.lower()

    # Step 3: Remove punctuation
    paragraph = paragraph.translate(str.maketrans("", "", string.punctuation))

    # Step 4: Strip extra whitespace
    paragraph = " ".join(paragraph.split())

    # Step 5: Expand abbreviations
    paragraph = expand_abbreviations(paragraph, abbreviation_dict)

    # Step 6: Remove unwanted Unicode characters (e.g., \u201325 \u00b0)
    paragraph = paragraph.encode('ascii', 'ignore').decode('ascii')

    # Step 7: Tokenize the text
    tokens = word_tokenize(paragraph)

    # Step 8: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 9: Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    cleaned_text = " ".join(tokens)

    return cleaned_text

def save_to_sqlite(unique_id, cleaned_content, metadata):
    """
    Saves the cleaned content and metadata to the SQLite database.
    """
    # Insert the text and metadata into the database
    cursor.execute("""
            INSERT OR REPLACE INTO documents (id, content, parent, page, type)
            VALUES (?, ?, ?, ?, ?)
        """, (unique_id, cleaned_content, metadata.get("parent"), metadata.get("page"),metadata.get("type")))

    conn.commit()

# Process all JSON files in the data folder
def process_json_files():
    json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

    for file_name in json_files:
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load JSON data

        # Extract title, content, and metadata from JSON
        title = data.get("title", file_name.replace(".json", ""))  # Use title or fallback to file name
        content = data.get("content", "")  # Extract content
        metadata = data.get("metadata", {})  # Extract metadata

        print(f"Processing {file_name}...\n")

        # 🔹 Skip cleaning if the file is **metadata**
        if metadata.get("type") == "metadata":
            processed_content = content  # No cleaning for metadata
            processed_content = json.dumps(processed_content, ensure_ascii=False)
        else:
            processed_content = clean_text(content)  # Clean only if not metadata

        # Print a preview of cleaned content
        print("Original Content Preview:\n", content)
        print("\nCleaned Content Preview:\n", processed_content)
        print("\n" + "-"*50 + "\n")


        save_to_sqlite(title, processed_content, metadata)

    print("All files processed and saved to SQLite database.")

def save_metadata_to_db( metadata_file):
    """Loads book metadata from JSON and saves it to the database."""
    # Load metadata from JSON file
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Convert editors list to a comma-separated string
    editors_str = ", ".join(metadata.get("editors", []))

    cursor.execute("""
        INSERT INTO BookMetadata 
        (title, edition, editors, publication_year, publisher, total_pages, copyright)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        metadata.get("title", "Unknown Title"),
        metadata.get("edition", "Unknown Edition"),
        editors_str,
        metadata.get("publication_year", "Unknown Year"),
        metadata.get("publisher", "Unknown Publisher"),
        metadata.get("total_pages", 0),
        metadata.get("copyright", "Unknown Copyright")
    ))
    conn.commit()
    return cursor.lastrowid  # Return the book_id of the inserted record

def view_database():
    """
    Queries and displays all records from the SQLite database.
    """
    # Query all documents
    cursor.execute("SELECT * FROM documents")
    rows = cursor.fetchall()

    # Print the results
    print("\nViewing Database Contents:")
    print("-" * 50)
    for row in rows:
        print(f"ID: {row[0]}")
        print(f"Content: {row[1]}")
        print(f"Parent: {row[2]}")
        print(f"Page: {row[3]}")
        print(f"type:{row[4]}")
        print("-" * 50)

def view_metadata_table():
    """Fetches and displays all records from the BookMetadata table."""
    cursor.execute("SELECT * FROM BookMetadata")
    rows = cursor.fetchall()

    if not rows:
        print("📭 No metadata records found.")
    else:
        print("\n📖 Book Metadata Table:\n")
        for row in rows:
            print(f"ID: {row[0]}")
            print(f"Title: {row[1]}")
            print(f"Edition: {row[2]}")
            print(f"Editors: {row[3]}")
            print(f"Publication Year: {row[4]}")
            print(f"Publisher: {row[5]}")
            print(f"Total Pages: {row[6]}")
            print(f"Copyright: {row[7]}")
            print("-" * 50)

process_json_files()
save_metadata_to_db(metadata_file="../../data_extraction/scripts/parsed_text_output/book_metadata.json")
# View the database contents
view_database()
# Close the database connection
conn.close()
