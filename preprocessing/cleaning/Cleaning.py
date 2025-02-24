import os
import json
import string
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data if not already installed
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the folder path containing JSON files
folder_path = "../../data_extraction/scripts/parsed_text_output"

# Connect to SQLite database
conn = sqlite3.connect("text_database.db")
cursor = conn.cursor()

# Create the 'chunks' table if not already created
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,        -- Unique ID for each chunk
    chunks TEXT,                -- The chunk of content
    title TEXT,                 -- The title (id from JSON)
    subchapter TEXT,            -- Subchapter information (if available)
    parent TEXT,                -- Parent metadata
    page INTEGER                -- Page metadata
)
""")
conn.commit()

def expand_abbreviations(text, abbreviation_dict):
    """Expands abbreviations in the text using the provided dictionary."""
    words = text.split()
    expanded_words = [abbreviation_dict.get(word, word) for word in words]
    return " ".join(expanded_words)

def clean_text(paragraph):
    """Cleans input text: lowercase, remove punctuation, expand abbreviations, remove stopwords, and lemmatize."""

    abbreviation_dict = {
        "etc.": "et cetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "vs.": "versus",
        "hr.": "hour",
        "min.": "minute"
    }

    # Convert to lowercase and remove punctuation
    paragraph = paragraph.lower().translate(str.maketrans("", "", string.punctuation))
    paragraph = expand_abbreviations(paragraph, abbreviation_dict)

    # Tokenize
    tokens = word_tokenize(paragraph)

    # Remove stopwords (handling potential missing stopwords issue)
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in tokens])

    return cleaned_text

def chunk_content(content, chunk_size=500):
    """Splits the content into chunks of a specified size (tokens)."""
    tokens = word_tokenize(content)
    chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def save_chunk_to_sqlite(unique_id, chunks, title, subchapter, parent, page):
    """Saves chunks and metadata into the SQLite database."""
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{unique_id}_chunk_{idx+1}"  # Ensure unique chunk ID
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO chunks (id, chunks, title, subchapter, parent, page)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, chunk, title, subchapter, parent, page))
        except sqlite3.IntegrityError:
            print(f"Skipping duplicate entry: {chunk_id}")
    conn.commit()

def process_json_files(folder_path):
    """Reads JSON files, cleans and chunks text, and saves them to the database."""
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the directory!")
        return

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        with open(json_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)

                # Extract metadata from JSON
                doc_id = data.get("id", json_file.replace(".json", ""))
                content = data.get("content", "").strip()
                parent = data.get("parent") if data.get("parent") else "Unknown Parent"
                page = int(data.get("page", 0))
                subchapter = data.get("subchapter", "Unknown Subchapter")

                if not content:
                    print(f"Skipping {json_file}: No content found.")
                    continue

                print(f"Processing {json_file} (ID: {doc_id})...\n")

                # Clean, chunk, and store the content
                cleaned_content = clean_text(content)
                chunks = chunk_content(cleaned_content)
                save_chunk_to_sqlite(doc_id, chunks, doc_id, subchapter, parent, page)

            except json.JSONDecodeError:
                print(f"Error reading {json_file}: Invalid JSON format.")
            except Exception as e:
                print(f"Unexpected error processing {json_file}: {e}")

# Run the function to process JSON files
process_json_files(folder_path)

def view_database():
    """Queries and displays all records from the SQLite database."""
    cursor.execute("SELECT * FROM chunks")
    rows = cursor.fetchall()

    if not rows:
        print("Database is empty. No records found.")
        return

    print("\nViewing Chunks Table:")
    print("-" * 50)
    for row in rows:
        print(f"ID: {row[0]}")
        print(f"Chunks: {row[1]}")
        print(f"Title: {row[2]}")
        print(f"Subchapter: {row[3]}")
        print(f"Parent: {row[4]}")
        print(f"Page: {row[5]}")
        print("-" * 50)

# View the chunks in the database
view_database()

# Close the database connection
conn.close()
