import os
import string
import sqlite3
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the folder path containing your JSON files
data_folder = "../../data_extraction/scripts/parsed_text_output"

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
conn = sqlite3.connect("text_database.db")
cursor = conn.cursor()

# Create a table to store the cleaned text
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,  -- Unique ID for each document (title from JSON)
    content TEXT         -- Cleaned text content
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
    1. Converting text to lowercase
    2. Removing punctuation
    3. Stripping extra whitespace
    4. Expanding abbreviations
    5. Removing stopwords
    6. Lemmatizing words
    7. Removing unwanted Unicode characters
    """
    # Step 1: Convert to lowercase
    paragraph = paragraph.lower()

    # Step 2: Remove punctuation
    paragraph = paragraph.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Strip extra whitespace
    paragraph = " ".join(paragraph.split())

    # Step 4: Expand abbreviations
    paragraph = expand_abbreviations(paragraph, abbreviation_dict)

    # Step 5: Remove unwanted Unicode characters (e.g., \u201325 \u00b0)
    paragraph = paragraph.encode('ascii', 'ignore').decode('ascii')

    # Step 6: Tokenize the text
    tokens = word_tokenize(paragraph)

    # Step 7: Remove stopwords
  #  stop_words = set(stopwords.words('english'))
 #  tokens = [word for word in tokens if word not in stop_words]

    # Step 8: Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    cleaned_text = " ".join(tokens)

    return cleaned_text

def save_to_sqlite(unique_id, cleaned_content):
    """
    Saves the cleaned content to the SQLite database.
    """
    # Insert the text into the database
    cursor.execute("INSERT OR REPLACE INTO documents (id, content) VALUES (?, ?)", (unique_id, cleaned_content))
    conn.commit()

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
        print("-" * 50)


# Process all JSON files in the data folder
json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

for file_name in json_files:
    file_path = os.path.join(data_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load JSON data

    # Extract title and content from JSON
    title = data.get("title", file_name.replace(".json", ""))  # Use title or fallback to file name
    content = data.get("content", "")  # Extract content

    print(f"Processing {file_name}...\n")

    # Clean the content
    processed_content = clean_text(content)

    # Print a preview of cleaned content
    print("Original Content Preview:\n", content[:300])
    print("\nCleaned Content Preview:\n", processed_content[:300])
    print("\n" + "-"*50 + "\n")

    # Save the cleaned content to SQLite using the title as the unique ID
    save_to_sqlite(title, processed_content)

print("All files processed and saved to SQLite database.")

# View the database contents
view_database()

# Close the database connection
conn.close()