import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Define the folder path
data_folder = "../../data_extraction/parsed_text_output"

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
    """
    # Step 1: Convert to lowercase
    paragraph = paragraph.lower()

    # Step 2: Remove punctuation
    paragraph = paragraph.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Strip extra whitespace
    paragraph = " ".join(paragraph.split())

    # Step 4: Expand abbreviations
    paragraph = expand_abbreviations(paragraph, abbreviation_dict)

    # Step 5: Tokenize the text
    tokens = word_tokenize(paragraph)

    # Step 6: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 7: Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    cleaned_text = " ".join(tokens)

    return cleaned_text

# Process all json files in myData folder
txt_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

for file_name in txt_files:
    file_path = os.path.join(data_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Processing {file_name}...\n")

    # Clean the content
    processed_content = clean_text(content)

    # Print a preview of cleaned content
    print("Original Content Preview:\n", content[:300])
    print("\nCleaned Content Preview:\n", processed_content[:300])
    print("\n" + "-"*50 + "\n")

