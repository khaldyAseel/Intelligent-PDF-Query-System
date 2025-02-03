import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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


# Example usage
text = "  The temp of the solution was measured at 25°C, and the visc was recorded at 50 rpm.   "
cleaned_text = clean_text(text)
print("Original Text:", text)
print("Cleaned Text:", cleaned_text)