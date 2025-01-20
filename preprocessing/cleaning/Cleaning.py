import string


def clean_text(paragraph):
    """
    Cleans the input paragraph by:
    1. Converting text to lowercase
    2. Removing punctuation
    3. Stripping extra whitespace
    """
    # Step 1: Convert to lowercase
    paragraph = paragraph.lower()

    # Step 2: Remove punctuation
    paragraph = paragraph.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Strip extra whitespace
    paragraph = " ".join(paragraph.split())

    return paragraph


# Example usage
text = "  Hello, World! This is a TEST paragraph...   "
cleaned_text = clean_text(text)
print("Original Text:", text)
print("Cleaned Text:", cleaned_text)
