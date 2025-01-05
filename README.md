# Intelligent PDF Query System

## Project Overview
This project builds a system that can extract data from a PDF book and allow users to ask questions, such as, “What is the origin of cacao?” The system retrieves and answers questions based on the extracted and processed content of the PDF. The system leverages embeddings, a pre-trained language model (e.g., Mistral-7B), and an efficient database for storing and querying the processed data.

---

## Features
- Extract text from PDF files using Python tools.
- Pre-process and clean the extracted data for consistency and usability.
- Generate embeddings for semantic similarity using Sentence-Transformers.
- Store embeddings in a database or file for efficient querying.
- Use a pre-trained language model (Mistral-7B) to answer user questions.
- Retrieve and rank relevant data using semantic search.

---

## Workflow
### 1. **PDF Text Extraction**
   - Tools: PyPDF2, pdfplumber, or PyMuPDF (fitz).
   - Extract textual content, images, or tables from the PDF.

### 2. **Data Pre-Processing**
   - Cleaning steps: remove irrelevant text, fix OCR errors, tokenize, lemmatize, and normalize.
   - Categorize content (e.g., paragraphs, tables, images) for structured storage.

### 3. **Embedding Creation**
   - Tool: Sentence-Transformers.
   - Generate high-dimensional vector embeddings to capture semantic meaning.

### 4. **Embedding Storage**
   - Store embeddings in a database (e.g., MySQL or PostgreSQL) or as .parquet files for easy access.

### 5. **Question Answering**
   - Use a pre-trained language model (e.g., Mistral-7B) to answer questions.
   - Perform semantic search to find the most relevant sections of the PDF based on user queries.

### 6. **Evaluation**
   - Metrics: Accuracy, F1-Score, Exact Match, and BLEU.
   - Ensure the system retrieves and ranks answers effectively.

---

## Tools and Libraries
- **PDF Text Extraction**: PyPDF2, pdfplumber, PyMuPDF.
- **Data Processing**: NLTK, spaCy, or custom Python scripts.
- **Embedding Generation**: Sentence-Transformers.
- **Database**: MySQL, PostgreSQL, or .parquet files.
- **Language Model**: Mistral-7B (locally hosted or via Hugging Face).
- **Evaluation**: scikit-learn, NLTK, or custom metrics implementation.


