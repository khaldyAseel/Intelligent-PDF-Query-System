# from megaparse.megaparse import MegaParse
# from megaparse.parser.llama import LlamaParser
# import os
#
# lLAMA_CLOUD_API = "llx-NuJxG1FwEceSYPdTWmRGoKrBHBoxxzjnmRQaTeGdQUt4HKq8"
# # Initialize parser and MegaParse
# parser = LlamaParser(api_key=lLAMA_CLOUD_API)
# megaparse = MegaParse(parser)

import os
import fitz  # PyMuPDF
from megaparse import MegaParse

def split_pdf(input_pdf, output_dir, chunk_size=10):
    """Split the PDF into smaller chunks."""
    doc = fitz.open(input_pdf)

    for i in range(0, len(doc), chunk_size):
        chunk = fitz.open()
        chunk.insert_pdf(doc, from_page=i, to_page=min(i + chunk_size - 1, len(doc) - 1))
        chunk_filename = f"{output_dir}/chunk_{i // chunk_size + 1}.pdf"
        chunk.save(chunk_filename)
        chunk.close()

    print(f"PDF split into chunks and saved in: {output_dir}")


def process_chunks(output_dir, output_file_prefix, max_file_size_mb=100):
    """Process each chunk and write the output to Markdown files."""
    megaparse = MegaParse()
    file_index = 1
    current_file_size = 0
    output_file = f"{output_file_prefix}_{file_index}.md"

    # Open the first output file
    with open(output_file, "w") as file:
        for chunk_file in sorted(os.listdir(output_dir)):
            chunk_path = os.path.join(output_dir, chunk_file)
            print(f"Processing: {chunk_path}")

            try:
                # Load the chunk using MegaParse
                response = megaparse.load(chunk_path)
                if response:
                    # Check if the current file size exceeds the limit
                    if current_file_size > max_file_size_mb * 1024 * 1024:  # Convert MB to bytes
                        file.close()
                        file_index += 1
                        output_file = f"{output_file_prefix}_{file_index}.md"
                        file = open(output_file, "w")
                        current_file_size = 0

                    # Write the response to the file
                    file.write(response + "\n\n")
                    current_file_size += len(response.encode("utf-8"))  # Track file size in bytes
                    print(f"Written response for {chunk_file} to {output_file}")
                else:
                    print(f"Warning: Empty response for {chunk_file}")
            except Exception as e:
                print(f"Error processing {chunk_file}: {e}")

    print(f"All chunks processed. Output saved to {output_file_prefix}_*.md")


if __name__ == "__main__":
    input_pdf = "book.pdf"
    output_dir = ".\chunks"
    output_file_prefix = "./Data"  # Prefix for output Markdown files

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Split the PDF
    # split_pdf(input_pdf, output_dir)

    # Step 2: Process each chunk and write to Markdown files
    # process_chunks(output_dir, output_file_prefix, max_file_size_mb=100)  # Adjust max_file_size_mb as needed
    megaparse = MegaParse()
    response = megaparse.load("./chunks/chunk_2.pdf")
    print(response)
