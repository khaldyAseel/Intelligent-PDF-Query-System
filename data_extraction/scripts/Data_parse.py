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
	doc = fitz.open(input_pdf)

	for i in range(0, len(doc), chunk_size):
		chunk = fitz.open()
		chunk.insert_pdf(doc, from_page=i, to_page=min(i + chunk_size - 1, len(doc) - 1))
		chunk_filename = f"{output_dir}/chunk_{i // chunk_size + 1}.pdf"
		chunk.save(chunk_filename)
		chunk.close()

	print(f"PDF split into chunks and saved in: {output_dir}")


if __name__ == "__main__":
	input_pdf = "book.pdf"
	output_dir = ".\chunks"
	os.makedirs(output_dir, exist_ok=True)

	# Step 1: Split the PDF
	split_pdf(input_pdf, output_dir)

	# Step 2: Load each chunk into MegaParse
	megaparse = MegaParse()
	all_responses = []  # Store all extracted text

	with open("./Data.md", "w") as file:
		for chunk_file in sorted(os.listdir(output_dir)):
			chunk_path = os.path.join(output_dir, chunk_file)
			print(f"Processing: {chunk_path}")  # Debugging print

			try:
				response = megaparse.load(chunk_path)
				if response:
					file.write(response + "\n\n")  # Write immediately
					print(f"Written response for {chunk_file}")
				else:
					print(f"Warning: Empty response for {chunk_file}")
			except Exception as e:
				print(f"Error processing {chunk_file}: {e}")