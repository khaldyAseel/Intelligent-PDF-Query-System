from megaparse.megaparse import MegaParse
from megaparse.parser.llama import LlamaParser
import os

lLAMA_CLOUD_API = "llx-NuJxG1FwEceSYPdTWmRGoKrBHBoxxzjnmRQaTeGdQUt4HKq8"
# Initialize parser and MegaParse
parser = LlamaParser(api_key=lLAMA_CLOUD_API)
megaparse = MegaParse(parser)

# Debug: Inspect the parser object
print(f"Parser object: {parser}")
print(f"Parser type: {type(parser)}")

# Split the PDF into smaller chunks
def split_pdf(input_pdf, output_dir, chunk_size=10):
	import fitz  # PyMuPDF
	doc = fitz.open(input_pdf)
	for i in range(0, len(doc), chunk_size):
		chunk = fitz.open()
		chunk.insert_pdf(doc, from_page=i, to_page=min(i + chunk_size - 1, len(doc) - 1))
		chunk.save(f"{output_dir}/chunk_{i // chunk_size + 1}.pdf")
		chunk.close()

if __name__ == "__main__":
	# Process each chunk
	output_dir = "./chunks"
	os.makedirs(output_dir, exist_ok=True)
	split_pdf("book.pdf", "./chunks")
	for chunk_file in os.listdir("./chunks"):
		try:
			response = megaparse.load(f"./chunks/{chunk_file}")
			print(response)
		except Exception as e:
			print(f"Error processing {chunk_file}: {e}")

# Save the final output
megaparse.save("./test.md")