import os
import json
from sentence_transformers import SentenceTransformer

# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Change model if needed

# Directory where JSON files are stored
json_dir = "path/to/json_files"

# Process each JSON file
for filename in os.listdir(json_dir):
	if filename.endswith(".json"):
		file_path = os.path.join(json_dir, filename)

		# Load JSON file
		with open(file_path, "r", encoding="utf-8") as f:
			data = json.load(f)

		# Extract content and generate embedding
		content = data.get("content", "")
		if content:
			embedding = model.encode(content).tolist()  # Convert to list for saving
			data["embedding"] = embedding  # Add embedding to JSON data

			# Save updated JSON file
			with open(file_path, "w", encoding="utf-8") as f:
				json.dump(data, f, indent=4)

		print(f"Processed: {filename}")
