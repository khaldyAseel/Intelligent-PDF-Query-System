from ragas import evaluate
from ragas.metrics import faithfulness, relevance, answer_correctness
from langchain.llms import OpenAI

# Sample data (Replace with your dataset)
questions = ["What is the best way to store food?"]
contexts = ["The best way to store food is in airtight containers."]
answers = ["Store food in airtight containers."]

# Define evaluation metrics
metrics = [faithfulness, relevance, answer_correctness]

# Run evaluation
results = evaluate(questions, contexts, answers, metrics)
print("Ragas Evaluation Results:", results)
