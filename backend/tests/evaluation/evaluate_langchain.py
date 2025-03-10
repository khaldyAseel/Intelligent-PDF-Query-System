from langchain.evaluation import load_evaluator

# Load evaluators
evaluator = load_evaluator("qa")

# Example evaluation (Replace with real data)
question = "What is the best way to store food?"
generated_answer = "Store food in plastic bags."
reference_answer = "Store food in airtight containers."

# Run evaluation
result = evaluator.evaluate_strings(
    prediction=generated_answer,
    reference=reference_answer,
    input=question
)

print("Langchain Evaluation Result:", result)
