from agent_2_paragraph_finder import agent_2
from MockData import mock_subchapters

# Example query
query = "How are cocoa nibs prepared?"

# Run Agent 2
answer = agent_2(query, mock_subchapters)
print(f"Final Answer:\n{answer}")
