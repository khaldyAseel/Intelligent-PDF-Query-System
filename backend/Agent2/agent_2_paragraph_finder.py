from embedding_utils import create_faiss_index, query_faiss_index
from text_utils import split_into_paragraphs
from MockData import mock_subchapters
from langchain.llms import OpenAI


# Define the Agent 2 workflow
def agent_2(query, subchapters):
    # Step 1: Create FAISS index for subchapters
    faiss_index, subchapter_map = create_faiss_index(subchapters)

    # Step 2: Find the most relevant subchapter
    relevant_subchapter = query_faiss_index(query, faiss_index, subchapter_map)

    # Step 3: Split the subchapter into paragraphs
    paragraphs = split_into_paragraphs(relevant_subchapter)

    # Step 4: Create FAISS index for paragraphs
    faiss_index_paragraphs, paragraph_map = create_faiss_index(paragraphs)

    # Step 5: Find the most relevant paragraph
    relevant_paragraph = query_faiss_index(query, faiss_index_paragraphs, paragraph_map)

    # Step 6: Refine the result with LLM
    llm = OpenAI(model="text-davinci-003")
    prompt = f"Based on the following text:\n\n{relevant_paragraph}\n\nAnswer the question: {query}"
    final_answer = llm(prompt)

    return final_answer
