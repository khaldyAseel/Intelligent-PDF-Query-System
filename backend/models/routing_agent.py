from hyprid_retrival import hybrid_node_retrieval, client
import openai

def is_query_book_related(client, query):
    """
    Uses LLaMA to classify whether the query is related to book knowledge.

    :param client: The LLaMA API client.
    :param query: The user query.
    :return: True if the query is book-related, False otherwise.
    """
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system",
             "content": "Determine if the query is related to book content. Respond with only 'yes' or 'no'."},
            {"role": "user", "content": f"Is this query related to book content? {query}"},
        ],
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer == "yes"


def is_generic_question(query):
    """
    Detects if a query is too generic or conversational.

    :param query: The user query.
    :return: True if the query is too general, False otherwise.
    """
    generic_phrases = [
        "how are you", "what's up", "tell me a joke", "thank you",
        "who are you", "help me", "where am i", "explain this",
        "give me an example", "what is this", "what do you mean"
    ]

    return any(phrase in query.lower() for phrase in generic_phrases)


def route_query_with_book_context(client, query, node_scores, threshold=0.7, soft_margin=0.05):
    """
    Routes the query to LLaMA with or without book context.

    :param client: The LLaMA API client.
    :param query: The user query.
    :param node_scores: List of tuples (node, similarity_score).
    :param threshold: Minimum similarity score for using book context.
    :param soft_margin: A margin to include near-threshold cases.
    :return: The response from LLaMA.
    """
    # Step 1: If the query is generic, use outside information immediately
    if is_generic_question(query):
        print("⚠️ Generic question detected! Using outside information.")
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content

    # Step 2: Ask LLaMA if the query is book-related
    is_llama_says_related = is_query_book_related(client, query)

    # Step 3: Compute similarity stats only if LLaMA says 'No'
    if not is_llama_says_related:
        similarity_scores = [score for _, score in node_scores]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        high_score_nodes = sum(1 for score in similarity_scores if score >= threshold)

        print(
            f"LLaMA says related: {is_llama_says_related} | Avg Sim: {avg_similarity} | Nodes above threshold: {high_score_nodes}")

        # Override LLaMA if similarity is consistently high
        if avg_similarity >= (threshold - soft_margin) and high_score_nodes >= 2:
            print("⚠️ LLaMA said 'No', but high average similarity detected. Using book context.")
            is_llama_says_related = True

    if is_llama_says_related:
        print("✅ Using book context with summarization.")
        relevant_chunks = [node.text for node, score in node_scores if score >= (threshold - soft_margin)]
        context = " ".join(relevant_chunks)

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "Provide a detailed response."},
                {"role": "user",
                 "content": f"Answer the question: {query}. You can also use additional context: {context}"},
            ],
        )
        response_text = response.choices[0].message.content

        # Summarize in 3-5 sentences
        summary_response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "Summarize in 3-5 sentences."},
                {"role": "user", "content": response_text},
            ],
        )
        return summary_response.choices[0].message.content
    else:
        print("❌ Using outside information (No book context detected).")
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content


# Example usage
query = "what is the weather today?"

top_nodes = hybrid_node_retrieval(query, alpha=0.6, top_k=5)

# Extract node scores (assuming hybrid retrieval returns (node, score))
node_scores = [(node, score) for node, score in top_nodes]

# Route the query
response = route_query_with_book_context(client, query, node_scores, threshold=0.6)

print(response)

