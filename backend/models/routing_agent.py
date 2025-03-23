from hyprid_retrival import hybrid_node_retrieval, client

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


def route_query_with_book_context(client, query, threshold=0.4, soft_margin=0.05):
    """
    Routes the query to LLaMA with or without book context, deciding whether to call hybrid retrieval.

    :param client: The LLaMA API client.
    :param query: The user query.
    :param threshold: Minimum similarity score for using book context.
    :param soft_margin: A margin to include near-threshold cases.
    :return: The response from LLaMA (with metadata if book-related).
    """
    print(f"Query: {query}")

    # Step 1: If the query is generic, use outside information immediately
    if is_generic_question(query):
        print("⚠️ Generic question detected! Using outside information.")
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": query}],
        )
        return response.choices[0].message.content

    # Step 3: Always run hybrid retrieval to check similarity scores
    print("🔍 Running hybrid retrieval to check similarity relevance...")
    node_scores = []
    top_nodes = hybrid_node_retrieval(query, alpha=0.6, top_k=5)
    node_scores = [(node, score) for node, score in top_nodes]

    # Step 4: Compute similarity scores to verify book relevance
    similarity_scores = [score for _, score in node_scores]
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    high_score_nodes = sum(1 for score in similarity_scores if score >= threshold)

    # Step 5: Override LLaMA's decision if similarity is strong
    if avg_similarity >= (threshold - soft_margin) and high_score_nodes >= 1:
        print("✅ Using book context with metadata.")
        relevant_nodes = [(node.text, node.metadata) for node, score in node_scores if score >= (threshold - soft_margin)]
        context = " ".join([text for text, _ in relevant_nodes])

        # Collect metadata
        metadata_dict = {}
        for _, meta in relevant_nodes:
            subchapter = meta.get("subchapter", "N/A")
            page = meta.get("page", "N/A")
            if subchapter not in metadata_dict:
                metadata_dict[subchapter] = set()
            metadata_dict[subchapter].add(page)

        # Format metadata
        metadata_info = "\n".join([f"📖 Subchapter: {sub}, Pages: {', '.join(map(str, sorted(pages)))}" for sub, pages in metadata_dict.items()])

        # Get LLaMA response with context
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": f"Answer the question: {query}. Use context: {context}"}],
        )
        response_text = response.choices[0].message.content

        # Summarize
        summary_response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": response_text}],
        )
        summarized_answer = summary_response.choices[0].message.content

        return f"{summarized_answer}\n\n📚 Metadata:\n{metadata_info}"

    # Step 5: If not book-related, use general chatbot response
    print("❌ Using outside information (No book context detected).")
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content


# Example usage
query = "what is the publication year of the book ?"
# Route the query
response = route_query_with_book_context(client, query, threshold=0.4)

print(response)

