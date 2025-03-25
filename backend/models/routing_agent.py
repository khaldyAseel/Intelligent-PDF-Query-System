from hyprid_retrival import retrieved_nodes_and_scores, client
from metadata_example import metadata_context, metadata_keywords

def is_generic_question(query):
    """
    Detects if a query is too generic or conversational.

    :param query: The user query.
    :return: True if the query is too general, False otherwise.
    """
    generic_phrases = [
        "hi", "hello", "hey", "how are you", "what's up", "tell me a joke", "thank you",
        "who are you", "help me", "where am i", "explain this",
        "give me an example", "what is this", "what do you mean",
        "define", "what is the meaning of", "how does it work",
        "why is that", "tell me something interesting", "can you help me",
        "what do you think", "is it true that", "what should I do"
    ]

    return any(phrase in query.lower() for phrase in generic_phrases)


def is_metadata_related(client, query, metadata_keywords):
    """
    Determines if a query is related to book metadata (e.g., title, author, publisher, ISBN, etc.).

    :param client: The LLaMA API client.
    :param query: The user query.
    :param metadata_keywords: List of metadata-related keywords.
    :return: True if the query is metadata-related, False otherwise.
    """
    metadata_kw_str = ", ".join(metadata_keywords)  # More natural formatting for LLaMA
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content":
                "Determine if the following query is related to book metadata, such as title, author, publisher, ISBN, edition, format, language, and publication year. "
                "Respond with only 'yes' or 'no' and no extra words."},
            {"role": "user", "content": f"Query: {query}\nMetadata categories: {metadata_kw_str}"},
        ],
    )

    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer  # More flexible check


def route_query(client, query, threshold=0.6, soft_margin=0.05):
    """
    Routes the query to LLaMA with or without book context, returning metadata if relevant.

    :param client: The LLaMA API client.
    :param query: The user query.
    :param node_scores: List of tuples (node, similarity_score).
    :param threshold: Minimum similarity score for using book context.
    :param soft_margin: A margin to include near-threshold cases.
    :return: The response from LLaMA (with metadata if book-related).
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

    #Step 2: check if it's a Metadata question
    metadata_related = is_metadata_related(client, query, metadata_keywords)
    if metadata_related:
        metadata_context_str = "\n".join(metadata_context)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "You are a book assistant with knowledge about book metadata."},
                {"role": "user",
                 "content": f"Answer the question: {query} based on the following metadata: {metadata_context_str}"},
            ],
        )
        return response.choices[0].message.content

    # Step 3: Always run hybrid retrieval to check similarity scores
    print("🔍 Running hybrid retrieval to check similarity relevance...")
    top_nodes, node_scores = retrieved_nodes_and_scores(query, alpha=0.6, top_k=5)
    # Step 4: Compute similarity scores to verify book relevance
    similarity_scores = list(node_scores)
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

    # Step 5: Override LLaMA's decision if similarity is strong
    if avg_similarity >= (threshold - soft_margin):
        print("✅ Using book context with metadata.")
        relevant_nodes = [(node.text, node.metadata) for node, score in zip(top_nodes, node_scores)  if score >= (threshold - soft_margin)]
        context = " ".join([text for text, _ in relevant_nodes])

        # Collect metadata for the response
        metadata_info = "\n".join(
            [f"📖 Subchapter: {meta.get('subchapter', 'N/A')}, Page: {meta.get('page', 'N/A')}" for _, meta in
             relevant_nodes])

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant. Answer the question clearly and concisely, integrating relevant book context."},
                {"role": "user", "content": f"Based on the following book content, answer the question: {query}\n\nBook Context: {context}"},
            ],
        )
        response_text = response.choices[0].message.content

        return f"{response_text}\n\n📚 **Metadata:**\n{metadata_info}"

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


# query = "who is the editors of the book?"
#
# response = route_query(client,query,threshold=0.6, soft_margin=0.05)
# print(response)