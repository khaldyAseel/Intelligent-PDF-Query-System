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
        messages=[{
            "role": "system",
            "content": "Determine if the query is related to book content. Respond with only 'yes' or 'no'."
        }, {
            "role": "user",
            "content": f"Is this query related to book content? {query}"
        }]
    )

    return response.choices[0].message.content.strip().lower() == "yes"


def route_query_with_book_context(client, query, node_scores, threshold=0.6, soft_margin=0.05):
    """
    Routes the query to LLaMA with or without book context, using node similarity scores
    or LLaMA classification to determine if the query is book-related.

    :param client: The LLaMA API client.
    :param query: The user query.
    :param node_scores: List of tuples (node, similarity_score).
    :param threshold: Minimum similarity score for using book context.
    :param soft_margin: A margin to include near-threshold cases.
    :return: The response from LLaMA.
    """
    # Check the highest similarity score from the nodes
    max_similarity = max((score for _, score in node_scores), default=0)

    print(f"Max similarity: {max_similarity} | Threshold: {threshold}")

    if max_similarity >= (threshold - soft_margin):
        # If the score is high enough, we use book context but validate with LLaMA
        relevant_chunks = [node.text for node, score in node_scores if score >= (threshold - soft_margin)]
        context = " ".join(relevant_chunks)

        print("✅ Using book context with summarization")

        # Now validate context using LLaMA
        validation_response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "Does the following context answer the query correctly?"},
                {"role": "user", "content": f"Query: {query} \nContext: {context}"}
            ]
        )

        # If LLaMA indicates that the context is useful
        validation_answer = validation_response.choices[0].message.content.strip().lower()

        if validation_answer == "yes":
            # If LLaMA confirms the context is valid, proceed with the detailed response
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": "Provide a detailed response."},
                    {"role": "user", "content": f"Answer the question: {query}. You can also use additional context: {context}"}
                ]
            )
            response_text = response.choices[0].message.content

            # Ask LLaMA to summarize the answer in 3-5 sentences
            summary_response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": "Summarize in 3-5 sentences."},
                    {"role": "user", "content": response_text}
                ]
            )
            return summary_response.choices[0].message.content

        else:
            # If LLaMA does not validate the context, return a fallback response
            print("❌ LLaMA did not validate the context, querying without context.")
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content

    else:
        # If similarity score is too low, ask LLaMA if it's related to the book
        print("❌ Using outside information (No book context detected), checking with LLaMA")

        # Fallback to LLaMA classification if needed
        is_related = is_query_book_related(client, query)

        if is_related:
            print("✅ LLaMA confirms book context is relevant, processing accordingly.")
            # Retrieve context (maybe rerun hybrid retrieval if needed) and proceed as before
            relevant_chunks = [node.text for node, score in node_scores]
            context = " ".join(relevant_chunks)

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": "Provide a detailed response."},
                    {"role": "user", "content": f"Answer the question: {query}. You can also use additional context: {context}"}
                ]
            )
            return response.choices[0].message.content

        else:
            print("❌ LLaMA confirms the query is not related to the book.")
            # Use LLaMA freely without book context
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content


# # Example usage
# query = "how are u today?"
#
# top_nodes = hybrid_node_retrieval(query, alpha=0.6, top_k=5)
#
# # Extract node scores (assuming hybrid retrieval returns (node, score))
# node_scores = [(node, score) for node, score in top_nodes]
#
# # Route the query
# response = route_query_with_book_context(client, query, node_scores, threshold=0.6)
#
# print(response)

