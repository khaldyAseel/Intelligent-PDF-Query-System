from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import List
from hyprid_retrival import hybrid_node_retrieval, bert_extract_answer, client

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (IPs/domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Request and Response Models
class QueryRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    results: List[str]


@app.post("/search", response_model=SearchResult)
async def search(request: QueryRequest):
    try:
        print(request)
        query = request.query
        print(query)
        # Perform hybrid retrieval to get top nodes
        top_nodes = hybrid_node_retrieval(query, alpha=0.6, top_k=5)

        # Extract best answer using BERT
        bert_answer = bert_extract_answer(query, top_nodes)

        # Prepare additional context for LLaMA
        all_context = "\n".join([node.text for node, _ in top_nodes])

        # Get a response from LLaMA
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user",
                 "content": f"Answer the question: {query}. Here is an extracted answer: {bert_answer}. You can also use additional context: {all_context}"},
            ],
        )

        full_answer = response.choices[0].message.content
        return SearchResult(results=[full_answer])

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Hybrid Model API is running"}