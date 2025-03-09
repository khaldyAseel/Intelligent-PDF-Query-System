from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import List
from hyprid_retrival import hybrid_node_retrieval, client
from routing_agent import route_query_with_book_context

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

        # Extract node scores (assuming hybrid retrieval returns (node, score))
        node_scores = [(node, score) for node, score in top_nodes]

        # Route the query
        response = route_query_with_book_context(client, query, node_scores, threshold=0.6)

        return SearchResult(results=[response])

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Hybrid Model API is running"}