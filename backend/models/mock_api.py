from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import List
from hyprid_retrival import client
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
        query = request.query.strip()  # Remove extra spaces
        print(f"📩 Received query: {query}")  # Debugging log

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        response = route_query_with_book_context(client, query, threshold=0.4)

        return SearchResult(results=[response])

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Hybrid Model API is running"}