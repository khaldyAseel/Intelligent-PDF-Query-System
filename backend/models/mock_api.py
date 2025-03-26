from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import List
from hyprid_retrival import client
from routing_agent import route_query

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
    print("hiiii")
    try:
        query = request.query
        # Route the query
        response = route_query(client, query, threshold=0.6)
        print(f"✅ Response from route_query: {response}")  # Debugging print

        if not response:
            raise HTTPException(status_code=404, detail="No relevant response found.")

        return SearchResult(results=[response])

    except Exception as e:
        print(f"❌ Error in search function: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Hybrid Model API is running"}