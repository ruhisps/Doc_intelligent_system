# api.py
"""
FastAPI service for the agentic RAG system.

Endpoints:
    GET  /health  - checks if the API is running
    POST /ask     - runs the RAG workflow and returns the answer,
                    citations, and path taken through the graph.

Citation helpers are imported from rag.py so the API uses the same
citation logic as the CLI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from langgraph_rag import rag_graph
from rag import build_citation_legend, extract_cited_ids

app = FastAPI(
    title="Research Paper Multimodal RAG API",
    description="API for retrieving and reasoning over research papers and visuals."
)

# --- Pydantic models ---
class AskRequest(BaseModel):
    question: str

class Citation(BaseModel):
    source: str
    page: str
    content_type: str
    visual_id: Optional[str] = None
    image_path: Optional[str] = None
    citation_text: str

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    path_taken: Dict[str, Any]

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Multimodal RAG API"}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    # Initial state for the RAG graph
    initial_state = {
        "question": request.question,
        "retry_count": 0,
        "documents": [],
        "relevant": False,
        "verified": False,
    }

    try:
        result = rag_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    answer = result.get("final_answer", "")
    docs = result.get("documents_used", [])

    # Build citations from the documents used by the graph
    legend = build_citation_legend(docs)

    # Only include citations that appear in the final answer
    used_ids = extract_cited_ids(answer)

    citations = []
    for doc_idx in used_ids:
        if doc_idx in legend:
            info = legend[doc_idx]
            citations.append(
                Citation(
                    source=info.get("source", "Unknown"),
                    page=str(info.get("page", "Unknown")),
                    content_type=info.get("content_type", "TEXT"),
                    visual_id=info.get("visual_id") or None,
                    image_path=info.get("image_path") or None,
                    citation_text=info.get("citation", "N/A")
                )
            )

    # Keep only the useful state information for debugging
    path_taken = {
        key: value for key, value in result.items()
        if key not in ["documents", "documents_used", "answer", "final_answer"]
    }

    return AskResponse(
        answer=answer,
        citations=citations,
        path_taken=path_taken
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
