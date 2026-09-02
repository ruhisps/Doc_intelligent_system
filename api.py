# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import the pre-compiled graph and citation helpers from your existing code
from langgraph_rag import rag_graph
from rag import build_citation_legend, extract_cited_ids

app = FastAPI(
    title="Research Paper Multimodal RAG API",
    description="LangGraph-powered API for retrieving and reasoning over research papers and visuals."
)

# --- Pydantic Models ---
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
    # Prepare the initial state expected by our LangGraph RAGState
    initial_state = {
        "question": request.question,
        "retry_count": 0,
        "documents": [],
        "relevant": False,
        "verified": False,
    }

    try:
        # Run the LangGraph state machine
        result = rag_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    answer = result.get("final_answer", "")
    docs = result.get("documents_used", [])

    # Reconstruct readable citations
    legend = build_citation_legend(docs)

    # We only care about citations the model *actually used* in its final
    # response. extract_cited_ids is imported from rag.py so this parsing
    # logic stays identical to what the CLI uses — one implementation,
    # not a second regex that can drift out of sync.
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

    # Filter state variables to surface the agent's internal path cleanly
    # (Exclude bulky Document objects and the massive answer texts)
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