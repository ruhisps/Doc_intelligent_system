from typing import TypedDict, List

from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

# ============================================================
# Import existing RAG components from rag.py
# ============================================================
#
# rag.py must expose:
#   - client          (OpenAI SDK client pointed at the vLLM
#                       OpenAI-compatible endpoint)
#   - VLLM_MODEL      (model name string)
#   - retrieve()      (hybrid dense + BM25 retrieval function,
#                       returns List[Document])
#   - format_context()          (turns retrieved docs into
#                                 [Sx] blocks)
#   - build_citation_legend()   (maps [Sx] -> source/page/image
#                                 details for display)
#   - print_citations_used()    (prints the citation breakdown
#                                 for whichever [Sx] tags appear
#                                 in a given answer)
#
# All of these are imported (not reimplemented here) so that
# citation formatting and display — page numbers, content type
# labels for figures/tables, visual IDs, citation metadata —
# is identical between rag.py's direct CLI and this graph. Two
# independent implementations would drift apart over time.

from rag import (
    client,
    VLLM_MODEL,
    retrieve,
    format_context,
    build_citation_legend,
    print_citations_used,
    generate_answer,
    build_image_content_blocks,
)


# ============================================================
# Configuration
# ============================================================

MAX_RETRIES = 2


# ============================================================
# State
# ============================================================

class RAGState(TypedDict, total=False):

    question: str

    rewritten_query: str

    documents: List[Document]

    retry_count: int

    route: str

    relevant: bool

    answer: str

    verified: bool

    final_answer: str

    documents_used: List[Document]


# ============================================================
# Fallback
# ============================================================

FALLBACK_MESSAGE = (
    "Not found in the research-paper corpus. "
    "The available papers do not provide sufficient "
    "evidence to answer this question."
)


# ============================================================
# vLLM call helper
# ============================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0,
) -> str:

    """
    Single-turn chat completion against the vLLM
    OpenAI-compatible endpoint.

    vLLM (with this model/template) does not support a
    separate "system" role, so the system-level instructions
    are folded into the single "user" message instead — same
    approach rag.py uses for its own vLLM calls.
    """

    combined_prompt = (
        f"{system_prompt.strip()}\n\n"
        f"{'=' * 60}\n\n"
        f"{user_prompt.strip()}"
    )

    response = client.chat.completions.create(

        model=VLLM_MODEL,

        messages=[

            {
                "role": "user",
                "content": combined_prompt,
            },

        ],

        temperature=temperature,

    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


def call_llm_multimodal(
    text_prompt: str,
    image_blocks: list,
    temperature: float = 0,
) -> str:

    """
    Same as call_llm, but attaches figure/table image content
    blocks (from build_image_content_blocks) alongside the text
    prompt in a single "user" message. Used by relevance_node
    and verification_node so they can judge claims about a
    figure/table against the actual image, not just its text
    metadata — the same images answer_node already sees via
    generate_answer.

    If image_blocks is empty, this behaves like a plain
    text-only call.
    """

    content = [
        {
            "type": "text",
            "text": text_prompt.strip(),
        }
    ]

    content.extend(image_blocks)

    response = client.chat.completions.create(

        model=VLLM_MODEL,

        messages=[

            {
                "role": "user",
                "content": content,
            },

        ],

        temperature=temperature,

    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# NODE 1: Router
# ============================================================

def router_node(
    state: RAGState
):

    question = state["question"]

    system_prompt = """
You are the query router for a
research-paper question answering system.

This system is ONLY intended for
research-paper-based questions.

Classify the user's query into exactly ONE:

RESEARCH_PAPER
GENERAL
OUT_OF_SCOPE


RESEARCH_PAPER:
Questions about research papers, their
methods, experiments, datasets, results,
limitations, findings, related work, etc.


GENERAL:
Simple conversation or chit chat questions
that do not require the research-paper corpus.
ANything which is technical even if you know should not be categorised as GENERAL

Examples:
- Hello
- How are you?
- What is 2 + 2?


OUT_OF_SCOPE:
Requests unrelated to this research-paper
question answering system.

Examples:
- Write a game.
- Book a hotel.
- What is today's weather?


Return ONLY:

RESEARCH_PAPER

or

GENERAL

or

OUT_OF_SCOPE
"""

    result = call_llm(
        system_prompt=system_prompt,
        user_prompt=question,
    ).upper()

    if "RESEARCH_PAPER" in result:

        route = "RESEARCH_PAPER"

    elif "OUT_OF_SCOPE" in result:

        route = "OUT_OF_SCOPE"

    else:

        route = "GENERAL"

    print(
        f"[ROUTER] {route}"
    )

    return {
        "route": route
    }


# ============================================================
# Router Decision
# ============================================================

def router_decision(
    state: RAGState
):

    route = state["route"]

    if route == "RESEARCH_PAPER":

        return "retrieve"

    if route == "GENERAL":

        return "general"

    return "fallback"


# ============================================================
# NODE 2: Retrieval
# ============================================================

def retrieval_node(
    state: RAGState
):

    question = state["question"]

    retry_count = state.get(
        "retry_count",
        0
    )

    query = state.get(
        "rewritten_query",
        question
    )

    print(
        f"\n[RETRIEVAL] {query}"
    )

    documents = retrieve(
        query
    )

    print(
        f"[RETRIEVAL] "
        f"{len(documents)} chunks retrieved"
    )

    return {
        "documents": documents,
        "retry_count": retry_count,
    }


# ============================================================
# NODE 3: Relevance Check
# ============================================================

def relevance_node(
    state: RAGState
):

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    if not documents:

        return {
            "relevant": False
        }

    context = format_context(
        documents
    )

    image_blocks, _ = build_image_content_blocks(
        documents
    )

    system_prompt = """
You are a strict relevance grader for
a research-paper RAG system.

Determine whether the retrieved chunks
(and any attached figure/table images)
contain enough information to answer the
question.

Return exactly one:

RELEVANT

or

NOT_RELEVANT

Choose RELEVANT only when the retrieved
text or images provide evidence that can
directly answer the question. If a figure
or table image is attached, inspect it
before deciding — the answer may depend
entirely on what the image shows rather
than on the surrounding text.

Do not use outside knowledge.
"""

    user_prompt = f"""
Question:

{question}


Retrieved research-paper chunks:

{context}
"""

    result = call_llm_multimodal(
        text_prompt=f"{system_prompt.strip()}\n\n{'=' * 60}\n\n{user_prompt.strip()}",
        image_blocks=image_blocks,
    ).upper()

    relevant = (
        result == "RELEVANT"
    )

    print(
        f"[RELEVANCE] {result}"
    )

    return {
        "relevant": relevant
    }


# ============================================================
# Relevance Decision
# ============================================================

def relevance_decision(
    state: RAGState
):

    relevant = state.get(
        "relevant",
        False
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    if relevant:

        return "answer"

    if retry_count < MAX_RETRIES:

        return "rewrite"

    return "fallback"


# ============================================================
# NODE 4: Query Rewrite
# ============================================================

def rewrite_node(
    state: RAGState
):

    question = state["question"]

    previous_query = state.get(
        "rewritten_query",
        question
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    system_prompt = """
You rewrite search queries for a
research-paper retrieval system.

Create a better search query that is likely
to retrieve passages directly answering
the user's question.

Rules:

- Preserve the original intent.
- Include important technical terms.
- Include relevant methods, models,
  datasets, metrics, or concepts.
- Do not answer the question.
- Do not invent information.
- Return ONLY the rewritten query.
"""

    user_prompt = f"""
Original question:

{question}


Previous query:

{previous_query}
"""

    rewritten_query = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print(
        f"[REWRITE] {rewritten_query}"
    )

    return {
        "rewritten_query": rewritten_query,

        "retry_count": retry_count + 1
    }


# ============================================================
# NODE 5: Answer Generation
# ============================================================

def answer_node(
    state: RAGState
):

    """
    Delegates to rag.py's generate_answer(), which builds the
    full multimodal message (context + citation rules + any
    figure/table images) and calls vLLM. This keeps prompting
    and generation identical between rag.py's own CLI and this
    graph — including image support, which this node previously
    lacked when it built its own text-only prompt.
    """

    question = state["question"]

    documents = state["documents"]

    answer = generate_answer(
        question,
        documents,
    )

    print(
        f"[ANSWER]\n{answer}"
    )

    return {
        "answer": answer
    }


# ============================================================
# NODE 6: Self Verification
# ============================================================

def verification_node(
    state: RAGState
):

    question = state["question"]

    answer = state["answer"]

    documents = state["documents"]

    context = format_context(
        documents
    )

    image_blocks, _ = build_image_content_blocks(
        documents
    )

    system_prompt = """
You are a strict hallucination checker
for a research-paper RAG system.

Determine whether the generated answer
is completely supported by the retrieved
research-paper context and any attached
figure/table images.

Check:

1. Every factual claim.

2. Whether the context or an attached image
   supports each claim. If a claim describes
   the contents of a figure or table, check
   it against the actual image, not just the
   surrounding caption text.

3. Whether citations refer to real sources.

4. Whether the answer contains information
   not present in the context or images.

5. Whether the answer fabricates research
   findings, methods, datasets, numbers,
   or conclusions.

6. Whether the answer contradicts the
   retrieved context or images.

Return exactly:

SUPPORTED

or

UNSUPPORTED

If even one important factual claim is
unsupported, return UNSUPPORTED. But do
not mark a claim UNSUPPORTED just because
it describes something only visible in an
attached image — inspect the image first.

Do not use outside knowledge.
"""

    user_prompt = f"""
Question:

{question}


Retrieved context:

{context}


Generated answer:

{answer}
"""

    result = call_llm_multimodal(
        text_prompt=f"{system_prompt.strip()}\n\n{'=' * 60}\n\n{user_prompt.strip()}",
        image_blocks=image_blocks,
    ).upper()

    verified = (
        result == "SUPPORTED"
    )

    print(
        f"[VERIFICATION] {result}"
    )

    return {
        "verified": verified
    }


# ============================================================
# Verification Decision
# ============================================================

def verification_decision(
    state: RAGState
):

    if state.get(
        "verified",
        False
    ):

        return "finish"

    return "fallback"


# ============================================================
# NODE 7: General Answer
# ============================================================

def general_node(
    state: RAGState
):

    question = state["question"]

    system_prompt = """
You are the general conversation component
of a research-paper QA assistant.

Answer the user's simple general question
normally.

Do NOT claim that the answer came from
the research-paper corpus.
"""

    answer = call_llm(
        system_prompt=system_prompt,
        user_prompt=question,
    )

    return {
        "final_answer": answer
    }


# ============================================================
# NODE 8: Fallback
# ============================================================

def fallback_node(
    state: RAGState
):

    return {
        "final_answer":
            FALLBACK_MESSAGE
    }


# ============================================================
# NODE 9: Finish
# ============================================================

def finish_node(
    state: RAGState
):

    return {
        "final_answer":
            state["answer"],

        # Carried through so the caller (CLI or otherwise) can
        # build a citation legend for whichever [Sx] tags show
        # up in the final answer. Only set on this successful,
        # verified path — general/fallback answers have no
        # grounded documents to cite.
        "documents_used":
            state["documents"],
    }


# ============================================================
# Build Graph
# ============================================================

builder = StateGraph(
    RAGState
)


# ============================================================
# Nodes
# ============================================================

builder.add_node(
    "router",
    router_node
)

builder.add_node(
    "retrieve",
    retrieval_node
)

builder.add_node(
    "relevance",
    relevance_node
)

builder.add_node(
    "rewrite",
    rewrite_node
)

builder.add_node(
    "answer",
    answer_node
)

builder.add_node(
    "verify",
    verification_node
)

builder.add_node(
    "general",
    general_node
)

builder.add_node(
    "fallback",
    fallback_node
)

builder.add_node(
    "finish",
    finish_node
)


# ============================================================
# Edges
# ============================================================

builder.add_edge(
    START,
    "router"
)


builder.add_conditional_edges(

    "router",

    router_decision,

    {
        "retrieve": "retrieve",
        "general": "general",
        "fallback": "fallback",
    }
)


builder.add_edge(
    "retrieve",
    "relevance"
)


builder.add_conditional_edges(

    "relevance",

    relevance_decision,

    {
        "answer": "answer",
        "rewrite": "rewrite",
        "fallback": "fallback",
    }
)


builder.add_edge(
    "rewrite",
    "retrieve"
)


builder.add_edge(
    "answer",
    "verify"
)


builder.add_conditional_edges(

    "verify",

    verification_decision,

    {
        "finish": "finish",
        "fallback": "fallback",
    }
)


builder.add_edge(
    "general",
    END
)


builder.add_edge(
    "fallback",
    END
)


builder.add_edge(
    "finish",
    END
)


# ============================================================
# Compile
# ============================================================

rag_graph = builder.compile()


# ============================================================
# Public function
# ============================================================

def ask(
    question: str
):

    """
    Returns a tuple: (final_answer, documents_used)

    documents_used is [] whenever the answer came from the
    general or fallback nodes (no grounded citations to show).
    It is only populated on the successful, verified
    research-paper answer path.
    """

    initial_state = {

        "question": question,

        "retry_count": 0,

        "documents": [],

        "relevant": False,

        "verified": False,

    }

    result = rag_graph.invoke(
        initial_state
    )

    return (
        result["final_answer"],
        result.get("documents_used", []),
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "Research Paper RAG"
    )

    print(
        "LangGraph + LangChain + Chroma + MiniLM + vLLM"
    )

    print("=" * 65)

    print(
        "\nThis system is designed ONLY for "
        "research-paper-based queries."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":

            break

        if not question:

            continue

        answer, docs = ask(
            question
        )

        print(
            "\nFinal Answer:"
        )

        print(answer)

        if docs:

            legend = build_citation_legend(docs)

            print_citations_used(answer, legend)