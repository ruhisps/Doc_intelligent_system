# streamlit_app.py
"""
Minimal Streamlit chat UI (bonus) sitting in front of the FastAPI
service. Sends questions to RAG_API_URL/ask, renders the grounded
answer, a card per citation (with the source figure/table image shown
inline when one exists), and an expandable view of the LangGraph
path_taken so the agent's routing/retry/verification decisions are
visible without reading logs.
"""

import os
import html
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = os.getenv(
    "RAG_API_URL",
    "http://rag-api:8080",
)

ASK_URL = f"{API_URL.rstrip('/')}/ask"

REQUEST_TIMEOUT = 300


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       Global
    -------------------------------------------------------- */

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* --------------------------------------------------------
       Header
    -------------------------------------------------------- */

    .app-header {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                #1e3a8a 0%,
                #2563eb 50%,
                #3b82f6 100%
            );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.18);
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
    }

    .app-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }


    /* --------------------------------------------------------
       Answer
    -------------------------------------------------------- */

    .answer-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem 1.7rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
    }

    .answer-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #111827;
    }


    /* --------------------------------------------------------
       Citation
    -------------------------------------------------------- */

    .citation-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.035);
    }

    .citation-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.8rem;
    }

    .citation-meta {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        line-height: 1.8;
        color: #374151;
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }

    .citation-label {
        font-weight: 700;
        color: #111827;
    }

    .visual-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.7rem;
    }


    /* --------------------------------------------------------
       Status
    -------------------------------------------------------- */

    .status-ok {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-error {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #fee2e2;
        color: #991b1b;
        font-size: 0.8rem;
        font-weight: 600;
    }


    /* --------------------------------------------------------
       Sidebar
    -------------------------------------------------------- */

    .sidebar-section {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }


    /* --------------------------------------------------------
       Debug
    -------------------------------------------------------- */

    .debug-note {
        color: #6b7280;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def resolve_image_path(image_path):
    """
    Convert the path returned by the API into a path that exists
    inside the Streamlit Docker container.

    API may return:

        extracted_visuals/BERT/p15_figure_11.png

    Inside Streamlit container this becomes:

        /app/extracted_visuals/BERT/p15_figure_11.png
    """

    if not image_path:
        return None

    image_path = str(image_path).strip()

    if not image_path:
        return None

    # Already an absolute container path
    if image_path.startswith("/app/"):
        return image_path

    # Relative path returned by ingestion
    if image_path.startswith("extracted_visuals/"):
        return "/app/" + image_path

    # Handle ./extracted_visuals/...
    if image_path.startswith("./extracted_visuals/"):
        return "/app/" + image_path[2:]

    return image_path


def safe_text(value):
    """
    Escape text before putting it into HTML.
    """
    if value is None:
        return ""

    return html.escape(str(value))


def check_api_health():
    """
    Check whether rag-api is reachable.
    """

    try:
        response = requests.get(
            f"{API_URL.rstrip('/')}/health",
            timeout=10,
        )

        return (
            response.status_code == 200,
            response.json() if response.content else {},
        )

    except Exception as e:
        return False, str(e)


def ask_rag(question):
    """
    Send a question to the RAG API.
    """

    payload = {
        "question": question
    }

    response = requests.post(
        ASK_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:

        try:
            detail = response.json()
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"RAG API returned HTTP "
            f"{response.status_code}: {detail}"
        )

    return response.json()


def display_citation(citation, index):
    """
    Render one citation card.
    """

    source = citation.get(
        "source",
        "Unknown",
    )

    page = citation.get(
        "page",
        "Unknown",
    )

    content_type = citation.get(
        "content_type",
        "TEXT",
    )

    visual_id = citation.get(
        "visual_id",
        "",
    )

    image_path = citation.get(
        "image_path",
        "",
    )

    citation_text = citation.get(
        "citation_text",
        "N/A",
    )

    source_safe = safe_text(source)
    page_safe = safe_text(page)
    content_type_safe = safe_text(content_type)
    visual_id_safe = safe_text(visual_id)
    citation_text_safe = safe_text(citation_text)

    st.markdown(
        f"""<div class="citation-card">
<div class="citation-title">[S{index}] {content_type_safe}</div>
<div class="citation-meta">
<span class="citation-label">Source:</span> {source_safe}<br>
<span class="citation-label">Page:</span> {page_safe}<br>
<span class="citation-label">Citation:</span> {citation_text_safe}
</div>
</div>""",
        unsafe_allow_html=True,
    )

    if visual_id:

        st.markdown(
            f"""<div class="visual-label">Visual: <b>{visual_id_safe}</b></div>""",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Display actual image
    # --------------------------------------------------------

    if content_type in (
        "FIGURE",
        "TABLE",
        "figure",
        "table",
    ):

        resolved_path = resolve_image_path(
            image_path
        )

        if resolved_path:

            if os.path.exists(resolved_path):

                st.image(
                    resolved_path,
                    caption=(
                        f"{source} — "
                        f"page {page}"
                    ),
                    use_container_width=True,
                )

            else:

                st.warning(
                    f"Visual image was referenced "
                    f"but could not be found inside "
                    f"the Streamlit container."
                )

                st.code(
                    resolved_path,
                    language=None,
                )


def display_path_taken(path_taken):
    """
    Display LangGraph execution information.
    """

    if not path_taken:
        st.info(
            "No graph execution details were returned."
        )
        return

    st.json(path_taken)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """<div class="app-header">
<div class="app-title">📚 Research Paper Multimodal RAG</div>
<div class="app-subtitle">Hybrid retrieval with Chroma + BM25, multimodal figures/tables, and vLLM</div>
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### ⚙️ System"
    )

    st.markdown(
        '<div class="sidebar-section">RAG API</div>',
        unsafe_allow_html=True,
    )

    st.code(
        API_URL,
        language=None,
    )

    if st.button(
        "🔄 Check API",
        use_container_width=True,
    ):

        ok, info = check_api_health()

        if ok:

            st.success(
                "RAG API is healthy."
            )

        else:

            st.error(
                f"RAG API unavailable: {info}"
            )

    st.divider()

    st.markdown(
        "### 🔎 Retrieval"
    )

    st.caption(
        "Dense retrieval + BM25 "
        "with weighted rank fusion."
    )

    st.markdown(
        "### 🖼️ Multimodal"
    )

    st.caption(
        "Retrieved figures and tables "
        "are displayed when available."
    )

    st.divider()

    st.markdown(
        "### 💡 Example questions"
    )

    example_questions = [
        "What is the main contribution of the paper?",
        "What architecture does the paper propose?",
        "What are the results shown in the figures?",
        "Compare the methods evaluated in the paper.",
        "What does Figure 11 show?",
    ]

    for example in example_questions:

        if st.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
        ):

            st.session_state["question"] = example
            st.rerun()


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    "### 🔍 Ask about your research papers"
)

question = st.text_area(
    "Question",
    value=st.session_state.get(
        "question",
        "",
    ),
    height=100,
    placeholder=(
        "Ask a question about the "
        "ingested research papers..."
    ),
    label_visibility="collapsed",
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_clicked = st.button(
    "🚀 Ask RAG",
    type="primary",
    use_container_width=True,
)


# ============================================================
# EXECUTE QUERY
# ============================================================

if ask_clicked:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # Save current question
    st.session_state["question"] = question

    # --------------------------------------------------------
    # API call
    # --------------------------------------------------------

    with st.spinner(
        "Searching papers and generating answer..."
    ):

        try:

            result = ask_rag(
                question
            )

        except requests.exceptions.Timeout:

            st.error(
                "The RAG API timed out. "
                "The vLLM generation may still be running."
            )

            st.stop()

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the RAG API. "
                "Make sure rag-api is running."
            )

            st.stop()

        except Exception as e:

            st.error(
                f"RAG request failed: {e}"
            )

            st.stop()

    # ========================================================
    # RESPONSE
    # ========================================================

    answer = result.get(
        "answer",
        "",
    )

    citations = result.get(
        "citations",
        [],
    )

    path_taken = result.get(
        "path_taken",
        {},
    )


    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        "### 💬 Answer"
    )

    st.markdown(
        f"""<div class="answer-card">
<div class="answer-title">Research Answer</div>
</div>""",
        unsafe_allow_html=True,
    )

    if answer:

        st.markdown(
            answer
        )

    else:

        st.warning(
            "The RAG API returned an empty answer."
        )


    # ========================================================
    # CITATIONS
    # ========================================================

    st.divider()

    st.markdown(
        "### 📚 Sources"
    )

    if not citations:

        st.info(
            "No citations were returned."
        )

    else:

        st.caption(
            f"{len(citations)} source(s) "
            f"used in the answer."
        )

        for index, citation in enumerate(
            citations,
            start=1,
        ):

            display_citation(
                citation,
                index,
            )


    # ========================================================
    # GRAPH DEBUG
    # ========================================================

    st.divider()

    with st.expander(
        "🧠 LangGraph execution details",
        expanded=False,
    ):

        display_path_taken(
            path_taken
        )


    # ========================================================
    # RAW API RESPONSE
    # ========================================================

    with st.expander(
        "🔧 Raw API response",
        expanded=False,
    ):

        st.json(
            result
        )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.info(
        "Enter a question above and click "
        "**Ask RAG** to search the research papers."
    )