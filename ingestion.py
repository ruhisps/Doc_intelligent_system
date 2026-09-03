"""
ingestion.py

Offline ingestion pipeline — run once (or whenever the
document set changes) to build the knowledge base the API queries.

For every PDF in DATA_DIR, this script:
    1. Extracts page text with PyMuPDF and splits it into overlapping
       chunks (RecursiveCharacterTextSplitter).
    2. Detects figures/tables per page with DocLayout-YOLO, crops them,
       and saves the crops to VISUALS_DIR.
    3. Sends each cropped figure/table to a vision-language model
       served via vLLM (Qwen2.5-VL) to generate a rich text description
    4. Embeds every text chunk and visual description with a
       sentence-transformers model and writes everything into a
       persisted Chroma collection, alongside the metadata (source,
       page, content_type, citation, image_path, ...) that api.py and
       rag.py rely on to produce grounded, page-accurate citations.

Run with:
    python ingestion.py
    docker compose --profile ingest run --rm ingest
"""

import os
import io
import uuid
import base64
import time
import traceback
import concurrent.futures
from pathlib import Path

import pymupdf
import torch

# Disable MKLDNN because of possible CPU issues
torch.backends.mkldnn.enabled = False

from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

import cv2

# ============================================================
# THREAD / CPU CONFIG
# ============================================================

cv2.setNumThreads(1)
torch.set_num_threads(1)

load_dotenv()

# ============================================================
# DEBUG / TIMING HELPERS
# ============================================================


def log(message):
    """
    Timestamped logging.

    flush=True is important when running inside Docker,
    terminals, nohup, VSCode, etc.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_exception(prefix):
    """
    Print an error and full traceback.
    """
    log(prefix)
    traceback.print_exc()


def elapsed(start):
    """
    Return elapsed time in seconds.
    """
    return time.perf_counter() - start


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "research_papers"
VISUALS_DIR = "extracted_visuals"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# DOCLAYOUT-YOLO SETTINGS
# ============================================================

DOCLAYOUT_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
DOCLAYOUT_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
DOCLAYOUT_IMGSZ = 1024
CONFIDENCE_THRESHOLD = 0.25

# Number of PDF pages sent to YOLO at once
YOLO_BATCH_SIZE = 8

# ============================================================
# VLM SETTINGS
# ============================================================

# Number of simultaneous VLM requests
VLM_MAX_WORKERS = 4

# ============================================================
# VISUAL EXTRACTION SETTINGS
# ============================================================

VISUAL_LABELS = {
    "figure",
    "table",
}

MIN_BOX_SIDE = 50

# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log("=" * 80)
log("APPLICATION STARTING")
log("=" * 80)
log(f"[DEVICE] {DEVICE}")

if DEVICE == "cuda":
    log(f"[CUDA] Device count: {torch.cuda.device_count()}")
    log(f"[CUDA] Device name: {torch.cuda.get_device_name(0)}")
    log(f"[CUDA] PyTorch CUDA version: {torch.version.cuda}")
else:
    log("[CPU] Running inference on CPU")

# ============================================================
# VLLM CONFIG
# ============================================================

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL",
    "http://127.0.0.1:8000/v1",
)

VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
)

VLLM_API_KEY = os.getenv(
    "VLLM_API_KEY",
    "EMPTY",
)

log("=" * 80)
log("[VLLM] Configuration")
log(f"[VLLM] Base URL : {VLLM_BASE_URL}")
log(f"[VLLM] Model    : {VLLM_MODEL}")
log(f"[VLLM] Workers  : {VLM_MAX_WORKERS}")
log("=" * 80)

# ============================================================
# VLLM CLIENT
# ============================================================

log("[VLLM] Initializing ChatOpenAI client...")

vllm_client_start = time.perf_counter()

try:
    llm = ChatOpenAI(
        model=VLLM_MODEL,
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
        temperature=0,
    )

    log(
        "[VLLM] Client initialized "
        f"| elapsed={elapsed(vllm_client_start):.3f}s"
    )

except Exception:
    log_exception("[VLLM ERROR] Failed to initialize client")
    raise

# ============================================================
# EMBEDDINGS
# ============================================================

log("=" * 80)
log("[EMBEDDINGS] Initializing embedding model")
log(f"[EMBEDDINGS] Model : {EMBEDDING_MODEL}")
log(f"[EMBEDDINGS] Device: {DEVICE}")
log("=" * 80)

embedding_start = time.perf_counter()

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    log(
        "[EMBEDDINGS] Model initialized "
        f"| elapsed={elapsed(embedding_start):.3f}s"
    )

except Exception:
    log_exception("[EMBEDDINGS ERROR] Failed to initialize embeddings")
    raise

# ============================================================
# CHROMA
# ============================================================

log("=" * 80)
log("[CHROMA] Initializing vector store")
log(f"[CHROMA] Directory : {CHROMA_DIR}")
log(f"[CHROMA] Collection: {COLLECTION_NAME}")
log("=" * 80)

chroma_init_start = time.perf_counter()

try:
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    log(
        "[CHROMA] Vector store initialized "
        f"| elapsed={elapsed(chroma_init_start):.3f}s"
    )

except Exception:
    log_exception("[CHROMA ERROR] Failed to initialize vector store")
    raise

# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

log("[TEXT] Text splitter initialized")

# ============================================================
# LOAD DOCLAYOUT-YOLO
# ============================================================


def load_layout_model():
    """
    Download (if needed) and load the DocLayout-YOLO weights used to
    detect figure/table regions on each rendered PDF page.
    """
    log("=" * 80)
    log("[YOLO] Loading DocLayout-YOLO")
    log(f"[YOLO] Repository: {DOCLAYOUT_REPO_ID}")
    log(f"[YOLO] Filename  : {DOCLAYOUT_FILENAME}")
    log("=" * 80)

    model_start = time.perf_counter()

    try:
        log("[YOLO] Calling hf_hub_download()...")

        download_start = time.perf_counter()

        weights_path = hf_hub_download(
            repo_id=DOCLAYOUT_REPO_ID,
            filename=DOCLAYOUT_FILENAME,
        )

        log(
            "[YOLO] Weights ready "
            f"| elapsed={elapsed(download_start):.3f}s"
        )
        log(f"[YOLO] Weights path: {weights_path}")

        log("[YOLO] Creating YOLOv10 model...")

        load_start = time.perf_counter()

        model = YOLOv10(weights_path)

        log(
            "[YOLO] YOLOv10 model created "
            f"| elapsed={elapsed(load_start):.3f}s"
        )
        log(
            "[YOLO] TOTAL model loading time "
            f"| elapsed={elapsed(model_start):.3f}s"
        )

        return model

    except Exception:
        log_exception("[YOLO ERROR] Model loading failed")
        raise


layout_model = load_layout_model()

# ============================================================
# RENDER PDF PAGE
# ============================================================


def render_page(page, scale=2.0):
    """
    Rasterize one PyMuPDF page to a PIL Image at the given scale
    (2.0 = ~144 DPI), so DocLayout-YOLO can run object detection on it.
    """
    matrix = pymupdf.Matrix(scale, scale)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )

    return image


# ============================================================
# YOLO RESULT PARSING
# ============================================================


def _parse_result(image, result):
    """
    Convert one YOLO result into a list of filtered detections: keeps
    only "figure"/"table" classes above CONFIDENCE_THRESHOLD and at
    least MIN_BOX_SIDE pixels wide/tall, clamped to the image bounds.
    """
    names = result.names
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        raw_label = names[class_id]

        if raw_label not in VISUAL_LABELS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])

        x1 = max(0, min(x1, image.width))
        y1 = max(0, min(y1, image.height))
        x2 = max(0, min(x2, image.width))
        y2 = max(0, min(y2, image.height))

        if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
            continue

        label = "Picture" if raw_label == "figure" else "Table"

        detections.append(
            {
                "label": label,
                "class_id": class_id,
                "confidence": score,
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ],
            }
        )

    return detections


# ============================================================
# YOLO INFERENCE
# ============================================================


def detect_visuals_batch(images):
    """
    Run DocLayout-YOLO over all rendered pages in batches of
    YOLO_BATCH_SIZE, returning one detections list per page (parallel
    to `images`) of the figure/table bounding boxes found on it.
    """
    log("=" * 80)
    log("[YOLO] STARTING VISUAL DETECTION")
    log(f"[YOLO] Pages       : {len(images)}")
    log(f"[YOLO] Batch size  : {YOLO_BATCH_SIZE}")
    log(f"[YOLO] Image size  : {DOCLAYOUT_IMGSZ}")
    log(f"[YOLO] Confidence  : {CONFIDENCE_THRESHOLD}")
    log(f"[YOLO] Device      : {DEVICE}")
    log("=" * 80)

    all_detections = []
    is_cpu = DEVICE == "cpu"

    total_batches = (
        len(images) + YOLO_BATCH_SIZE - 1
    ) // YOLO_BATCH_SIZE

    total_inference_time = 0.0
    total_parse_time = 0.0

    for batch_index, batch_start in enumerate(
        range(0, len(images), YOLO_BATCH_SIZE),
        start=1,
    ):
        batch = images[
            batch_start:batch_start + YOLO_BATCH_SIZE
        ]

        batch_end = batch_start + len(batch)

        log("")
        log("-" * 70)
        log(f"[YOLO] BATCH {batch_index}/{total_batches}")
        log(f"[YOLO] Pages: {batch_start + 1}-{batch_end}")
        log(f"[YOLO] Batch size: {len(batch)}")

        if DEVICE == "cuda":
            log("[YOLO] CUDA synchronize BEFORE inference...")
            torch.cuda.synchronize()

        log("[YOLO] >>> INFERENCE START")

        inference_start = time.perf_counter()

        try:
            results = layout_model.predict(
                batch,
                imgsz=DOCLAYOUT_IMGSZ,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                half=(not is_cpu),
                verbose=False,
            )

        except Exception:
            log_exception(
                f"[YOLO ERROR] Inference failed "
                f"for pages {batch_start + 1}-{batch_end}"
            )
            raise

        if DEVICE == "cuda":
            log("[YOLO] CUDA synchronize AFTER inference...")
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - inference_start
        total_inference_time += inference_time

        avg_per_page = inference_time / len(batch)
        throughput = (
            len(batch) / inference_time
            if inference_time > 0
            else 0
        )

        log("[YOLO] <<< INFERENCE END")
        log(f"[YOLO] Inference time : {inference_time:.3f}s")
        log(f"[YOLO] Avg/page       : {avg_per_page:.3f}s")
        log(f"[YOLO] Throughput     : {throughput:.2f} pages/s")

        log("[YOLO] Starting result parsing...")

        parse_start = time.perf_counter()

        for page_offset, (image, result) in enumerate(
            zip(batch, results)
        ):
            page_number = batch_start + page_offset + 1

            detections = _parse_result(image, result)
            all_detections.append(detections)

            log(
                f"[YOLO] Page {page_number}: "
                f"{len(detections)} visual(s)"
            )

        parse_time = time.perf_counter() - parse_start
        total_parse_time += parse_time

        log(f"[YOLO] Result parsing time: {parse_time:.3f}s")
        log(
            f"[YOLO] BATCH {batch_index}/{total_batches} "
            f"COMPLETE"
        )

    total_visuals = sum(
        len(detections)
        for detections in all_detections
    )

    avg_inference_per_page = (
        total_inference_time / len(images)
        if images
        else 0
    )

    total_throughput = (
        len(images) / total_inference_time
        if total_inference_time > 0
        else 0
    )

    log("")
    log("=" * 80)
    log("[YOLO] INFERENCE SUMMARY")
    log("=" * 80)
    log(f"[YOLO] Pages              : {len(images)}")
    log(f"[YOLO] Batches            : {total_batches}")
    log(f"[YOLO] Total inference    : {total_inference_time:.3f}s")
    log(f"[YOLO] Average/page       : {avg_inference_per_page:.3f}s")
    log(f"[YOLO] Throughput         : {total_throughput:.2f} pages/s")
    log(f"[YOLO] Total parsing      : {total_parse_time:.3f}s")
    log(f"[YOLO] Visual regions     : {total_visuals}")
    log("=" * 80)

    return all_detections


# ============================================================
# CROP VISUAL
# ============================================================


def crop_visual(image, bbox, padding=20):
    """
    Crop a detected figure/table region out of the full page image,
    with a small pixel padding so borders/axis labels aren't clipped.
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    return image.crop((x1, y1, x2, y2))


# ============================================================
# SAVE VISUAL
# ============================================================


def save_visual(image, source, page_number, visual_id):
    """
    Save a cropped figure/table PNG to
    VISUALS_DIR/<pdf_stem>/p<page>_<visual_id>.png and return that
    path — this is the same path stored in Chroma metadata and later
    resolved by the Streamlit app to display the image next to its
    citation.
    """
    doc_stem = Path(source).stem
    out_dir = Path(VISUALS_DIR) / doc_stem

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = out_dir / f"p{page_number}_{visual_id}.png"

    image.save(
        file_path,
        format="PNG",
    )

    return str(file_path)


# ============================================================
# IMAGE -> BASE64
# ============================================================


def image_to_base64(image):
    """
    Encode a PIL Image as a base64 PNG string for embedding in the
    VLM's OpenAI-compatible multimodal chat request.
    """
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# VLM DESCRIPTION
# ============================================================


def describe_visual(
    image,
    source,
    page_number,
    visual_number,
    element_type,
):
    """
    Send one cropped figure/table image to the vLLM-served
    vision-language model and return a detailed factual text
    description of it (Part E, Option 1: image-aware ingestion).
    The prompt explicitly forbids guessing at unreadable content.
    Returns None on any encoding or inference failure so the caller
    can skip that visual rather than fail the whole ingestion run.
    """
    job_name = (
        f"{source} "
        f"| p.{page_number} "
        f"| visual={visual_number}"
    )

    log(
        f"[VLM] START {job_name} "
        f"| type={element_type} "
        f"| size={image.width}x{image.height}"
    )

    encode_start = time.perf_counter()

    try:
        image_base64 = image_to_base64(image)

    except Exception:
        log_exception(
            f"[VLM ERROR] Image encoding failed "
            f"| {job_name}"
        )
        return None

    encode_time = time.perf_counter() - encode_start

    log(
        f"[VLM] Image encoding | {job_name} "
        f"| time={encode_time:.3f}s "
        f"| base64_chars={len(image_base64)}"
    )

    prompt = f"""
You are analyzing a visual element from a research paper.

Document: {source}
Page: {page_number}
Visual: {visual_number}
Element type: {element_type}

Create a detailed factual description of this visual for a research-paper RAG system.
Describe ONLY information visible in the image.

If it is a chart, describe: chart type, title, x-axis, y-axis, units, legend,
variables, categories, important values, trends, comparisons, highest/lowest values.

If it is a diagram, describe: components, inputs, outputs, connections, arrows,
processing stages, architecture, relationships.

If it is a table, describe: column names, row names, important values, units,
comparisons, best/worst results.

Do NOT guess or invent information. If something cannot be read, say:
"Not legible in the provided image."

Return only the description.
"""

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64,"
                        f"{image_base64}"
                    )
                },
            },
        ]
    )

    log(f"[VLM] >>> INFERENCE START | {job_name}")

    vlm_start = time.perf_counter()

    try:
        response = llm.invoke([message])

    except Exception as e:
        vlm_time = time.perf_counter() - vlm_start

        log(
            f"[VLM ERROR] | {job_name} "
            f"| inference_time={vlm_time:.3f}s "
            f"| error={repr(e)}"
        )

        traceback.print_exc()
        return None

    vlm_time = time.perf_counter() - vlm_start

    log(
        f"[VLM] <<< INFERENCE END | {job_name} "
        f"| inference_time={vlm_time:.3f}s"
    )

    response_start = time.perf_counter()
    content = response.content

    if isinstance(content, str):
        content = content.strip()
    else:
        content = str(content).strip()

    response_processing_time = time.perf_counter() - response_start

    log(
        f"[VLM] Response processing | {job_name} "
        f"| time={response_processing_time:.3f}s "
        f"| chars={len(content)}"
    )

    total_vlm_time = time.perf_counter() - vlm_start

    log(
        f"[VLM] COMPLETE | {job_name} "
        f"| total_request_time={total_vlm_time:.3f}s"
    )

    if vlm_time > 60:
        log(
            f"[VLM WARNING] Slow VLM request "
            f"| {job_name} "
            f"| time={vlm_time:.3f}s"
        )

    return content


# ============================================================
# TEXT EXTRACTION
# ============================================================


def extract_text_documents(pdf, source):
    """
    Extract and chunk every page's plain text into LangChain Document
    objects (content_type="text"), each tagged with source, page
    number, a stable chunk_id, and a human-readable [source, p. N]
    citation string.
    """
    log("=" * 80)
    log(f"[TEXT] START | {source}")
    log(f"[TEXT] Pages: {len(pdf)}")

    documents = []
    extraction_start = time.perf_counter()

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        page_number = page_index + 1

        log(
            f"[TEXT] Page {page_number}/{len(pdf)} "
            f"| extracting..."
        )

        page_start = time.perf_counter()

        try:
            text = page.get_text("text").strip()

        except Exception:
            log_exception(
                f"[TEXT ERROR] Failed page {page_number}"
            )
            continue

        page_time = time.perf_counter() - page_start

        if not text:
            log(
                f"[TEXT] Page {page_number} "
                f"| empty "
                f"| time={page_time:.3f}s"
            )
            continue

        chunks = text_splitter.split_text(text)

        log(
            f"[TEXT] Page {page_number} "
            f"| chars={len(text)} "
            f"| chunks={len(chunks)} "
            f"| time={page_time:.3f}s"
        )

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = (
                f"{source}_p{page_number}_chunk{chunk_index}"
            )

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "page": page_number,
                        "content_type": "text",
                        "chunk_id": chunk_id,
                        "citation": (
                            f"[{source}, p. {page_number}]"
                        ),
                    },
                )
            )

    total_time = time.perf_counter() - extraction_start

    log(
        f"[TEXT] COMPLETE "
        f"| chunks={len(documents)} "
        f"| total_time={total_time:.3f}s"
    )

    return documents


# ============================================================
# VISUAL EXTRACTION
# ============================================================


def extract_visual_documents(pdf, source):
    """
    Full per-PDF visual pipeline: render every page -> detect
    figures/tables with DocLayout-YOLO -> crop and save each one ->
    describe each one via the VLM (in parallel, VLM_MAX_WORKERS at a
    time) -> return one Document per successfully described visual
    (content_type="figure"/"table"), citation-tagged the same way as
    text chunks.
    """
    log("=" * 80)
    log(f"[VISUAL] START | {source}")
    log(f"[VISUAL] Pages: {len(pdf)}")

    visual_total_start = time.perf_counter()

    # ========================================================
    # STEP 1 — RENDER PAGES
    # ========================================================

    log("=" * 60)
    log("[RENDER] STARTING PAGE RENDERING")
    log("=" * 60)

    page_images = []
    render_start = time.perf_counter()

    for i in range(len(pdf)):
        page_number = i + 1

        log(
            f"[RENDER] Page {page_number}/{len(pdf)} "
            f"| START"
        )

        page_start = time.perf_counter()

        try:
            image = render_page(
                pdf[i],
                scale=2.0,
            )
            page_images.append(image)

        except Exception:
            log_exception(
                f"[RENDER ERROR] Page {page_number}"
            )
            raise

        page_time = time.perf_counter() - page_start

        log(
            f"[RENDER] Page {page_number}/{len(pdf)} "
            f"| END "
            f"| time={page_time:.3f}s "
            f"| size={image.width}x{image.height}"
        )

    render_time = time.perf_counter() - render_start

    if page_images:
        log(
            f"[RENDER] COMPLETE "
            f"| pages={len(page_images)} "
            f"| total_time={render_time:.3f}s "
            f"| avg/page={render_time / len(page_images):.3f}s"
        )
    else:
        log("[RENDER] COMPLETE | pages=0")

    # ========================================================
    # STEP 2 — YOLO
    # ========================================================

    log("")
    log("[VISUAL] Calling YOLO detection...")

    yolo_start = time.perf_counter()

    detections_per_page = detect_visuals_batch(page_images)

    yolo_total_time = time.perf_counter() - yolo_start

    log(
        f"[VISUAL] YOLO COMPLETE "
        f"| total_time={yolo_total_time:.3f}s"
    )

    # ========================================================
    # STEP 3 — CROP + SAVE
    # ========================================================

    log("=" * 60)
    log("[VISUAL] STARTING CROP + SAVE")
    log("=" * 60)

    jobs = []
    visual_number = 0
    crop_save_start = time.perf_counter()

    for page_index, detections in enumerate(
        detections_per_page
    ):
        page_number = page_index + 1
        page_image = page_images[page_index]

        log(
            f"[VISUAL] Page {page_number} "
            f"| detections={len(detections)}"
        )

        for detection in detections:
            visual_number += 1

            label = detection["label"]
            bbox = detection["bbox"]
            confidence = detection["confidence"]

            log(
                f"[VISUAL] Visual {visual_number} "
                f"| page={page_number} "
                f"| type={label} "
                f"| confidence={confidence:.3f} "
                f"| bbox={bbox}"
            )

            crop_start = time.perf_counter()

            cropped = crop_visual(
                page_image,
                bbox,
                padding=20,
            )

            crop_time = time.perf_counter() - crop_start

            log(
                f"[VISUAL] Crop complete "
                f"| visual={visual_number} "
                f"| size={cropped.width}x{cropped.height} "
                f"| time={crop_time:.3f}s"
            )

            if cropped.width < 100 or cropped.height < 100:
                log(
                    f"[SKIP] Visual {visual_number} "
                    f"| too small"
                )
                continue

            content_type = (
                "figure"
                if label == "Picture"
                else "table"
            )

            visual_id = (
                f"{content_type}_{visual_number}"
            )

            log(f"[VISUAL] Saving {visual_id}...")

            save_start = time.perf_counter()

            try:
                image_path = save_visual(
                    cropped,
                    source,
                    page_number,
                    visual_id,
                )

            except Exception:
                log_exception(
                    f"[VISUAL ERROR] "
                    f"Failed saving {visual_id}"
                )
                continue

            save_time = time.perf_counter() - save_start

            log(
                f"[VISUAL] Saved {visual_id} "
                f"| time={save_time:.3f}s "
                f"| path={image_path}"
            )

            jobs.append(
                {
                    "cropped": cropped,
                    "page_number": page_number,
                    "visual_number": visual_number,
                    "label": label,
                    "content_type": content_type,
                    "visual_id": visual_id,
                    "image_path": image_path,
                }
            )

    crop_save_time = time.perf_counter() - crop_save_start

    log(
        f"[VISUAL] CROP + SAVE COMPLETE "
        f"| jobs={len(jobs)} "
        f"| time={crop_save_time:.3f}s"
    )

    # ========================================================
    # STEP 4 — VLM
    # ========================================================

    if not jobs:
        log("[VLM] No visual jobs. Skipping VLM.")
        return []

    log("")
    log("=" * 80)
    log("[VLM] STARTING VLM PROCESSING")
    log(f"[VLM] Jobs    : {len(jobs)}")
    log(f"[VLM] Workers : {VLM_MAX_WORKERS}")
    log("=" * 80)

    documents = []
    vlm_total_start = time.perf_counter()

    def _describe(job):
        worker_start = time.perf_counter()

        log(
            f"[VLM WORKER] START "
            f"| {job['visual_id']} "
            f"| page={job['page_number']}"
        )

        try:
            description = describe_visual(
                job["cropped"],
                source,
                job["page_number"],
                job["visual_number"],
                job["label"],
            )

            worker_time = time.perf_counter() - worker_start

            log(
                f"[VLM WORKER] END "
                f"| {job['visual_id']} "
                f"| time={worker_time:.3f}s "
                f"| success={description is not None}"
            )

            return job, description

        except Exception:
            log_exception(
                f"[VLM WORKER ERROR] {job['visual_id']}"
            )
            return job, None

    log("[VLM] Creating ThreadPoolExecutor...")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=VLM_MAX_WORKERS
    ) as executor:
        log("[VLM] ThreadPoolExecutor created")
        log("[VLM] Submitting jobs...")

        results_iterator = executor.map(
            _describe,
            jobs,
        )

        log("[VLM] All jobs submitted.")
        log("[VLM] Waiting for results...")

        completed = 0

        for job, description in results_iterator:
            completed += 1

            log(
                f"[VLM] RESULT "
                f"{completed}/{len(jobs)} "
                f"| {job['visual_id']} "
                f"| success={description is not None}"
            )

            if not description:
                log(
                    f"[SKIP] No description "
                    f"| visual={job['visual_id']} "
                    f"| image={job['image_path']}"
                )
                continue

            content = f"""
[RESEARCH PAPER {job['content_type'].upper()}]

Document: {source}
Page: {job['page_number']}
Visual ID: {job['visual_id']}
Element type: {job['label']}

Description:

{description}
""".strip()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "page": job["page_number"],
                        "content_type": job["content_type"],
                        "element_type": job["label"],
                        "visual_id": job["visual_id"],
                        "image_path": job["image_path"],
                        "citation": (
                            f"[{job['content_type'].capitalize()}: "
                            f"{source}, "
                            f"p. {job['page_number']}]"
                        ),
                    },
                )
            )

    vlm_total_time = time.perf_counter() - vlm_total_start

    log("")
    log("=" * 80)
    log("[VLM] VLM PROCESSING COMPLETE")
    log(
        f"[VLM] Successful descriptions: "
        f"{len(documents)}/{len(jobs)}"
    )
    log(
        f"[VLM] Wall-clock time: "
        f"{vlm_total_time:.3f}s"
    )
    log("=" * 80)

    visual_total_time = time.perf_counter() - visual_total_start

    log(
        f"[VISUAL] COMPLETE "
        f"| descriptions={len(documents)} "
        f"| total_time={visual_total_time:.3f}s"
    )

    return documents


# ============================================================
# PROCESS PDF
# ============================================================


def process_pdf(pdf_path):
    """
    Process a single PDF end-to-end: text extraction + chunking, then
    visual extraction + VLM description. Returns the combined list of
    text and visual Documents ready to be embedded and stored.
    """
    source = os.path.basename(pdf_path)

    log("")
    log("=" * 80)
    log(f"[PDF] PROCESSING {source}")
    log("=" * 80)

    pdf_total_start = time.perf_counter()

    log(f"[PDF] Opening: {pdf_path}")

    open_start = time.perf_counter()

    try:
        pdf = pymupdf.open(pdf_path)

    except Exception:
        log_exception(
            f"[PDF ERROR] Failed to open {pdf_path}"
        )
        raise

    open_time = time.perf_counter() - open_start

    log(
        f"[PDF] Opened "
        f"| pages={len(pdf)} "
        f"| time={open_time:.3f}s"
    )

    try:
        log("[PDF] Starting text extraction...")

        text_documents = extract_text_documents(
            pdf,
            source,
        )

        log(
            f"[PDF] Text extraction COMPLETE "
            f"| chunks={len(text_documents)}"
        )

        log("[PDF] Starting visual extraction...")

        visual_documents = extract_visual_documents(
            pdf,
            source,
        )

        log(
            f"[PDF] Visual extraction COMPLETE "
            f"| descriptions={len(visual_documents)}"
        )

    finally:
        log("[PDF] Closing PDF...")
        pdf.close()
        log("[PDF] PDF closed.")

    pdf_total_time = time.perf_counter() - pdf_total_start

    total_documents = (
        len(text_documents) + len(visual_documents)
    )

    log("=" * 80)
    log(f"[PDF] COMPLETE | {source}")
    log(f"[PDF] Text documents   : {len(text_documents)}")
    log(f"[PDF] Visual documents : {len(visual_documents)}")
    log(f"[PDF] Total documents  : {total_documents}")
    log(f"[PDF] Total time       : {pdf_total_time:.3f}s")
    log("=" * 80)

    return text_documents + visual_documents


# ============================================================
# CHROMA INGESTION
# ============================================================


def add_to_chroma(documents):
    """
    Embed and persist all extracted Documents (text + figure/table
    descriptions) into the Chroma collection. IDs are deterministic
    per source/page/chunk (or source/page/visual_id for visuals), so
    re-running ingestion on the same PDFs upserts rather than
    duplicates entries.
    """
    log("")
    log("=" * 80)
    log("[CHROMA] STARTING INGESTION")
    log(f"[CHROMA] Documents: {len(documents)}")
    log("=" * 80)

    if not documents:
        log("[CHROMA] No documents. Nothing to add.")
        return

    log("[CHROMA] Generating IDs...")

    id_start = time.perf_counter()
    ids = []

    for index, document in enumerate(documents):
        metadata = document.metadata

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        content_type = metadata.get("content_type", "text")

        if content_type in ("figure", "table"):
            visual_id = metadata.get(
                "visual_id",
                str(uuid.uuid4()),
            )

            doc_id = (
                f"{source}_p{page}_{visual_id}"
            )

        else:
            doc_id = metadata.get(
                "chunk_id",
                str(uuid.uuid4()),
            )

        ids.append(doc_id)

    id_time = time.perf_counter() - id_start

    log(
        f"[CHROMA] IDs generated "
        f"| count={len(ids)} "
        f"| time={id_time:.3f}s"
    )

    log("")
    log("[CHROMA] >>> add_documents() START")
    log(
        "[CHROMA] This includes embedding "
        "generation + vector DB insertion."
    )

    chroma_start = time.perf_counter()

    try:
        vectorstore.add_documents(
            documents=documents,
            ids=ids,
        )

    except Exception:
        chroma_time = time.perf_counter() - chroma_start

        log(
            f"[CHROMA ERROR] add_documents() failed "
            f"| elapsed={chroma_time:.3f}s"
        )

        traceback.print_exc()
        raise

    chroma_time = time.perf_counter() - chroma_start

    log("[CHROMA] <<< add_documents() END")
    log(f"[CHROMA] Total time: {chroma_time:.3f}s")

    if documents:
        log(
            f"[CHROMA] Time/document: "
            f"{chroma_time / len(documents):.3f}s"
        )

    log("[CHROMA] INGESTION COMPLETE")


# ============================================================
# MAIN
# ============================================================


def main():
    """
    Ingestion entry point: find every *.pdf in DATA_DIR, run
    process_pdf() on each, log a text/figure/table summary, and write
    everything to the persisted Chroma collection.
    """
    main_start = time.perf_counter()

    log("")
    log("=" * 80)
    log("MAIN START")
    log("=" * 80)

    data_path = Path(DATA_DIR)

    log(
        f"[MAIN] Data directory: "
        f"{data_path.resolve()}"
    )

    if not data_path.exists():
        raise RuntimeError(
            f"Data directory not found: {DATA_DIR}"
        )

    log("[MAIN] Searching for PDFs...")

    pdf_files = list(
        data_path.glob("*.pdf")
    )

    log(
        f"[MAIN] Found {len(pdf_files)} PDF(s)"
    )

    if not pdf_files:
        log("[MAIN] No PDF files found.")
        return

    all_documents = []

    for file_index, pdf_file in enumerate(
        pdf_files,
        start=1,
    ):
        log("")
        log("=" * 80)
        log(
            f"[MAIN] FILE "
            f"{file_index}/{len(pdf_files)}"
        )
        log(f"[MAIN] {pdf_file.name}")
        log("=" * 80)

        file_start = time.perf_counter()

        try:
            documents = process_pdf(
                str(pdf_file)
            )

        except Exception:
            log_exception(
                f"[MAIN ERROR] "
                f"Failed processing {pdf_file.name}"
            )
            raise

        all_documents.extend(documents)

        file_time = time.perf_counter() - file_start

        log(
            f"[MAIN] FILE COMPLETE "
            f"| {pdf_file.name} "
            f"| documents={len(documents)} "
            f"| time={file_time:.3f}s"
        )

        log(
            f"[MAIN] Accumulated documents: "
            f"{len(all_documents)}"
        )

    text_count = sum(
        1
        for doc in all_documents
        if doc.metadata.get("content_type") == "text"
    )

    figure_count = sum(
        1
        for doc in all_documents
        if doc.metadata.get("content_type") == "figure"
    )

    table_count = sum(
        1
        for doc in all_documents
        if doc.metadata.get("content_type") == "table"
    )

    log("")
    log("=" * 80)
    log("INGESTION SUMMARY")
    log("=" * 80)
    log(f"Text chunks : {text_count}")
    log(f"Figures     : {figure_count}")
    log(f"Tables      : {table_count}")
    log(f"Total       : {len(all_documents)}")
    log("=" * 80)

    log("[MAIN] Starting Chroma ingestion...")

    add_to_chroma(all_documents)

    total_time = time.perf_counter() - main_start

    log("")
    log("=" * 80)
    log("INGESTION COMPLETED")
    log("=" * 80)
    log(f"Total elapsed time: {total_time:.3f}s")
    log("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    log("[BOOT] __main__ entered.")

    try:
        main()

    except KeyboardInterrupt:
        log("[STOP] KeyboardInterrupt received.")

    except Exception:
        log_exception("[FATAL] Application crashed.")
        raise

    finally:
        log("[BOOT] Program exiting.")

