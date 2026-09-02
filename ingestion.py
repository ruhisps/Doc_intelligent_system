import os
import io
import json
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

    print(
        f"[{timestamp}] {message}",
        flush=True,
    )


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

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# DOCLAYOUT-YOLO SETTINGS
# ============================================================

DOCLAYOUT_REPO_ID = (
    "juliozhao/DocLayout-YOLO-DocStructBench"
)

DOCLAYOUT_FILENAME = (
    "doclayout_yolo_docstructbench_imgsz1024.pt"
)

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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


log("=" * 80)
log("APPLICATION STARTING")
log("=" * 80)

log(f"[DEVICE] {DEVICE}")

if DEVICE == "cuda":

    log(
        f"[CUDA] Device count: "
        f"{torch.cuda.device_count()}"
    )

    log(
        f"[CUDA] Device name: "
        f"{torch.cuda.get_device_name(0)}"
    )

    log(
        f"[CUDA] PyTorch CUDA version: "
        f"{torch.version.cuda}"
    )

else:

    log(
        "[CPU] Running inference on CPU"
    )


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

    log_exception(
        "[VLLM ERROR] Failed to initialize client"
    )

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
        model_kwargs={
            "device": DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    log(
        "[EMBEDDINGS] Model initialized "
        f"| elapsed={elapsed(embedding_start):.3f}s"
    )

except Exception:

    log_exception(
        "[EMBEDDINGS ERROR] "
        "Failed to initialize embeddings"
    )

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

    log_exception(
        "[CHROMA ERROR] "
        "Failed to initialize vector store"
    )

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

    log("=" * 80)
    log("[YOLO] Loading DocLayout-YOLO")
    log(f"[YOLO] Repository: {DOCLAYOUT_REPO_ID}")
    log(f"[YOLO] Filename  : {DOCLAYOUT_FILENAME}")
    log("=" * 80)

    model_start = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Download / locate weights
        # ----------------------------------------------------

        log(
            "[YOLO] Calling hf_hub_download()..."
        )

        download_start = time.perf_counter()

        weights_path = hf_hub_download(
            repo_id=DOCLAYOUT_REPO_ID,
            filename=DOCLAYOUT_FILENAME,
        )

        log(
            "[YOLO] Weights ready "
            f"| elapsed={elapsed(download_start):.3f}s"
        )

        log(
            f"[YOLO] Weights path: {weights_path}"
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        log(
            "[YOLO] Creating YOLOv10 model..."
        )

        load_start = time.perf_counter()

        model = YOLOv10(
            weights_path
        )

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

        log_exception(
            "[YOLO ERROR] Model loading failed"
        )

        raise


layout_model = load_layout_model()


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_page(
    page,
    scale=2.0,
):

    matrix = pymupdf.Matrix(
        scale,
        scale,
    )

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

def _parse_result(
    image,
    result,
):

    names = result.names

    detections = []

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        raw_label = names[class_id]

        if raw_label not in VISUAL_LABELS:
            continue

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        score = float(
            box.conf[0]
        )

        x1 = max(
            0,
            min(x1, image.width)
        )

        y1 = max(
            0,
            min(y1, image.height)
        )

        x2 = max(
            0,
            min(x2, image.width)
        )

        y2 = max(
            0,
            min(y2, image.height)
        )

        if (
            (x2 - x1) < MIN_BOX_SIDE
            or
            (y2 - y1) < MIN_BOX_SIDE
        ):
            continue

        label = (
            "Picture"
            if raw_label == "figure"
            else "Table"
        )

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

def detect_visuals_batch(
    images
):

    log("=" * 80)
    log("[YOLO] STARTING VISUAL DETECTION")
    log(f"[YOLO] Pages       : {len(images)}")
    log(f"[YOLO] Batch size  : {YOLO_BATCH_SIZE}")
    log(f"[YOLO] Image size  : {DOCLAYOUT_IMGSZ}")
    log(f"[YOLO] Confidence  : {CONFIDENCE_THRESHOLD}")
    log(f"[YOLO] Device      : {DEVICE}")
    log("=" * 80)

    all_detections = []

    is_cpu = (
        DEVICE == "cpu"
    )

    total_batches = (
        len(images)
        + YOLO_BATCH_SIZE
        - 1
    ) // YOLO_BATCH_SIZE

    total_inference_time = 0.0

    total_parse_time = 0.0

    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    for batch_index, batch_start in enumerate(
        range(
            0,
            len(images),
            YOLO_BATCH_SIZE,
        ),
        start=1,
    ):

        batch = images[
            batch_start:
            batch_start + YOLO_BATCH_SIZE
        ]

        batch_end = (
            batch_start
            + len(batch)
        )

        log("")
        log("-" * 70)

        log(
            f"[YOLO] BATCH "
            f"{batch_index}/{total_batches}"
        )

        log(
            f"[YOLO] Pages: "
            f"{batch_start + 1}-"
            f"{batch_end}"
        )

        log(
            f"[YOLO] Batch size: "
            f"{len(batch)}"
        )

        # ----------------------------------------------------
        # GPU SYNC BEFORE INFERENCE
        # ----------------------------------------------------

        if DEVICE == "cuda":

            log(
                "[YOLO] CUDA synchronize "
                "BEFORE inference..."
            )

            torch.cuda.synchronize()

        # ----------------------------------------------------
        # INFERENCE START
        # ----------------------------------------------------

        log(
            "[YOLO] >>> INFERENCE START"
        )

        inference_start = (
            time.perf_counter()
        )

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
                f"[YOLO ERROR] "
                f"Inference failed "
                f"for pages "
                f"{batch_start + 1}-"
                f"{batch_end}"
            )

            raise

        # ----------------------------------------------------
        # GPU SYNC AFTER INFERENCE
        # ----------------------------------------------------

        if DEVICE == "cuda":

            log(
                "[YOLO] CUDA synchronize "
                "AFTER inference..."
            )

            torch.cuda.synchronize()

        # ----------------------------------------------------
        # INFERENCE END
        # ----------------------------------------------------

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        total_inference_time += (
            inference_time
        )

        avg_per_page = (
            inference_time
            / len(batch)
        )

        throughput = (
            len(batch)
            / inference_time
            if inference_time > 0
            else 0
        )

        log(
            "[YOLO] <<< INFERENCE END"
        )

        log(
            f"[YOLO] Inference time : "
            f"{inference_time:.3f}s"
        )

        log(
            f"[YOLO] Avg/page       : "
            f"{avg_per_page:.3f}s"
        )

        log(
            f"[YOLO] Throughput     : "
            f"{throughput:.2f} pages/s"
        )

        # ----------------------------------------------------
        # RESULT PARSING
        # ----------------------------------------------------

        log(
            "[YOLO] Starting result parsing..."
        )

        parse_start = (
            time.perf_counter()
        )

        for page_offset, (
            image,
            result,
        ) in enumerate(
            zip(
                batch,
                results,
            )
        ):

            page_number = (
                batch_start
                + page_offset
                + 1
            )

            detections = _parse_result(
                image,
                result,
            )

            all_detections.append(
                detections
            )

            log(
                f"[YOLO] Page "
                f"{page_number}: "
                f"{len(detections)} visual(s)"
            )

        parse_time = (
            time.perf_counter()
            - parse_start
        )

        total_parse_time += (
            parse_time
        )

        log(
            f"[YOLO] Result parsing time: "
            f"{parse_time:.3f}s"
        )

        log(
            f"[YOLO] BATCH "
            f"{batch_index}/{total_batches} "
            f"COMPLETE"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_visuals = sum(
        len(detections)
        for detections
        in all_detections
    )

    avg_inference_per_page = (
        total_inference_time
        / len(images)
        if images
        else 0
    )

    total_throughput = (
        len(images)
        / total_inference_time
        if total_inference_time > 0
        else 0
    )

    log("")
    log("=" * 80)
    log("[YOLO] INFERENCE SUMMARY")
    log("=" * 80)

    log(
        f"[YOLO] Pages              : "
        f"{len(images)}"
    )

    log(
        f"[YOLO] Batches            : "
        f"{total_batches}"
    )

    log(
        f"[YOLO] Total inference    : "
        f"{total_inference_time:.3f}s"
    )

    log(
        f"[YOLO] Average/page       : "
        f"{avg_inference_per_page:.3f}s"
    )

    log(
        f"[YOLO] Throughput         : "
        f"{total_throughput:.2f} pages/s"
    )

    log(
        f"[YOLO] Total parsing      : "
        f"{total_parse_time:.3f}s"
    )

    log(
        f"[YOLO] Visual regions     : "
        f"{total_visuals}"
    )

    log("=" * 80)

    return all_detections


# ============================================================
# CROP VISUAL
# ============================================================

def crop_visual(
    image,
    bbox,
    padding=20,
):

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        x1 - padding
    )

    y1 = max(
        0,
        y1 - padding
    )

    x2 = min(
        image.width,
        x2 + padding
    )

    y2 = min(
        image.height,
        y2 + padding
    )

    return image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )


# ============================================================
# SAVE VISUAL
# ============================================================

def save_visual(
    image,
    source,
    page_number,
    visual_id,
):

    doc_stem = Path(
        source
    ).stem

    out_dir = (
        Path(VISUALS_DIR)
        / doc_stem
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        out_dir
        / f"p{page_number}_{visual_id}.png"
    )

    image.save(
        file_path,
        format="PNG",
    )

    return str(
        file_path
    )


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(
    image
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )


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

    job_name = (
        f"{source} "
        f"| p.{page_number} "
        f"| visual={visual_number}"
    )

    log("")
    log(
        f"[VLM] START "
        f"{job_name} "
        f"| type={element_type} "
        f"| size={image.width}x{image.height}"
    )

    # --------------------------------------------------------
    # BASE64 ENCODING
    # --------------------------------------------------------

    encode_start = (
        time.perf_counter()
    )

    try:

        image_base64 = (
            image_to_base64(
                image
            )
        )

    except Exception:

        log_exception(
            f"[VLM ERROR] "
            f"Image encoding failed "
            f"| {job_name}"
        )

        return None

    encode_time = (
        time.perf_counter()
        - encode_start
    )

    log(
        f"[VLM] Image encoding "
        f"| {job_name} "
        f"| time={encode_time:.3f}s "
        f"| base64_chars={len(image_base64)}"
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VLM INFERENCE / REQUEST
    # --------------------------------------------------------

    log(
        f"[VLM] >>> INFERENCE START "
        f"| {job_name}"
    )

    vlm_start = (
        time.perf_counter()
    )

    try:

        response = llm.invoke(
            [message]
        )

    except Exception as e:

        vlm_time = (
            time.perf_counter()
            - vlm_start
        )

        log(
            f"[VLM ERROR] "
            f"| {job_name} "
            f"| inference_time={vlm_time:.3f}s "
            f"| error={repr(e)}"
        )

        traceback.print_exc()

        return None

    # --------------------------------------------------------
    # VLM INFERENCE END
    # --------------------------------------------------------

    vlm_time = (
        time.perf_counter()
        - vlm_start
    )

    log(
        f"[VLM] <<< INFERENCE END "
        f"| {job_name} "
        f"| inference_time={vlm_time:.3f}s"
    )

    # --------------------------------------------------------
    # RESPONSE PROCESSING
    # --------------------------------------------------------

    response_start = (
        time.perf_counter()
    )

    content = response.content

    if isinstance(
        content,
        str,
    ):

        content = content.strip()

    else:

        content = str(
            content
        ).strip()

    response_processing_time = (
        time.perf_counter()
        - response_start
    )

    log(
        f"[VLM] Response processing "
        f"| {job_name} "
        f"| time={response_processing_time:.3f}s "
        f"| chars={len(content)}"
    )

    # --------------------------------------------------------
    # TOTAL VLM TIME
    # --------------------------------------------------------

    total_vlm_time = (
        time.perf_counter()
        - vlm_start
    )

    log(
        f"[VLM] COMPLETE "
        f"| {job_name} "
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

def extract_text_documents(
    pdf,
    source,
):

    log("=" * 80)
    log(
        f"[TEXT] START "
        f"| {source}"
    )

    log(
        f"[TEXT] Pages: {len(pdf)}"
    )

    documents = []

    extraction_start = (
        time.perf_counter()
    )

    for page_index in range(
        len(pdf)
    ):

        page = pdf[
            page_index
        ]

        page_number = (
            page_index + 1
        )

        log(
            f"[TEXT] Page "
            f"{page_number}/{len(pdf)} "
            f"| extracting..."
        )

        page_start = (
            time.perf_counter()
        )

        try:

            text = (
                page
                .get_text("text")
                .strip()
            )

        except Exception:

            log_exception(
                f"[TEXT ERROR] "
                f"Failed page "
                f"{page_number}"
            )

            continue

        page_time = (
            time.perf_counter()
            - page_start
        )

        if not text:

            log(
                f"[TEXT] Page "
                f"{page_number} "
                f"| empty "
                f"| time={page_time:.3f}s"
            )

            continue

        chunks = (
            text_splitter
            .split_text(text)
        )

        log(
            f"[TEXT] Page "
            f"{page_number} "
            f"| chars={len(text)} "
            f"| chunks={len(chunks)} "
            f"| time={page_time:.3f}s"
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                f"{source}_p"
                f"{page_number}_chunk"
                f"{chunk_index}"
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
                            f"[{source}, "
                            f"p. {page_number}]"
                        ),
                    },
                )
            )

    total_time = (
        time.perf_counter()
        - extraction_start
    )

    log(
        f"[TEXT] COMPLETE "
        f"| chunks={len(documents)} "
        f"| total_time={total_time:.3f}s"
    )

    return documents


# ============================================================
# VISUAL EXTRACTION
# ============================================================

def extract_visual_documents(
    pdf,
    source,
):

    log("=" * 80)
    log(
        f"[VISUAL] START "
        f"| {source}"
    )

    log(
        f"[VISUAL] Pages: {len(pdf)}"
    )

    visual_total_start = (
        time.perf_counter()
    )

    # ========================================================
    # STEP 1 — RENDER PAGES
    # ========================================================

    log("=" * 60)
    log("[RENDER] STARTING PAGE RENDERING")
    log("=" * 60)

    page_images = []

    render_start = (
        time.perf_counter()
    )

    for i in range(
        len(pdf)
    ):

        page_number = (
            i + 1
        )

        log(
            f"[RENDER] Page "
            f"{page_number}/{len(pdf)} "
            f"| START"
        )

        page_start = (
            time.perf_counter()
        )

        try:

            image = render_page(
                pdf[i],
                scale=2.0,
            )

            page_images.append(
                image
            )

        except Exception:

            log_exception(
                f"[RENDER ERROR] "
                f"Page {page_number}"
            )

            raise

        page_time = (
            time.perf_counter()
            - page_start
        )

        log(
            f"[RENDER] Page "
            f"{page_number}/{len(pdf)} "
            f"| END "
            f"| time={page_time:.3f}s "
            f"| size={image.width}x{image.height}"
        )

    render_time = (
        time.perf_counter()
        - render_start
    )

    log(
        f"[RENDER] COMPLETE "
        f"| pages={len(page_images)} "
        f"| total_time={render_time:.3f}s "
        f"| avg/page={render_time / len(page_images):.3f}s"
        if page_images
        else
        "[RENDER] COMPLETE | pages=0"
    )

    # ========================================================
    # STEP 2 — YOLO
    # ========================================================

    log("")
    log(
        "[VISUAL] Calling YOLO detection..."
    )

    yolo_start = (
        time.perf_counter()
    )

    detections_per_page = (
        detect_visuals_batch(
            page_images
        )
    )

    yolo_total_time = (
        time.perf_counter()
        - yolo_start
    )

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

    crop_save_start = (
        time.perf_counter()
    )

    for page_index, detections in enumerate(
        detections_per_page
    ):

        page_number = (
            page_index + 1
        )

        page_image = (
            page_images[page_index]
        )

        log(
            f"[VISUAL] Page "
            f"{page_number} "
            f"| detections={len(detections)}"
        )

        for detection in detections:

            visual_number += 1

            label = (
                detection["label"]
            )

            bbox = (
                detection["bbox"]
            )

            confidence = (
                detection["confidence"]
            )

            log(
                f"[VISUAL] Visual "
                f"{visual_number} "
                f"| page={page_number} "
                f"| type={label} "
                f"| confidence={confidence:.3f} "
                f"| bbox={bbox}"
            )

            crop_start = (
                time.perf_counter()
            )

            cropped = crop_visual(
                page_image,
                bbox,
                padding=20,
            )

            crop_time = (
                time.perf_counter()
                - crop_start
            )

            log(
                f"[VISUAL] Crop complete "
                f"| visual={visual_number} "
                f"| size={cropped.width}x{cropped.height} "
                f"| time={crop_time:.3f}s"
            )

            if (
                cropped.width < 100
                or cropped.height < 100
            ):

                log(
                    f"[SKIP] Visual "
                    f"{visual_number} "
                    f"| too small"
                )

                continue

            content_type = (
                "figure"
                if label == "Picture"
                else "table"
            )

            visual_id = (
                f"{content_type}_"
                f"{visual_number}"
            )

            log(
                f"[VISUAL] Saving "
                f"{visual_id}..."
            )

            save_start = (
                time.perf_counter()
            )

            try:

                image_path = (
                    save_visual(
                        cropped,
                        source,
                        page_number,
                        visual_id,
                    )
                )

            except Exception:

                log_exception(
                    f"[VISUAL ERROR] "
                    f"Failed saving "
                    f"{visual_id}"
                )

                continue

            save_time = (
                time.perf_counter()
                - save_start
            )

            log(
                f"[VISUAL] Saved "
                f"{visual_id} "
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

    crop_save_time = (
        time.perf_counter()
        - crop_save_start
    )

    log(
        f"[VISUAL] CROP + SAVE COMPLETE "
        f"| jobs={len(jobs)} "
        f"| time={crop_save_time:.3f}s"
    )

    # ========================================================
    # STEP 4 — VLM
    # ========================================================

    if not jobs:

        log(
            "[VLM] No visual jobs. "
            "Skipping VLM."
        )

        return []

    log("")
    log("=" * 80)
    log("[VLM] STARTING VLM PROCESSING")
    log(f"[VLM] Jobs    : {len(jobs)}")
    log(f"[VLM] Workers : {VLM_MAX_WORKERS}")
    log("=" * 80)

    documents = []

    vlm_total_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # WORKER
    # --------------------------------------------------------

    def _describe(job):

        worker_start = (
            time.perf_counter()
        )

        log(
            f"[VLM WORKER] START "
            f"| {job['visual_id']} "
            f"| page={job['page_number']}"
        )

        try:

            description = (
                describe_visual(
                    job["cropped"],
                    source,
                    job["page_number"],
                    job["visual_number"],
                    job["label"],
                )
            )

            worker_time = (
                time.perf_counter()
                - worker_start
            )

            log(
                f"[VLM WORKER] END "
                f"| {job['visual_id']} "
                f"| time={worker_time:.3f}s "
                f"| success={description is not None}"
            )

            return (
                job,
                description,
            )

        except Exception:

            log_exception(
                f"[VLM WORKER ERROR] "
                f"{job['visual_id']}"
            )

            return (
                job,
                None,
            )

    # --------------------------------------------------------
    # EXECUTOR
    # --------------------------------------------------------

    log(
        "[VLM] Creating ThreadPoolExecutor..."
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=VLM_MAX_WORKERS
    ) as executor:

        log(
            "[VLM] ThreadPoolExecutor created"
        )

        log(
            "[VLM] Submitting jobs..."
        )

        results_iterator = (
            executor.map(
                _describe,
                jobs,
            )
        )

        log(
            "[VLM] All jobs submitted."
        )

        log(
            "[VLM] Waiting for results..."
        )

        completed = 0

        for job, description in (
            results_iterator
        ):

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

    vlm_total_time = (
        time.perf_counter()
        - vlm_total_start
    )

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

    # ========================================================
    # VISUAL TOTAL
    # ========================================================

    visual_total_time = (
        time.perf_counter()
        - visual_total_start
    )

    log(
        f"[VISUAL] COMPLETE "
        f"| descriptions={len(documents)} "
        f"| total_time={visual_total_time:.3f}s"
    )

    return documents


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path
):

    source = os.path.basename(
        pdf_path
    )

    log("")
    log("=" * 80)
    log(
        f"[PDF] PROCESSING "
        f"{source}"
    )
    log("=" * 80)

    pdf_total_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # OPEN PDF
    # --------------------------------------------------------

    log(
        f"[PDF] Opening: "
        f"{pdf_path}"
    )

    open_start = (
        time.perf_counter()
    )

    try:

        pdf = pymupdf.open(
            pdf_path
        )

    except Exception:

        log_exception(
            f"[PDF ERROR] "
            f"Failed to open "
            f"{pdf_path}"
        )

        raise

    open_time = (
        time.perf_counter()
        - open_start
    )

    log(
        f"[PDF] Opened "
        f"| pages={len(pdf)} "
        f"| time={open_time:.3f}s"
    )

    try:

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        log(
            "[PDF] Starting text extraction..."
        )

        text_documents = (
            extract_text_documents(
                pdf,
                source,
            )
        )

        log(
            f"[PDF] Text extraction COMPLETE "
            f"| chunks={len(text_documents)}"
        )

        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        log(
            "[PDF] Starting visual extraction..."
        )

        visual_documents = (
            extract_visual_documents(
                pdf,
                source,
            )
        )

        log(
            f"[PDF] Visual extraction COMPLETE "
            f"| descriptions={len(visual_documents)}"
        )

    finally:

        log(
            "[PDF] Closing PDF..."
        )

        pdf.close()

        log(
            "[PDF] PDF closed."
        )

    pdf_total_time = (
        time.perf_counter()
        - pdf_total_start
    )

    total_documents = (
        len(text_documents)
        + len(visual_documents)
    )

    log("=" * 80)
    log(
        f"[PDF] COMPLETE "
        f"| {source}"
    )

    log(
        f"[PDF] Text documents   : "
        f"{len(text_documents)}"
    )

    log(
        f"[PDF] Visual documents : "
        f"{len(visual_documents)}"
    )

    log(
        f"[PDF] Total documents  : "
        f"{total_documents}"
    )

    log(
        f"[PDF] Total time       : "
        f"{pdf_total_time:.3f}s"
    )

    log("=" * 80)

    return (
        text_documents
        + visual_documents
    )


# ============================================================
# CHROMA INGESTION
# ============================================================

def add_to_chroma(
    documents
):

    log("")
    log("=" * 80)
    log("[CHROMA] STARTING INGESTION")
    log(
        f"[CHROMA] Documents: "
        f"{len(documents)}"
    )
    log("=" * 80)

    if not documents:

        log(
            "[CHROMA] No documents. "
            "Nothing to add."
        )

        return

    # --------------------------------------------------------
    # CREATE IDS
    # --------------------------------------------------------

    log(
        "[CHROMA] Generating IDs..."
    )

    id_start = (
        time.perf_counter()
    )

    ids = []

    for index, document in enumerate(
        documents
    ):

        metadata = (
            document.metadata
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        content_type = metadata.get(
            "content_type",
            "text",
        )

        if content_type in (
            "figure",
            "table",
        ):

            visual_id = metadata.get(
                "visual_id",
                str(uuid.uuid4()),
            )

            doc_id = (
                f"{source}_p{page}_"
                f"{visual_id}"
            )

        else:

            doc_id = metadata.get(
                "chunk_id",
                str(uuid.uuid4()),
            )

        ids.append(
            doc_id
        )

    id_time = (
        time.perf_counter()
        - id_start
    )

    log(
        f"[CHROMA] IDs generated "
        f"| count={len(ids)} "
        f"| time={id_time:.3f}s"
    )

    # --------------------------------------------------------
    # CHROMA / EMBEDDING
    # --------------------------------------------------------

    log("")
    log(
        "[CHROMA] >>> add_documents() START"
    )

    log(
        "[CHROMA] This includes embedding "
        "generation + vector DB insertion."
    )

    chroma_start = (
        time.perf_counter()
    )

    try:

        vectorstore.add_documents(
            documents=documents,
            ids=ids,
        )

    except Exception:

        chroma_time = (
            time.perf_counter()
            - chroma_start
        )

        log(
            f"[CHROMA ERROR] "
            f"add_documents() failed "
            f"| elapsed={chroma_time:.3f}s"
        )

        traceback.print_exc()

        raise

    chroma_time = (
        time.perf_counter()
        - chroma_start
    )

    log(
        "[CHROMA] <<< add_documents() END"
    )

    log(
        f"[CHROMA] Total time: "
        f"{chroma_time:.3f}s"
    )

    if documents:

        log(
            f"[CHROMA] Time/document: "
            f"{chroma_time / len(documents):.3f}s"
        )

    log(
        "[CHROMA] INGESTION COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    main_start = (
        time.perf_counter()
    )

    log("")
    log("=" * 80)
    log("MAIN START")
    log("=" * 80)

    # --------------------------------------------------------
    # DATA DIRECTORY
    # --------------------------------------------------------

    data_path = Path(
        DATA_DIR
    )

    log(
        f"[MAIN] Data directory: "
        f"{data_path.resolve()}"
    )

    if not data_path.exists():

        raise RuntimeError(
            f"Data directory not found: "
            f"{DATA_DIR}"
        )

    # --------------------------------------------------------
    # FIND PDFS
    # --------------------------------------------------------

    log(
        "[MAIN] Searching for PDFs..."
    )

    pdf_files = list(
        data_path.glob(
            "*.pdf"
        )
    )

    log(
        f"[MAIN] Found "
        f"{len(pdf_files)} PDF(s)"
    )

    if not pdf_files:

        log(
            "[MAIN] No PDF files found."
        )

        return

    all_documents = []

    # --------------------------------------------------------
    # PROCESS EACH PDF
    # --------------------------------------------------------

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

        log(
            f"[MAIN] {pdf_file.name}"
        )

        log("=" * 80)

        file_start = (
            time.perf_counter()
        )

        try:

            documents = process_pdf(
                str(pdf_file)
            )

        except Exception:

            log_exception(
                f"[MAIN ERROR] "
                f"Failed processing "
                f"{pdf_file.name}"
            )

            raise

        all_documents.extend(
            documents
        )

        file_time = (
            time.perf_counter()
            - file_start
        )

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

    # ========================================================
    # SUMMARY
    # ========================================================

    text_count = sum(
        1
        for doc in all_documents
        if doc.metadata.get(
            "content_type"
        ) == "text"
    )

    figure_count = sum(
        1
        for doc in all_documents
        if doc.metadata.get(
            "content_type"
        ) == "figure"
    )

    table_count = sum(
        1
        for doc in all_documents
        if doc.metadata.get(
            "content_type"
        ) == "table"
    )

    log("")
    log("=" * 80)
    log("INGESTION SUMMARY")
    log("=" * 80)

    log(
        f"Text chunks : {text_count}"
    )

    log(
        f"Figures     : {figure_count}"
    )

    log(
        f"Tables      : {table_count}"
    )

    log(
        f"Total       : {len(all_documents)}"
    )

    log("=" * 80)

    # ========================================================
    # CHROMA
    # ========================================================

    log(
        "[MAIN] Starting Chroma ingestion..."
    )

    add_to_chroma(
        all_documents
    )

    # ========================================================
    # TOTAL
    # ========================================================

    total_time = (
        time.perf_counter()
        - main_start
    )

    log("")
    log("=" * 80)
    log("INGESTION COMPLETED")
    log("=" * 80)

    log(
        f"Total elapsed time: "
        f"{total_time:.3f}s"
    )

    log("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    log(
        "[BOOT] __main__ entered."
    )

    try:

        main()

    except KeyboardInterrupt:

        log(
            "[STOP] "
            "KeyboardInterrupt received."
        )

    except Exception:

        log_exception(
            "[FATAL] "
            "Application crashed."
        )

        raise

    finally:

        log(
            "[BOOT] Program exiting."
        )
