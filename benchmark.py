"""
benchmark.py

Benchmark script for testing throughput and latency of the vLLM server.

For each request, it measures:
    - TTFT      : time until the first streamed token
    - Latency   : total time for the complete response
    - Tokens    : number of output tokens generated

For each concurrency level, it reports:
    - Requests/sec
    - Output tokens/sec
    - TTFT (mean, P50, P95, P99)
    - End-to-end latency (mean, P50, P95, P99)
    - Per-request output tokens/sec
    - Error count and error rate

Usage:
    python benchmark.py
    python benchmark.py --concurrency 1 10 20 50 --requests-per-level 40
    python benchmark.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ
    python benchmark.py --prompts-file my_prompts.txt --max-tokens 256

Notes:
    - Uses the same environment variables as rag.py and ingestion.py.
    - Requests are streamed so TTFT can be measured.
    - One warm-up request is sent before each concurrency level.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import os

from openai import AsyncOpenAI


# ============================================================
# CONFIG DEFAULTS (same env vars used elsewhere in the repo)
# ============================================================

load_dotenv()

DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")


# ============================================================
# LOGGING
# ============================================================

def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================================
# DEFAULT PROMPTS
# ============================================================
#
# Includes a few short questions and some longer prompts to make the
# benchmark closer to a normal RAG workload. Prompt length can affect
# TTFT and overall throughput.

SHORT_PROMPTS = [
    "What is the difference between supervised and unsupervised learning?",
    "Summarize the main idea of transfer learning in two sentences.",
    "What does the term 'overfitting' mean in machine learning?",
    "List three common evaluation metrics for classification models.",
    "Explain what a vector database is used for.",
]

LONG_CONTEXT_TEMPLATE = """You are a research-paper question answering assistant.
Answer ONLY using the context below and cite sources as [Sx].

Context:
[S1] The proposed method uses a two-stage retrieval pipeline combining
dense embeddings from a bi-encoder with sparse BM25 lexical matching,
fused via reciprocal rank fusion with a weighting of 0.7 dense / 0.3
sparse. Ablations show this improves recall@5 by 6.2 points over dense
retrieval alone on the evaluation benchmark. [S1]

[S2] The generation stage prompts an instruction-tuned 7B parameter
model served through vLLM with a max sequence length of 8192 tokens and
GPU memory utilization capped at 0.75 to leave headroom for concurrent
requests. Quantization via AWQ reduced memory footprint by approximately
60% with less than 1 point of degradation on downstream QA accuracy. [S2]

[S3] A self-verification step re-checks each generated answer against
the retrieved context using the same served model, rejecting answers
that reference information not present in context and falling back to
an explicit "not found" response rather than returning an unsupported
claim. This reduced hallucination rate from 14% to under 3% on a
held-out set of adversarial questions designed to probe for fabricated
citations. [S3]

Question: {question}
Answer (with citations):"""

LONG_QUESTIONS = [
    "What retrieval fusion weighting was used and why?",
    "How much did quantization reduce the model's memory footprint?",
    "What happens when the model's answer isn't supported by the context?",
    "What was the improvement in recall@5 from adding BM25?",
    "What sequence length and GPU memory settings does the server use?",
]


def default_prompts() -> List[str]:
    prompts = list(SHORT_PROMPTS)
    for q in LONG_QUESTIONS:
        prompts.append(LONG_CONTEXT_TEMPLATE.format(question=q))
    return prompts


def load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return default_prompts()

    text = Path(path).read_text(encoding="utf-8")
    prompts = [line.strip() for line in text.split("\n---\n") if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    log(f"[PROMPTS] Loaded {len(prompts)} prompt(s) from {path}")
    return prompts


# ============================================================
# RESULT RECORD
# ============================================================

@dataclass
class RequestResult:
    ok: bool
    ttft: float = 0.0
    latency: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""


@dataclass
class LevelResult:
    concurrency: int
    wall_time: float
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successes(self) -> List[RequestResult]:
        return [r for r in self.results if r.ok]

    @property
    def failures(self) -> List[RequestResult]:
        return [r for r in self.results if not r.ok]


# ============================================================
# SINGLE REQUEST
# ============================================================

async def run_single_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:

    start = time.perf_counter()
    first_token_time: Optional[float] = None
    output_tokens = 0
    prompt_tokens = 0

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:

            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_tokens += 1  # Use the usage count below when it is available

            # The last chunk contains usage information
            if getattr(chunk, "usage", None):
                if chunk.usage.completion_tokens:
                    output_tokens = chunk.usage.completion_tokens
                if chunk.usage.prompt_tokens:
                    prompt_tokens = chunk.usage.prompt_tokens

        end = time.perf_counter()

        if first_token_time is None:
            # No tokens were returned, so don't count this as a successful request.
            return RequestResult(ok=False, error="empty completion (no tokens streamed)")

        return RequestResult(
            ok=True,
            ttft=first_token_time - start,
            latency=end - start,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
        )

    except Exception as e:
        return RequestResult(ok=False, error=repr(e))


# ============================================================
# ONE CONCURRENCY LEVEL
# ============================================================

async def run_level(
    client: AsyncOpenAI,
    model: str,
    prompts: List[str],
    concurrency: int,
    requests_per_level: int,
    max_tokens: int,
) -> LevelResult:

    log("=" * 70)
    log(f"[LEVEL] concurrency={concurrency} requests={requests_per_level}")
    log("=" * 70)

    # Send one request first so the measured requests aren't affected
    # by the initial server setup.
    log("[WARMUP] Sending 1 untimed warm-up request...")
    await run_single_request(client, model, prompts[0], max_tokens)
    log("[WARMUP] Done.")

    # Reuse the prompts if more requests are needed than prompts available.
    queue = [prompts[i % len(prompts)] for i in range(requests_per_level)]

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(prompt: str) -> RequestResult:
        async with semaphore:
            return await run_single_request(client, model, prompt, max_tokens)

    start = time.perf_counter()
    results = await asyncio.gather(*(bounded(p) for p in queue))
    wall_time = time.perf_counter() - start

    ok_count = sum(1 for r in results if r.ok)
    log(
        f"[LEVEL] complete | ok={ok_count}/{len(results)} "
        f"| wall_time={wall_time:.2f}s"
    )

    return LevelResult(concurrency=concurrency, wall_time=wall_time, results=results)


# ============================================================
# STATS HELPERS
# ============================================================

def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    idx = min(max(int(p) - 1, 0), 98)
    return quantiles[idx]


def summarize(level: LevelResult) -> dict:

    successes = level.successes
    failures = level.failures

    ttfts = [r.ttft for r in successes]
    latencies = [r.latency for r in successes]
    per_req_tok_s = [
        (r.output_tokens / r.latency) for r in successes if r.latency > 0
    ]
    total_output_tokens = sum(r.output_tokens for r in successes)

    summary = {
        "concurrency": level.concurrency,
        "total_requests": len(level.results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "error_rate_pct": round(100 * len(failures) / len(level.results), 2) if level.results else 0.0,
        "wall_time_s": round(level.wall_time, 3),
        "requests_per_sec": round(len(successes) / level.wall_time, 3) if level.wall_time > 0 else 0.0,
        "aggregate_output_tokens_per_sec": round(total_output_tokens / level.wall_time, 2) if level.wall_time > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
        "ttft_mean_s": round(statistics.mean(ttfts), 4) if ttfts else 0.0,
        "ttft_p50_s": round(pct(ttfts, 50), 4) if ttfts else 0.0,
        "ttft_p95_s": round(pct(ttfts, 95), 4) if ttfts else 0.0,
        "ttft_p99_s": round(pct(ttfts, 99), 4) if ttfts else 0.0,
        "latency_mean_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "latency_p50_s": round(pct(latencies, 50), 4) if latencies else 0.0,
        "latency_p95_s": round(pct(latencies, 95), 4) if latencies else 0.0,
        "latency_p99_s": round(pct(latencies, 99), 4) if latencies else 0.0,
        "per_request_tok_s_mean": round(statistics.mean(per_req_tok_s), 2) if per_req_tok_s else 0.0,
        "per_request_tok_s_p50": round(pct(per_req_tok_s, 50), 2) if per_req_tok_s else 0.0,
        "sample_errors": list({r.error for r in failures})[:5],
    }

    return summary


# ============================================================
# OUTPUT: console table
# ============================================================

def print_summary_table(summaries: List[dict]) -> None:

    headers = [
        "Concurrency", "Req/s", "Tok/s", "TTFT p50", "TTFT p95",
        "Lat p50", "Lat p95", "Errors",
    ]

    rows = []
    for s in summaries:
        rows.append([
            str(s["concurrency"]),
            f"{s['requests_per_sec']:.2f}",
            f"{s['aggregate_output_tokens_per_sec']:.1f}",
            f"{s['ttft_p50_s']*1000:.0f}ms",
            f"{s['ttft_p95_s']*1000:.0f}ms",
            f"{s['latency_p50_s']:.2f}s",
            f"{s['latency_p95_s']:.2f}s",
            f"{s['failed_requests']}/{s['total_requests']}",
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) + 2
        for i in range(len(headers))
    ]

    def fmt_row(cells):
        return "".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print()
    print("=" * sum(widths))
    print("BENCHMARK SUMMARY")
    print("=" * sum(widths))
    print(fmt_row(headers))
    print("-" * sum(widths))
    for row in rows:
        print(fmt_row(row))
    print("=" * sum(widths))
    print()


def markdown_table(summaries: List[dict], model: str) -> str:

    lines = [
        f"### vLLM Benchmark — `{model}`",
        "",
        "| Concurrency | Req/s | Output tok/s | TTFT (p50) | TTFT (p95) | Latency (p50) | Latency (p95) | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for s in summaries:
        lines.append(
            f"| {s['concurrency']} "
            f"| {s['requests_per_sec']:.2f} "
            f"| {s['aggregate_output_tokens_per_sec']:.1f} "
            f"| {s['ttft_p50_s']*1000:.0f} ms "
            f"| {s['ttft_p95_s']*1000:.0f} ms "
            f"| {s['latency_p50_s']:.2f} s "
            f"| {s['latency_p95_s']:.2f} s "
            f"| {s['failed_requests']}/{s['total_requests']} |"
        )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

async def main_async(args: argparse.Namespace) -> None:

    log("=" * 70)
    log("vLLM BENCHMARK")
    log(f"[CONFIG] base_url          : {args.base_url}")
    log(f"[CONFIG] model             : {args.model}")
    log(f"[CONFIG] concurrency levels: {args.concurrency}")
    log(f"[CONFIG] requests/level    : {args.requests_per_level}")
    log(f"[CONFIG] max_tokens        : {args.max_tokens}")
    log("=" * 70)

    prompts = load_prompts(args.prompts_file)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    summaries = []

    for concurrency in args.concurrency:
        level = await run_level(
            client=client,
            model=args.model,
            prompts=prompts,
            concurrency=concurrency,
            requests_per_level=args.requests_per_level,
            max_tokens=args.max_tokens,
        )
        summaries.append(summarize(level))

    print_summary_table(summaries)

    md = markdown_table(summaries, args.model)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        log(f"[OUTPUT] JSON written to {args.output_json}")

    if args.output_markdown:
        Path(args.output_markdown).write_text(md, encoding="utf-8")
        log(f"[OUTPUT] Markdown table written to {args.output_markdown}")
        log("[OUTPUT] Paste this file's contents directly into the README benchmark section.")

    for s in summaries:
        if s["failed_requests"] and s["sample_errors"]:
            log(f"[ERRORS] concurrency={s['concurrency']} sample: {s['sample_errors']}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Throughput/latency benchmark for the vLLM OpenAI-compatible server."
    )

    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"vLLM OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name as registered with vLLM (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY,
        help="API key (vLLM ignores the value but the client requires one).",
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 10, 20, 50],
        help="Concurrency levels to test in sequence (default: 1 10 20 50).",
    )
    parser.add_argument(
        "--requests-per-level", type=int, default=40,
        help="Number of requests to send at each concurrency level (default: 40).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="max_tokens per generation request (default: 256).",
    )
    parser.add_argument(
        "--prompts-file", default=None,
        help="Optional path to a text file of prompts separated by lines "
             "containing only '---'. Defaults to a built-in mix of short "
             "and long-context prompts.",
    )
    parser.add_argument(
        "--output-json", default="benchmark_results.json",
        help="Path to write full results as JSON (default: benchmark_results.json). "
             "Pass '' to skip.",
    )
    parser.add_argument(
        "--output-markdown", default="benchmark_results.md",
        help="Path to write a ready-to-paste markdown table (default: benchmark_results.md). "
             "Pass '' to skip.",
    )

    args = parser.parse_args()

    args.output_json = args.output_json or None
    args.output_markdown = args.output_markdown or None

    return args


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.run(main_async(parsed))
