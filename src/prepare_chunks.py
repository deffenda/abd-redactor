import json
import logging
import os
from pathlib import Path

import boto3
import fitz


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

s3_client = boto3.client("s3")


def _normalize_prefix(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return value if value.endswith("/") else f"{value}/"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid int for %s=%r, using default %s", name, raw, default)
        return default


DEFAULT_WORK_PREFIX = _normalize_prefix(os.getenv("WORK_PREFIX", "work/"))
DEFAULT_CHUNK_CHAR_LIMIT = _env_int("CHUNK_CHAR_LIMIT", 4500)
DEFAULT_CHUNKS_PER_BATCH = _env_int("CHUNKS_PER_BATCH", 20)


def chunk_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            window_start = min(start + int(max_chars * 0.6), end)
            split_at = text.rfind(" ", window_start, end)
            if split_at > start:
                end = split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end <= start:
            end = min(start + max_chars, text_len)
        start = end

    return chunks


def _write_json(bucket: str, key: str, payload: dict) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


def lambda_handler(event: dict, _context: object) -> dict:
    LOGGER.info("Preparing chunks for event: %s", json.dumps(event))

    job_id = event["job_id"]
    input_bucket = event["input_bucket"]
    input_key = event["input_key"]
    output_bucket = event.get("output_bucket") or input_bucket
    work_bucket = event.get("work_bucket") or output_bucket
    work_prefix = _normalize_prefix(event.get("work_prefix", DEFAULT_WORK_PREFIX))
    chunk_char_limit = int(event.get("chunk_char_limit", DEFAULT_CHUNK_CHAR_LIMIT))
    chunks_per_batch = max(int(event.get("chunks_per_batch", DEFAULT_CHUNKS_PER_BATCH)), 1)

    local_pdf = Path(f"/tmp/{job_id}-input.pdf")
    s3_client.download_file(input_bucket, input_key, str(local_pdf))

    batch_manifest_keys: list[str] = []
    current_batch: list[dict] = []

    total_pages = 0
    total_chunks = 0
    total_chars = 0
    batch_index = 0

    def flush_batch() -> None:
        nonlocal current_batch, batch_index
        if not current_batch:
            return
        batch_key = f"{work_prefix}{job_id}/batches/batch-{batch_index:05d}.json"
        _write_json(
            bucket=work_bucket,
            key=batch_key,
            payload={
                "job_id": job_id,
                "batch_index": batch_index,
                "input_bucket": input_bucket,
                "input_key": input_key,
                "chunks": current_batch,
            },
        )
        batch_manifest_keys.append(batch_key)
        batch_index += 1
        current_batch = []

    with fitz.open(local_pdf) as document:
        total_pages = document.page_count
        for page in document:
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            for chunk_index, chunk in enumerate(chunk_text(page_text, chunk_char_limit)):
                chunk_record = {
                    "page_number": page.number + 1,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
                current_batch.append(chunk_record)
                total_chunks += 1
                total_chars += len(chunk)

                if len(current_batch) >= chunks_per_batch:
                    flush_batch()

    flush_batch()

    result = {
        "job_id": job_id,
        "input_bucket": input_bucket,
        "input_key": input_key,
        "output_bucket": output_bucket,
        "work_bucket": work_bucket,
        "input_prefix": event.get("input_prefix", ""),
        "output_prefix": event.get("output_prefix", ""),
        "report_prefix": event.get("report_prefix", ""),
        "work_prefix": work_prefix,
        "min_entity_score": float(event.get("min_entity_score", 0.8)),
        "language_code": event.get("language_code", "en"),
        "map_max_concurrency": max(int(event.get("map_max_concurrency", 5)), 1),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "total_batches": len(batch_manifest_keys),
        "estimated_characters_processed": total_chars,
        "batch_manifest_keys": batch_manifest_keys,
    }
    LOGGER.info("Prepared chunk manifests: %s", json.dumps(result))
    return result
