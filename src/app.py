import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote_plus

import boto3
import fitz  # PyMuPDF


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

s3_client = boto3.client("s3")
comprehend_client = boto3.client("comprehend")

INPUT_PREFIX = os.getenv("INPUT_PREFIX", "incoming/")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "redacted/")
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "")
def _env_float(name: str, default: float) -> float:
    """Read an environment variable and parse it as float, returning
    `default` when the variable is missing, empty, or invalid.
    """
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid float for %s: %r — using default %s", name, val, default)
        return default

MIN_ENTITY_SCORE = _env_float("MIN_ENTITY_SCORE", 0.8)
COMPREHEND_LANGUAGE = os.getenv("COMPREHEND_LANGUAGE", "en")
MAX_COMPREHEND_TEXT_LEN = int(os.getenv("MAX_COMPREHEND_TEXT_LEN", "4500"))


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str


def chunk_text(text: str, max_len: int = MAX_COMPREHEND_TEXT_LEN) -> Iterable[tuple[str, int]]:
    """Yield chunks with their absolute start offsets in the full string."""
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        yield text[start:end], start
        start = end


def detect_pii_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    for chunk, _ in chunk_text(text):
        if not chunk.strip():
            continue
        response = comprehend_client.detect_pii_entities(
            Text=chunk,
            LanguageCode=COMPREHEND_LANGUAGE,
        )
        for entity in response.get("Entities", []):
            score = float(entity.get("Score", 0.0))
            if score < MIN_ENTITY_SCORE:
                continue
            begin = int(entity.get("BeginOffset", 0))
            end = int(entity.get("EndOffset", 0))
            if begin >= end:
                continue
            phrase = chunk[begin:end].strip()
            if len(phrase) >= 2:
                phrases.add(phrase)
    return phrases


def redact_page(page: fitz.Page) -> int:
    text = page.get_text("text")
    if not text.strip():
        return 0

    pii_phrases = detect_pii_phrases(text)
    if not pii_phrases:
        return 0

    hits = 0
    seen_rects: set[tuple[float, float, float, float]] = set()

    # Longer phrases first reduces partial-overlap duplicates.
    for phrase in sorted(pii_phrases, key=len, reverse=True):
        rects = page.search_for(phrase, quads=False)
        for rect in rects:
            rect_key = (round(rect.x0, 3), round(rect.y0, 3), round(rect.x1, 3), round(rect.y1, 3))
            if rect_key in seen_rects:
                continue
            seen_rects.add(rect_key)
            page.add_redact_annot(rect, fill=(0, 0, 0))
            hits += 1

    if hits:
        page.apply_redactions()
    return hits


def redact_pdf(input_path: Path, output_path: Path) -> int:
    total_hits = 0
    with fitz.open(input_path) as doc:
        for page in doc:
            total_hits += redact_page(page)
        doc.save(output_path, garbage=4, deflate=True, clean=True)
    return total_hits


def parse_s3_records(event: dict) -> list[S3ObjectRef]:
    objects: list[S3ObjectRef] = []
    for record in event.get("Records", []):
        if record.get("eventSource") != "aws:s3":
            continue
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        objects.append(S3ObjectRef(bucket=bucket, key=key))
    return objects


def build_output_key(input_key: str) -> str:
    relative = input_key[len(INPUT_PREFIX) :] if input_key.startswith(INPUT_PREFIX) else Path(input_key).name
    stem = str(Path(relative).with_suffix(""))
    return f"{OUTPUT_PREFIX}{stem}-redacted.pdf"


def lambda_handler(event: dict, _context: object) -> dict:
    LOGGER.info("Received event: %s", json.dumps(event))
    records = parse_s3_records(event)
    if not records:
        return {"processed": 0, "results": []}

    results = []
    for record in records:
        if not record.key.lower().endswith(".pdf"):
            LOGGER.info("Skipping non-PDF key: %s", record.key)
            continue
        if record.key.startswith(OUTPUT_PREFIX):
            LOGGER.info("Skipping output-prefix key to avoid recursion: %s", record.key)
            continue

        output_bucket = OUTPUT_BUCKET or record.bucket
        output_key = build_output_key(record.key)

        local_in = Path("/tmp/input.pdf")
        local_out = Path("/tmp/output-redacted.pdf")

        LOGGER.info("Downloading s3://%s/%s", record.bucket, record.key)
        s3_client.download_file(record.bucket, record.key, str(local_in))

        hits = redact_pdf(local_in, local_out)

        LOGGER.info("Uploading redacted file to s3://%s/%s", output_bucket, output_key)
        s3_client.upload_file(
            str(local_out),
            output_bucket,
            output_key,
            ExtraArgs={"ContentType": "application/pdf"},
        )

        result = {
            "input_bucket": record.bucket,
            "input_key": record.key,
            "output_bucket": output_bucket,
            "output_key": output_key,
            "redaction_hits": hits,
        }
        results.append(result)
        LOGGER.info("Processed PDF result: %s", json.dumps(result))

    return {"processed": len(results), "results": results}
