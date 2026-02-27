import json
import logging
import os
import uuid
from urllib.parse import unquote_plus

import boto3


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

sfn_client = boto3.client("stepfunctions")


def _normalize_prefix(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return value if value.endswith("/") else f"{value}/"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid float for %s=%r, using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid int for %s=%r, using default %s", name, raw, default)
        return default


STATE_MACHINE_ARN = os.getenv("STATE_MACHINE_ARN", "")
INPUT_PREFIX = _normalize_prefix(os.getenv("INPUT_PREFIX", "incoming/"))
OUTPUT_PREFIX = _normalize_prefix(os.getenv("OUTPUT_PREFIX", "redacted/"))
REPORT_PREFIX = _normalize_prefix(os.getenv("REPORT_PREFIX", "reports/"))
WORK_PREFIX = _normalize_prefix(os.getenv("WORK_PREFIX", "work/"))
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "")
from .detect_pii_settings import MIN_ENTITY_SCORE, COMPREHEND_LANGUAGE, MAX_COMPREHEND_TEXT_LEN
CHUNK_CHAR_LIMIT = _env_int("CHUNK_CHAR_LIMIT", 4500)
CHUNKS_PER_BATCH = _env_int("CHUNKS_PER_BATCH", 20)
MAP_MAX_CONCURRENCY = _env_int("MAP_MAX_CONCURRENCY", 5)


def _extract_s3_objects(event: dict) -> list[tuple[str, str]]:
    objects: list[tuple[str, str]] = []

    if event.get("Records"):
        for record in event.get("Records", []):
            if record.get("eventSource") != "aws:s3":
                continue
            bucket = record.get("s3", {}).get("bucket", {}).get("name")
            key = record.get("s3", {}).get("object", {}).get("key")
            if bucket and key:
                objects.append((bucket, unquote_plus(key)))
        return objects

    detail = event.get("detail", {})
    bucket = detail.get("bucket", {}).get("name")
    key = detail.get("object", {}).get("key")
    if bucket and key:
        objects.append((bucket, unquote_plus(key)))

    return objects


def _is_source_pdf(key: str) -> bool:
    lowered = key.lower()
    if not lowered.endswith(".pdf"):
        return False
    if INPUT_PREFIX and not key.startswith(INPUT_PREFIX):
        return False
    if OUTPUT_PREFIX and key.startswith(OUTPUT_PREFIX):
        return False
    if REPORT_PREFIX and key.startswith(REPORT_PREFIX):
        return False
    if WORK_PREFIX and key.startswith(WORK_PREFIX):
        return False
    return True


def _build_execution_input(bucket: str, key: str) -> dict:
    job_id = uuid.uuid4().hex
    return {
        "job_id": job_id,
        "input_bucket": bucket,
        "input_key": key,
        "output_bucket": OUTPUT_BUCKET or bucket,
        "work_bucket": OUTPUT_BUCKET or bucket,
        "input_prefix": INPUT_PREFIX,
        "output_prefix": OUTPUT_PREFIX,
        "report_prefix": REPORT_PREFIX,
        "work_prefix": WORK_PREFIX,
        "min_entity_score": MIN_ENTITY_SCORE,
        "language_code": COMPREHEND_LANGUAGE,
        "chunk_char_limit": CHUNK_CHAR_LIMIT,
        "chunks_per_batch": CHUNKS_PER_BATCH,
        "map_max_concurrency": max(MAP_MAX_CONCURRENCY, 1),
    }


def lambda_handler(event: dict, _context: object) -> dict:
    LOGGER.info("Received event: %s", json.dumps(event))

    if not STATE_MACHINE_ARN:
        raise RuntimeError("STATE_MACHINE_ARN environment variable is required")

    started = []
    skipped = []

    for bucket, key in _extract_s3_objects(event):
        if not _is_source_pdf(key):
            skipped.append({"bucket": bucket, "key": key, "reason": "not_source_pdf"})
            continue

        execution_input = _build_execution_input(bucket, key)
        execution_name = f"pdf-redact-{execution_input['job_id']}"
        response = sfn_client.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps(execution_input),
        )
        started.append(
            {
                "bucket": bucket,
                "key": key,
                "execution_arn": response["executionArn"],
                "job_id": execution_input["job_id"],
            }
        )

    result = {
        "started_count": len(started),
        "started": started,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    LOGGER.info("Start execution result: %s", json.dumps(result))
    return result
