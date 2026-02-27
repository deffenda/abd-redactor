import json
import logging
import os
import uuid
from urllib.parse import unquote_plus

import boto3


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# AWS-specific: boto3 resolves credentials from the runtime environment (IAM role/env/profile).
# This client requires `states:StartExecution` on the configured state machine ARN.
# NIST 800-53 alignment:
# - AC-6: keep role policy scoped to this workflow ARN only.
# - AU-2/AU-12: rely on CloudTrail + CloudWatch logs for invocation audit trail.
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    LOGGER.warning("Invalid bool for %s=%r, using default %s", name, raw, default)
    return default


STATE_MACHINE_ARN = os.getenv("STATE_MACHINE_ARN", "")
INPUT_PREFIX = _normalize_prefix(os.getenv("INPUT_PREFIX", "incoming/"))
OUTPUT_PREFIX = _normalize_prefix(os.getenv("OUTPUT_PREFIX", "redacted/"))
REPORT_PREFIX = _normalize_prefix(os.getenv("REPORT_PREFIX", "reports/"))
WORK_PREFIX = _normalize_prefix(os.getenv("WORK_PREFIX", "work/"))
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "")
try:
    from .detect_pii_settings import MIN_ENTITY_SCORE, COMPREHEND_LANGUAGE, MAX_COMPREHEND_TEXT_LEN
except ImportError:
    from detect_pii_settings import MIN_ENTITY_SCORE, COMPREHEND_LANGUAGE, MAX_COMPREHEND_TEXT_LEN
CHUNK_CHAR_LIMIT = _env_int("CHUNK_CHAR_LIMIT", 4500)
CHUNKS_PER_BATCH = _env_int("CHUNKS_PER_BATCH", 20)
MAP_MAX_CONCURRENCY = _env_int("MAP_MAX_CONCURRENCY", 5)
ENABLE_S3_AUTHORSHIP = _env_bool("ENABLE_S3_AUTHORSHIP", True)
ENABLE_S3_DOCUMENT_SUMMARY = _env_bool("ENABLE_S3_DOCUMENT_SUMMARY", True)
REQUIRE_S3_CAPABILITIES = _env_bool("REQUIRE_S3_CAPABILITIES", False)
S3_AUTHORSHIP_DETECTOR = os.getenv("S3_AUTHORSHIP_DETECTOR", "heuristic").strip().lower() or "heuristic"
S3_AUTHORSHIP_MODEL = os.getenv("S3_AUTHORSHIP_MODEL", "").strip()
S3_SUMMARY_MODEL = os.getenv("S3_SUMMARY_MODEL", "").strip()
S3_SUMMARY_DIRECTIONS = os.getenv("S3_SUMMARY_DIRECTIONS", "").strip()


def _extract_s3_objects(event: dict) -> list[tuple[str, str]]:
    # Supports both native S3 event records and EventBridge "Object Created" events.
    # If your trigger source changes, extend this parser to match the new event shape.
    # AU-3: preserve source bucket/key metadata through the workflow for traceability.
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
    # This payload is the contract with the Step Functions workflow.
    # If you add/rename fields here, update the state machine definition and downstream lambdas.
    # CM-3/CM-6: treat this schema as controlled configuration and version changes with IaC updates.
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
        "enable_s3_authorship": ENABLE_S3_AUTHORSHIP,
        "enable_s3_document_summary": ENABLE_S3_DOCUMENT_SUMMARY,
        "require_s3_capabilities": REQUIRE_S3_CAPABILITIES,
        "s3_authorship_detector": S3_AUTHORSHIP_DETECTOR,
        "s3_authorship_model_name": S3_AUTHORSHIP_MODEL,
        "s3_summary_model_name": S3_SUMMARY_MODEL,
        "s3_summary_directions": S3_SUMMARY_DIRECTIONS,
    }


def lambda_handler(event: dict, _context: object) -> dict:
    LOGGER.info("Received event: %s", json.dumps(event))

    if not STATE_MACHINE_ARN:
        # In AWS, set STATE_MACHINE_ARN via SAM/CloudFormation env vars.
        # CM-2: missing env configuration indicates deployment drift/misconfiguration.
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
