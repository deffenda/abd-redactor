import json
import logging
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
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


def _load_json(bucket: str, key: str) -> dict:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def _upload_json(bucket: str, key: str, payload: dict) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _build_output_key(input_key: str, input_prefix: str, output_prefix: str, suffix: str) -> str:
    input_prefix = _normalize_prefix(input_prefix)
    output_prefix = _normalize_prefix(output_prefix)

    relative = input_key[len(input_prefix) :] if input_prefix and input_key.startswith(input_prefix) else Path(input_key).name
    stem = str(Path(relative).with_suffix(""))
    return f"{output_prefix}{stem}{suffix}"


def _delete_prefix(bucket: str, prefix: str) -> int:
    deleted = 0
    token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3_client.list_objects_v2(**kwargs)
        keys = [{"Key": item["Key"]} for item in response.get("Contents", [])]
        if keys:
            s3_client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            deleted += len(keys)

        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")

    return deleted


def lambda_handler(event: dict, _context: object) -> dict:
    LOGGER.info("Assembling redacted output for event: %s", json.dumps(event))

    job_id = event["job_id"]
    input_bucket = event["input_bucket"]
    input_key = event["input_key"]
    output_bucket = event.get("output_bucket") or input_bucket
    work_bucket = event.get("work_bucket") or output_bucket

    input_prefix = event.get("input_prefix", os.getenv("INPUT_PREFIX", "incoming/"))
    output_prefix = event.get("output_prefix", os.getenv("OUTPUT_PREFIX", "redacted/"))
    report_prefix = event.get("report_prefix", os.getenv("REPORT_PREFIX", "reports/"))
    work_prefix = _normalize_prefix(event.get("work_prefix", os.getenv("WORK_PREFIX", "work/")))

    batch_results = event.get("batch_results", [])

    phrases_by_page: dict[int, set[str]] = defaultdict(set)
    entities_by_type: Counter[str] = Counter()
    findings_detected = 0

    for batch in batch_results:
        finding_key = batch.get("finding_key")
        if not finding_key:
            continue

        finding_payload = _load_json(work_bucket, finding_key)
        for finding in finding_payload.get("findings", []):
            page_number = int(finding.get("page_number", 0))
            phrase = str(finding.get("phrase", "")).strip()
            if page_number <= 0 or len(phrase) < 2:
                continue
            phrases_by_page[page_number].add(phrase)
            entities_by_type[str(finding.get("entity_type", "UNKNOWN"))] += 1
            findings_detected += 1

    local_input = Path(f"/tmp/{job_id}-input.pdf")
    local_output = Path(f"/tmp/{job_id}-redacted.pdf")

    s3_client.download_file(input_bucket, input_key, str(local_input))

    total_boxes = 0
    redactions_per_page: dict[str, int] = {}
    pages_with_redactions = 0

    with fitz.open(local_input) as document:
        for page in document:
            page_number = page.number + 1
            phrases = sorted(phrases_by_page.get(page_number, set()), key=len, reverse=True)
            if not phrases:
                continue

            page_hits = 0
            seen_rects: set[tuple[float, float, float, float]] = set()

            for phrase in phrases:
                for rect in page.search_for(phrase, quads=False):
                    rect_key = (round(rect.x0, 3), round(rect.y0, 3), round(rect.x1, 3), round(rect.y1, 3))
                    if rect_key in seen_rects:
                        continue
                    seen_rects.add(rect_key)
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    page_hits += 1

            if page_hits:
                page.apply_redactions()
                pages_with_redactions += 1
                total_boxes += page_hits
                redactions_per_page[str(page_number)] = page_hits

        document.save(str(local_output), garbage=4, deflate=True, clean=True)

    redacted_key = _build_output_key(input_key, input_prefix, output_prefix, "-redacted.pdf")
    report_key = _build_output_key(input_key, input_prefix, report_prefix, "-redaction-report.json")

    s3_client.upload_file(
        str(local_output),
        output_bucket,
        redacted_key,
        ExtraArgs={"ContentType": "application/pdf"},
    )

    report = {
        "job_id": job_id,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "input": {
            "bucket": input_bucket,
            "key": input_key,
        },
        "output": {
            "bucket": output_bucket,
            "redacted_pdf_key": redacted_key,
            "report_key": report_key,
        },
        "pipeline": {
            "total_batches": int(event.get("total_batches", 0)),
            "total_chunks": int(event.get("total_chunks", 0)),
            "total_pages": int(event.get("total_pages", 0)),
            "estimated_characters_processed": int(event.get("estimated_characters_processed", 0)),
            "map_max_concurrency": int(event.get("map_max_concurrency", 1)),
            "language_code": event.get("language_code", "en"),
            "min_entity_score": float(event.get("min_entity_score", 0.8)),
        },
        "results": {
            "pii_findings_detected": findings_detected,
            "unique_phrases_detected": sum(len(values) for values in phrases_by_page.values()),
            "pages_with_redactions": pages_with_redactions,
            "redaction_boxes_applied": total_boxes,
            "changes_made": total_boxes,
            "redactions_per_page": redactions_per_page,
            "entities_by_type": dict(entities_by_type),
        },
    }

    _upload_json(output_bucket, report_key, report)

    cleanup_deleted = _delete_prefix(work_bucket, f"{work_prefix}{job_id}/")

    result = {
        "job_id": job_id,
        "status": "COMPLETED",
        "output_bucket": output_bucket,
        "redacted_pdf_key": redacted_key,
        "report_key": report_key,
        "changes_made": total_boxes,
        "work_objects_deleted": cleanup_deleted,
    }
    LOGGER.info("Assemble result: %s", json.dumps(result))
    return result
