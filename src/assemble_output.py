import json
import logging
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
import time

import boto3
import fitz


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

s3_client = boto3.client("s3")
comprehend_client = boto3.client("comprehend")


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


def _get_file_size(bucket: str, key: str) -> int:
    """Get file size in bytes from S3."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response.get("ContentLength", 0)
    except Exception:
        return 0


def _calculate_compression_ratio(original_size: int, redacted_size: int) -> float:
    """Calculate compression ratio as percentage change."""
    if original_size == 0:
        return 0.0
    return ((original_size - redacted_size) / original_size) * 100


def _calculate_redaction_effectiveness(total_boxes: int, total_pages: int, findings: int) -> float:
    """Calculate redaction effectiveness score (0-100)."""
    if findings == 0 or total_pages == 0:
        return 100.0
    # Score based on how much of the detected PII was actually redacted
    effectiveness = (total_boxes / max(findings, 1)) * 100
    return min(100.0, effectiveness)


def _identify_suspicious_patterns(phrases_by_page: dict[int, set[str]]) -> list[dict]:
    """Identify pages with unusually high entity concentration."""
    suspicious = []
    avg_entities = sum(len(v) for v in phrases_by_page.values()) / max(len(phrases_by_page), 1)
    threshold = avg_entities * 2  # Flag pages with 2x average
    
    for page_num, phrases in phrases_by_page.items():
        if len(phrases) > threshold:
            suspicious.append({
                "page": page_num,
                "entity_count": len(phrases),
                "reason": "High concentration of PII entities",
                "recommendation": "Manual review recommended"
            })
    
    return suspicious


def _get_top_entities(phrases_by_page: dict[int, set[str]], limit: int = 10) -> list[dict]:
    """Get top N most frequently redacted phrases."""
    phrase_counts = Counter()
    for phrases in phrases_by_page.values():
        phrase_counts.update(phrases)
    
    return [
        {"phrase": phrase, "count": count}
        for phrase, count in phrase_counts.most_common(limit)
    ]


def _get_entity_concentration_by_page(phrases_by_page: dict[int, set[str]], entity_counts: dict) -> dict:
    """Analyze which pages have the most PII."""
    page_entity_counts = {str(page): len(phrases) for page, phrases in phrases_by_page.items()}
    rarest_types = sorted(entity_counts.items(), key=lambda x: x[1])[:3]
    
    return {
        "entities_per_page": page_entity_counts,
        "pages_with_most_pii": sorted(
            [(page, count) for page, count in page_entity_counts.items()],
            key=lambda x: int(x[1]),
            reverse=True
        )[:5],
        "rarest_entity_types": [{"type": et, "count": count} for et, count in rarest_types if count > 0],
    }


def lambda_handler(event: dict, _context: object) -> dict:
    execution_start = time.time()
    start_time_utc = datetime.now(UTC).isoformat()
    
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

    # Get original file size
    original_size = _get_file_size(input_bucket, input_key)
    
    local_input = Path(f"/tmp/{job_id}-input.pdf")
    local_output = Path(f"/tmp/{job_id}-redacted.pdf")

    s3_client.download_file(input_bucket, input_key, str(local_input))

    total_boxes = 0
    redactions_per_page: dict[str, int] = {}
    pages_with_redactions = 0
    time_per_page: dict[int, float] = {}

    with fitz.open(local_input) as document:
        total_pages_in_doc = len(document)
        
        for page in document:
            page_start = time.time()
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
            
            time_per_page[page_number] = time.time() - page_start

        document.save(str(local_output), garbage=4, deflate=True, clean=True)

    redacted_key = _build_output_key(input_key, input_prefix, output_prefix, "-redacted.pdf")
    report_key = _build_output_key(input_key, input_prefix, report_prefix, "-redaction-report.json")

    s3_client.upload_file(
        str(local_output),
        output_bucket,
        redacted_key,
        ExtraArgs={"ContentType": "application/pdf"},
    )

    # Get redacted file size
    redacted_size = _get_file_size(output_bucket, redacted_key)
    
    # Calculate execution metrics
    execution_duration = time.time() - execution_start
    estimated_chars = int(event.get("estimated_characters_processed", 0))
    processing_rate = estimated_chars / max(execution_duration, 0.001)  # chars/second
    
    # Calculate page metrics
    avg_time_per_page = sum(time_per_page.values()) / max(len(time_per_page), 1) if time_per_page else 0
    coverage_percentage = (pages_with_redactions / max(total_pages_in_doc, 1)) * 100
    redaction_density = total_boxes / max(pages_with_redactions, 1) if pages_with_redactions > 0 else 0
    
    # Calculate effectiveness and get analytics
    effectiveness_score = _calculate_redaction_effectiveness(total_boxes, total_pages_in_doc, findings_detected)
    compression_ratio = _calculate_compression_ratio(original_size, redacted_size)
    suspicious_patterns = _identify_suspicious_patterns(phrases_by_page)
    top_entities = _get_top_entities(phrases_by_page)
    concentration_analysis = _get_entity_concentration_by_page(phrases_by_page, dict(entities_by_type))
    
    # Entity distribution percentages
    total_entities = sum(entities_by_type.values())
    entity_distribution = {
        entity_type: round((count / max(total_entities, 1)) * 100, 2)
        for entity_type, count in entities_by_type.items()
    }
    
    # Compliance flags
    needs_review = len(suspicious_patterns) > 0 or effectiveness_score < 90
    
    report = {
        "job_id": job_id,
        "processing": {
            "start_time_utc": start_time_utc,
            "end_time_utc": datetime.now(UTC).isoformat(),
            "total_duration_seconds": round(execution_duration, 2),
            "total_pages_processed": total_pages_in_doc,
            "pages_with_redactions": pages_with_redactions,
            "coverage_percentage": round(coverage_percentage, 2),
        },
        "performance_metrics": {
            "processing_rate_chars_per_second": round(processing_rate, 2),
            "average_time_per_page_ms": round(avg_time_per_page * 1000, 2),
            "time_per_page_breakdown": {str(page): round(t * 1000, 2) for page, t in time_per_page.items()},
            "estimated_vs_actual_characters": {
                "estimated": estimated_chars,
                "average_per_page": round(estimated_chars / max(event.get("total_pages", 1), 1), 0) if event.get("total_pages") else 0,
            },
        },
        "file_metrics": {
            "original_file_size_bytes": original_size,
            "original_file_size_kb": round(original_size / 1024, 2),
            "redacted_file_size_bytes": redacted_size,
            "redacted_file_size_kb": round(redacted_size / 1024, 2),
            "compression_ratio_percent": round(compression_ratio, 2),
        },
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
            "estimated_characters_processed": estimated_chars,
            "map_max_concurrency": int(event.get("map_max_concurrency", 1)),
            "language_code": event.get("language_code", "en"),
            "min_entity_score": float(event.get("min_entity_score", 0.8)),
        },
        "pii_detection": {
            "total_findings_detected": findings_detected,
            "unique_phrases_detected": sum(len(values) for values in phrases_by_page.values()),
            "redaction_boxes_applied": total_boxes,
            "changes_made": total_boxes,
            "average_redactions_per_page": round(redaction_density, 2),
            "redactions_per_page": redactions_per_page,
            "entities_by_type": dict(entities_by_type),
            "entity_distribution_percentage": entity_distribution,
            "top_redacted_phrases": top_entities,
        },
        "quality_metrics": {
            "redaction_effectiveness_score": round(effectiveness_score, 2),
            "entity_concentration": concentration_analysis,
            "pages_with_high_pii_concentration": [s for s in suspicious_patterns],
        },
        "compliance": {
            "manual_review_recommended": needs_review,
            "review_reasons": [
                "High concentration of PII entities" if len(suspicious_patterns) > 0 else None,
                "Effectiveness score below 90%" if effectiveness_score < 90 else None,
            ],
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
        "effectiveness_score": round(effectiveness_score, 2),
        "work_objects_deleted": cleanup_deleted,
    }
    LOGGER.info("Assemble result: %s", json.dumps(result))
    return result
