import json
import logging
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

import boto3
import fitz

from ai_authorship import DETECTOR_MODEL, analyze_file_authorship, normalize_detector
from document_summary import generate_document_summary_file, generate_document_summary_report


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# AWS-specific clients:
# - s3_client requires read/write/delete permissions on input/output/work prefixes.
# - keep at module scope so tests can monkeypatch with stub clients.
# NIST references:
# - AC-6 least privilege: scope roles to required bucket prefixes.
# - AU-9 protection of audit information: protect report/work artifacts from tampering.
s3_client = boto3.client("s3")
comprehend_client = boto3.client("comprehend")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


DEFAULT_S3_AUTHORSHIP_DETECTOR = os.getenv("S3_AUTHORSHIP_DETECTOR", "heuristic").strip().lower() or "heuristic"
DEFAULT_S3_AUTHORSHIP_MODEL = os.getenv("S3_AUTHORSHIP_MODEL", "").strip() or None
DEFAULT_S3_SUMMARY_MODEL = os.getenv("S3_SUMMARY_MODEL", "").strip() or None
DEFAULT_S3_SUMMARY_DIRECTIONS = os.getenv("S3_SUMMARY_DIRECTIONS", "").strip()
DEFAULT_ENABLE_S3_AUTHORSHIP = _env_bool("ENABLE_S3_AUTHORSHIP", True)
DEFAULT_ENABLE_S3_DOCUMENT_SUMMARY = _env_bool("ENABLE_S3_DOCUMENT_SUMMARY", True)
DEFAULT_REQUIRE_S3_CAPABILITIES = _env_bool("REQUIRE_S3_CAPABILITIES", False)


def _normalize_prefix(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return value if value.endswith("/") else f"{value}/"


def _load_json(bucket: str, key: str) -> dict:
    # Reads intermediate/final artifacts from S3; in AWS this depends on `s3:GetObject`.
    # AU-6: this input should come from trusted workflow output prefixes only.
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def _upload_json(bucket: str, key: str, payload: dict) -> None:
    # Writes report artifacts to S3; in AWS this depends on `s3:PutObject`.
    # SC-28: ensure bucket/object encryption meets your boundary requirement (typically KMS CMK).
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


def _run_s3_capabilities(
    *,
    local_input: Path,
    input_key: str,
    input_prefix: str,
    output_bucket: str,
    report_prefix: str,
    include_authorship: bool,
    include_document_summary: bool,
    require_all_capabilities: bool,
    authorship_detector: str,
    authorship_model_name: str | None,
    document_summary_model_name: str | None,
    document_summary_directions: str | None,
) -> dict[str, Any]:
    capability_outputs: dict[str, Any] = {}

    if include_authorship:
        try:
            selected_detector = normalize_detector(authorship_detector)
            selected_model = authorship_model_name if selected_detector == DETECTOR_MODEL else None
            authorship_result, extracted_text = analyze_file_authorship(
                local_input,
                filename=Path(input_key).name,
                detector=selected_detector,
                model_name=selected_model,
            )
            authorship_report_key = _build_output_key(
                input_key,
                input_prefix,
                report_prefix,
                "-authorship-report.json",
            )
            _upload_json(
                output_bucket,
                authorship_report_key,
                {
                    "input_key": input_key,
                    "detector_used": selected_detector,
                    "model_name_used": selected_model,
                    "label": authorship_result.label,
                    "ai_probability": authorship_result.ai_probability,
                    "confidence": authorship_result.confidence,
                    "summary": authorship_result.summary,
                    "features": authorship_result.features.__dict__,
                    "text_preview": extracted_text[:600],
                    "text_preview_truncated": len(extracted_text) > 600,
                },
            )
            capability_outputs["authorship"] = {
                "status": "COMPLETED",
                "report_key": authorship_report_key,
                "detector_used": selected_detector,
                "model_name_used": selected_model,
                "label": authorship_result.label,
                "ai_probability": authorship_result.ai_probability,
            }
        except Exception as exc:
            capability_outputs["authorship"] = {"status": "FAILED", "error": str(exc)}
            if require_all_capabilities:
                raise
    else:
        capability_outputs["authorship"] = {"status": "SKIPPED"}

    if include_document_summary:
        summary_output_path: Path | None = None
        try:
            summary_report_payload, summary_metrics = generate_document_summary_report(
                local_input,
                filename=Path(input_key).name,
                model_name=document_summary_model_name,
                additional_directions=document_summary_directions,
            )
            summary_output_path, summary_metrics = generate_document_summary_file(
                local_input,
                filename=Path(input_key).name,
                model_name=document_summary_model_name,
                additional_directions=document_summary_directions,
            )
            summary_pdf_key = _build_output_key(
                input_key,
                input_prefix,
                report_prefix,
                "-document-summary.pdf",
            )
            s3_client.upload_file(
                str(summary_output_path),
                output_bucket,
                summary_pdf_key,
                ExtraArgs={"ContentType": "application/pdf"},
            )
            summary_report_key = _build_output_key(
                input_key,
                input_prefix,
                report_prefix,
                "-document-summary-report.json",
            )
            _upload_json(
                output_bucket,
                summary_report_key,
                {
                    "input_key": input_key,
                    "summary_pdf_key": summary_pdf_key,
                    **summary_report_payload,
                },
            )
            capability_outputs["document_summary"] = {
                "status": "COMPLETED",
                "summary_pdf_key": summary_pdf_key,
                "report_key": summary_report_key,
                **summary_metrics,
            }
        except Exception as exc:
            capability_outputs["document_summary"] = {"status": "FAILED", "error": str(exc)}
            if require_all_capabilities:
                raise
        finally:
            if summary_output_path and summary_output_path.exists():
                summary_output_path.unlink(missing_ok=True)
    else:
        capability_outputs["document_summary"] = {"status": "SKIPPED"}

    return capability_outputs


def _delete_prefix(bucket: str, prefix: str) -> int:
    # Cleanup requires `s3:ListBucket` + `s3:DeleteObject` on the work prefix.
    # MP-6/SI-12: cleanup supports minimizing retained transient sensitive data.
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


def _calculate_detection_accuracy(findings_detected: int, total_boxes: int, confidence_scores: list[float] = None) -> dict:
    """Calculate precision, recall, and F1 score metrics."""
    if confidence_scores is None:
        confidence_scores = []

    findings = max(findings_detected, 0)
    boxes = max(total_boxes, 0)
    matched = min(findings, boxes)

    # Precision: proportion of applied redaction boxes backed by detected findings.
    precision = (matched / boxes) * 100 if boxes > 0 else 0.0

    # Recall: proportion of findings that were converted into redaction boxes.
    recall = (matched / findings) * 100 if findings > 0 else 0.0

    # F1 Score: harmonic mean
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    # Confidence analysis
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.8
    confidence_std_dev = (sum((s - avg_confidence) ** 2 for s in confidence_scores) / len(confidence_scores)) ** 0.5 if confidence_scores else 0.0
    
    return {
        "precision_percent": round(min(100.0, precision), 2),
        "recall_percent": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "average_confidence_score": round(avg_confidence, 3),
        "confidence_score_std_dev": round(confidence_std_dev, 3),
    }


def _calculate_document_quality(total_pages: int, pages_with_redactions: int, original_size: int, redacted_size: int) -> dict:
    """Calculate readability and legibility metrics."""
    pages_without_redactions = max(total_pages - pages_with_redactions, 0)
    readability_score = (pages_without_redactions / max(total_pages, 1)) * 100
    
    # Estimate information preservation
    size_ratio = redacted_size / max(original_size, 1)
    info_preservation = max(0, 100 - (size_ratio * 50))  # Rough approximation
    
    # Document usability index
    usability_score = (readability_score * 0.4 + info_preservation * 0.6)
    
    return {
        "readability_score": round(readability_score, 2),
        "information_preservation_percent": round(info_preservation, 2),
        "document_usability_index": round(usability_score, 2),
        "pages_with_complete_content": pages_without_redactions,
        "content_density_change": round((redacted_size - original_size) / max(original_size, 1) * 100, 2),
    }


def _calculate_consistency_metrics(phrases_by_page: dict[int, set[str]], redactions_per_page: dict[str, int]) -> dict:
    """Calculate redaction consistency across pages."""
    all_phrases = []
    for phrases in phrases_by_page.values():
        all_phrases.extend(phrases)
    
    phrase_counts = Counter(all_phrases)
    
    # Duplicate coverage: how many repeated phrases are all redacted
    repeated_phrases = {phrase: count for phrase, count in phrase_counts.items() if count > 1}
    duplicate_coverage = len(repeated_phrases) / max(len(phrase_counts), 1) * 100 if phrase_counts else 0
    
    # Redaction uniformity: consistency across pages
    redaction_counts = [v for v in redactions_per_page.values()]
    if redaction_counts:
        avg_redactions = sum(redaction_counts) / len(redaction_counts)
        uniformity_score = 100 - (sum(abs(r - avg_redactions) for r in redaction_counts) / (len(redaction_counts) * max(avg_redactions, 1)) * 100)
    else:
        uniformity_score = 100.0
    
    return {
        "duplicate_coverage_percent": round(duplicate_coverage, 2),
        "redaction_uniformity_score": round(max(0, uniformity_score), 2),
        "total_unique_phrases": len(phrase_counts),
        "repeated_phrases_detected": len(repeated_phrases),
    }


def _calculate_compliance_scores() -> dict:
    """Calculate HIPAA, GDPR, and other compliance scores."""
    # These values are heuristic placeholders and should be replaced with
    # policy-specific rules mapped to validated controls.
    return {
        "hipaa_estimated_compliance": 95.0,  # High confidence if medical PII present
        "gdpr_estimated_compliance": 92.0,   # High confidence if personal data redacted
        "sox_estimated_compliance": 88.0,    # Based on financial data protection
        "pci_dss_estimated_compliance": 85.0,  # Credit card protection level
        "nist_cybersecurity_level": "Level 3 - Protected",
    }


def _calculate_risk_assessment(total_boxes: int, findings_detected: int, pages_with_redactions: int, total_pages: int) -> dict:
    """Calculate residual PII risk and re-identification risk."""
    findings = max(findings_detected, 0)
    boxes = max(total_boxes, 0)
    if findings == 0:
        redaction_coverage = 100.0
    else:
        redaction_coverage = (min(boxes, findings) / findings) * 100
    redaction_coverage = min(100.0, max(0.0, redaction_coverage))
    unredacted_risk = 100.0 - redaction_coverage

    # Re-identification risk based on coverage gaps
    page_coverage = (pages_with_redactions / max(total_pages, 1)) * 100
    page_coverage = min(100.0, max(0.0, page_coverage))
    re_identification_risk = 100 - page_coverage

    # Overall risk score
    overall_risk = (unredacted_risk * 0.6 + re_identification_risk * 0.4)
    overall_risk = min(100.0, max(0.0, overall_risk))

    risk_level = "LOW" if overall_risk < 20 else "MEDIUM" if overall_risk < 50 else "HIGH"

    return {
        "residual_pii_risk_percent": round(unredacted_risk, 2),
        "re_identification_risk_percent": round(re_identification_risk, 2),
        "overall_risk_score": round(overall_risk, 2),
        "risk_level": risk_level,
        "audit_confidence_percent": round(100 - overall_risk, 2),
    }


def _calculate_entity_specific_metrics(entities_by_type: Counter[str], entity_confidence: dict = None) -> dict:
    """Calculate per-entity-type quality metrics."""
    if entity_confidence is None:
        entity_confidence = {}
    
    total_entities = sum(entities_by_type.values())
    
    entity_metrics = {}
    for entity_type, count in entities_by_type.items():
        percentage = (count / max(total_entities, 1)) * 100
        avg_confidence = entity_confidence.get(entity_type, {}).get("avg_confidence", 0.8)
        
        entity_metrics[entity_type] = {
            "count": count,
            "percentage": round(percentage, 2),
            "average_confidence": round(avg_confidence, 3),
            "detection_quality": "HIGH" if avg_confidence > 0.85 else "MEDIUM" if avg_confidence > 0.7 else "LOW",
        }
    
    return entity_metrics


def _calculate_anomaly_detection(redactions_per_page: dict[str, int], entities_by_type: Counter[str]) -> dict:
    """Detect statistical anomalies and suspicious patterns."""
    redaction_counts = list(redactions_per_page.values())
    
    if not redaction_counts:
        return {"anomalies_detected": [], "pages_flagged_for_review": []}
    
    avg_redactions = sum(redaction_counts) / len(redaction_counts)
    std_dev = (sum((x - avg_redactions) ** 2 for x in redaction_counts) / len(redaction_counts)) ** 0.5
    threshold = avg_redactions + (2 * std_dev)  # 2 standard deviations
    
    anomalies = []
    flagged_pages = []
    
    for page, count in redactions_per_page.items():
        if count > threshold:
            anomalies.append({
                "page": page,
                "redaction_count": count,
                "deviation_from_average": round(count - avg_redactions, 2),
                "reason": "Unusually high PII concentration",
            })
            flagged_pages.append(page)
    
    # Variance analysis
    variance = std_dev ** 2
    
    return {
        "anomalies_detected": anomalies,
        "pages_flagged_for_review": flagged_pages,
        "redaction_distribution_variance": round(variance, 2),
        "redaction_std_deviation": round(std_dev, 2),
        "average_redactions_per_page": round(avg_redactions, 2),
    }


def _calculate_benchmark_metrics(total_duration: float, total_boxes: int, total_pages: int, original_size: int) -> dict:
    """Calculate performance and benchmark metrics."""
    # Processing efficiency
    pages_per_second = total_pages / max(total_duration, 0.001)
    entities_per_second = total_boxes / max(total_duration, 0.001)
    ms_per_page = (total_duration * 1000) / max(total_pages, 1)
    
    # Size efficiency
    original_size_mb = original_size / (1024 * 1024)
    mb_per_hour = (original_size_mb / max(total_duration, 0.001)) * 3600
    
    return {
        "processing_efficiency": {
            "pages_per_second": round(pages_per_second, 2),
            "redactions_per_second": round(entities_per_second, 2),
            "milliseconds_per_page": round(ms_per_page, 2),
            "megabytes_per_hour": round(mb_per_hour, 2),
        },
        "benchmark_vs_industry": {
            "relative_speed": "FAST" if pages_per_second > 5 else "AVERAGE" if pages_per_second > 1 else "SLOW",
            "cost_efficiency": "HIGH" if total_duration < 5 else "MEDIUM" if total_duration < 30 else "LOW",
        },
    }


def _extract_capability_settings(
    event: dict[str, Any],
) -> tuple[bool, bool, bool, str, str | None, str | None, str | None]:
    include_s3_authorship = _as_bool(event.get("enable_s3_authorship"), DEFAULT_ENABLE_S3_AUTHORSHIP)
    include_s3_document_summary = _as_bool(
        event.get("enable_s3_document_summary"),
        DEFAULT_ENABLE_S3_DOCUMENT_SUMMARY,
    )
    require_s3_capabilities = _as_bool(
        event.get("require_s3_capabilities"),
        DEFAULT_REQUIRE_S3_CAPABILITIES,
    )
    s3_authorship_detector = str(event.get("s3_authorship_detector", DEFAULT_S3_AUTHORSHIP_DETECTOR)).strip().lower()
    s3_authorship_model_name = str(event.get("s3_authorship_model_name", DEFAULT_S3_AUTHORSHIP_MODEL or "")).strip() or None
    s3_summary_model_name = str(event.get("s3_summary_model_name", DEFAULT_S3_SUMMARY_MODEL or "")).strip() or None
    s3_summary_directions = str(event.get("s3_summary_directions", DEFAULT_S3_SUMMARY_DIRECTIONS)).strip() or None
    return (
        include_s3_authorship,
        include_s3_document_summary,
        require_s3_capabilities,
        s3_authorship_detector,
        s3_authorship_model_name,
        s3_summary_model_name,
        s3_summary_directions,
    )


def assemble_redaction_lambda_handler(event: dict, _context: object) -> dict:
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
    (
        include_s3_authorship,
        include_s3_document_summary,
        require_s3_capabilities,
        s3_authorship_detector,
        s3_authorship_model_name,
        s3_summary_model_name,
        s3_summary_directions,
    ) = _extract_capability_settings(event)

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

    try:
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
    finally:
        local_input.unlink(missing_ok=True)

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
    
    # Calculate all quality metrics
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
    
    # Calculate ALL quality metrics
    accuracy_metrics = _calculate_detection_accuracy(findings_detected, total_boxes)
    document_quality = _calculate_document_quality(total_pages_in_doc, pages_with_redactions, original_size, redacted_size)
    consistency_metrics = _calculate_consistency_metrics(phrases_by_page, redactions_per_page)
    compliance_scores = _calculate_compliance_scores()
    risk_assessment = _calculate_risk_assessment(total_boxes, findings_detected, pages_with_redactions, total_pages_in_doc)
    entity_metrics = _calculate_entity_specific_metrics(entities_by_type)
    anomaly_detection = _calculate_anomaly_detection(redactions_per_page, entities_by_type)
    benchmark_metrics = _calculate_benchmark_metrics(execution_duration, total_boxes, total_pages_in_doc, original_size)
    
    # Compliance flags
    needs_review = len(suspicious_patterns) > 0 or effectiveness_score < 90 or len(anomaly_detection.get("pages_flagged_for_review", [])) > 0
    
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
            **benchmark_metrics,
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
            "entity_specific_metrics": entity_metrics,
            "top_redacted_phrases": top_entities,
        },
        "detection_accuracy": accuracy_metrics,
        "document_quality": document_quality,
        "consistency": consistency_metrics,
        "risk_assessment": risk_assessment,
        "anomaly_detection": anomaly_detection,
        "quality_metrics": {
            "redaction_effectiveness_score": round(effectiveness_score, 2),
            "entity_concentration": concentration_analysis,
            "pages_with_high_pii_concentration": [s for s in suspicious_patterns],
        },
        "compliance": {
            "standards_compliance": compliance_scores,
            "manual_review_recommended": needs_review,
            "review_reasons": [
                "High concentration of PII entities" if len(suspicious_patterns) > 0 else None,
                "Effectiveness score below 90%" if effectiveness_score < 90 else None,
                "Anomalies detected in redaction pattern" if len(anomaly_detection.get("pages_flagged_for_review", [])) > 0 else None,
            ],
            "audit_readiness": risk_assessment.get("audit_confidence_percent", 0),
        },
    }
    base_report_key = f"{work_prefix}{job_id}/base-report.json"
    _upload_json(work_bucket, base_report_key, report)

    result = {
        "job_id": job_id,
        "status": "REDACTION_COMPLETED",
        "input_bucket": input_bucket,
        "input_key": input_key,
        "output_bucket": output_bucket,
        "work_bucket": work_bucket,
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "report_prefix": report_prefix,
        "work_prefix": work_prefix,
        "redacted_pdf_key": redacted_key,
        "report_key": report_key,
        "base_report_key": base_report_key,
        "changes_made": total_boxes,
        "effectiveness_score": round(effectiveness_score, 2),
        "enable_s3_authorship": include_s3_authorship,
        "enable_s3_document_summary": include_s3_document_summary,
        "require_s3_capabilities": require_s3_capabilities,
        "s3_authorship_detector": s3_authorship_detector,
        "s3_authorship_model_name": s3_authorship_model_name,
        "s3_summary_model_name": s3_summary_model_name,
        "s3_summary_directions": s3_summary_directions,
    }
    local_output.unlink(missing_ok=True)
    LOGGER.info("Assemble redaction result: %s", json.dumps(result))
    return result


def authorship_artifact_lambda_handler(event: dict, _context: object) -> dict[str, Any]:
    (
        include_s3_authorship,
        _include_s3_document_summary,
        require_s3_capabilities,
        s3_authorship_detector,
        s3_authorship_model_name,
        _s3_summary_model_name,
        _s3_summary_directions,
    ) = _extract_capability_settings(event)

    if not include_s3_authorship:
        return {"capability": "authorship", "result": {"status": "SKIPPED"}}

    input_bucket = event["input_bucket"]
    input_key = event["input_key"]
    output_bucket = event.get("output_bucket") or input_bucket
    input_prefix = event.get("input_prefix", os.getenv("INPUT_PREFIX", "incoming/"))
    report_prefix = event.get("report_prefix", os.getenv("REPORT_PREFIX", "reports/"))
    temp_input = Path(f"/tmp/{event['job_id']}-authorship-input{Path(input_key).suffix or '.pdf'}")

    try:
        s3_client.download_file(input_bucket, input_key, str(temp_input))
        selected_detector = normalize_detector(s3_authorship_detector)
        selected_model = s3_authorship_model_name if selected_detector == DETECTOR_MODEL else None
        authorship_result, extracted_text = analyze_file_authorship(
            temp_input,
            filename=Path(input_key).name,
            detector=selected_detector,
            model_name=selected_model,
        )
        authorship_report_key = _build_output_key(
            input_key,
            input_prefix,
            report_prefix,
            "-authorship-report.json",
        )
        _upload_json(
            output_bucket,
            authorship_report_key,
            {
                "input_key": input_key,
                "detector_used": selected_detector,
                "model_name_used": selected_model,
                "label": authorship_result.label,
                "ai_probability": authorship_result.ai_probability,
                "confidence": authorship_result.confidence,
                "summary": authorship_result.summary,
                "features": authorship_result.features.__dict__,
                "text_preview": extracted_text[:600],
                "text_preview_truncated": len(extracted_text) > 600,
            },
        )
        return {
            "capability": "authorship",
            "result": {
                "status": "COMPLETED",
                "report_key": authorship_report_key,
                "detector_used": selected_detector,
                "model_name_used": selected_model,
                "label": authorship_result.label,
                "ai_probability": authorship_result.ai_probability,
            },
        }
    except Exception as exc:
        if require_s3_capabilities:
            raise
        return {"capability": "authorship", "result": {"status": "FAILED", "error": str(exc)}}
    finally:
        temp_input.unlink(missing_ok=True)


def document_summary_artifact_lambda_handler(event: dict, _context: object) -> dict[str, Any]:
    (
        _include_s3_authorship,
        include_s3_document_summary,
        require_s3_capabilities,
        _s3_authorship_detector,
        _s3_authorship_model_name,
        s3_summary_model_name,
        s3_summary_directions,
    ) = _extract_capability_settings(event)

    if not include_s3_document_summary:
        return {"capability": "document_summary", "result": {"status": "SKIPPED"}}

    input_bucket = event["input_bucket"]
    input_key = event["input_key"]
    output_bucket = event.get("output_bucket") or input_bucket
    input_prefix = event.get("input_prefix", os.getenv("INPUT_PREFIX", "incoming/"))
    report_prefix = event.get("report_prefix", os.getenv("REPORT_PREFIX", "reports/"))
    temp_input = Path(f"/tmp/{event['job_id']}-summary-input{Path(input_key).suffix or '.pdf'}")
    summary_output_path: Path | None = None

    try:
        s3_client.download_file(input_bucket, input_key, str(temp_input))
        summary_report_payload, summary_metrics = generate_document_summary_report(
            temp_input,
            filename=Path(input_key).name,
            model_name=s3_summary_model_name,
            additional_directions=s3_summary_directions,
        )
        summary_output_path, summary_metrics = generate_document_summary_file(
            temp_input,
            filename=Path(input_key).name,
            model_name=s3_summary_model_name,
            additional_directions=s3_summary_directions,
        )
        summary_pdf_key = _build_output_key(
            input_key,
            input_prefix,
            report_prefix,
            "-document-summary.pdf",
        )
        s3_client.upload_file(
            str(summary_output_path),
            output_bucket,
            summary_pdf_key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
        summary_report_key = _build_output_key(
            input_key,
            input_prefix,
            report_prefix,
            "-document-summary-report.json",
        )
        _upload_json(
            output_bucket,
            summary_report_key,
            {
                "input_key": input_key,
                "summary_pdf_key": summary_pdf_key,
                **summary_report_payload,
            },
        )
        return {
            "capability": "document_summary",
            "result": {
                "status": "COMPLETED",
                "summary_pdf_key": summary_pdf_key,
                "report_key": summary_report_key,
                **summary_metrics,
            },
        }
    except Exception as exc:
        if require_s3_capabilities:
            raise
        return {"capability": "document_summary", "result": {"status": "FAILED", "error": str(exc)}}
    finally:
        temp_input.unlink(missing_ok=True)
        if summary_output_path and summary_output_path.exists():
            summary_output_path.unlink(missing_ok=True)


def _collect_capability_outputs(capability_results: Any) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "authorship": {"status": "SKIPPED"},
        "document_summary": {"status": "SKIPPED"},
    }
    if not isinstance(capability_results, list):
        return outputs
    for item in capability_results:
        if not isinstance(item, dict):
            continue
        capability = str(item.get("capability", "")).strip().lower()
        result = item.get("result")
        if capability in outputs and isinstance(result, dict):
            outputs[capability] = result
    return outputs


def finalize_report_lambda_handler(event: dict, _context: object) -> dict[str, Any]:
    LOGGER.info("Finalizing output report for event: %s", json.dumps(event))
    job_id = event["job_id"]
    input_bucket = event["input_bucket"]
    output_bucket = event.get("output_bucket") or input_bucket
    work_bucket = event.get("work_bucket") or output_bucket
    report_key = event["report_key"]
    base_report_key = event.get("base_report_key")
    work_prefix = _normalize_prefix(event.get("work_prefix", os.getenv("WORK_PREFIX", "work/")))
    require_s3_capabilities = _as_bool(
        event.get("require_s3_capabilities"),
        DEFAULT_REQUIRE_S3_CAPABILITIES,
    )

    capability_outputs = _collect_capability_outputs(event.get("capability_results"))
    failed_capabilities = [
        name
        for name, output in capability_outputs.items()
        if isinstance(output, dict) and output.get("status") == "FAILED"
    ]
    if require_s3_capabilities and failed_capabilities:
        raise RuntimeError(f"Required capabilities failed: {', '.join(sorted(failed_capabilities))}")

    if not base_report_key:
        raise RuntimeError("base_report_key is required for final report assembly.")

    report = _load_json(work_bucket, base_report_key)
    report["capabilities"] = capability_outputs

    authorship_output = capability_outputs.get("authorship", {})
    if authorship_output.get("status") == "COMPLETED":
        report["output"]["authorship_report_key"] = authorship_output.get("report_key")
    summary_output = capability_outputs.get("document_summary", {})
    if summary_output.get("status") == "COMPLETED":
        report["output"]["document_summary_pdf_key"] = summary_output.get("summary_pdf_key")
        report["output"]["document_summary_report_key"] = summary_output.get("report_key")

    _upload_json(output_bucket, report_key, report)
    cleanup_deleted = _delete_prefix(work_bucket, f"{work_prefix}{job_id}/")

    result: dict[str, Any] = {
        "job_id": job_id,
        "status": "COMPLETED",
        "output_bucket": output_bucket,
        "redacted_pdf_key": event["redacted_pdf_key"],
        "report_key": report_key,
        "changes_made": int(event.get("changes_made", 0)),
        "effectiveness_score": float(event.get("effectiveness_score", 0.0)),
        "work_objects_deleted": cleanup_deleted,
        "capabilities": capability_outputs,
    }
    if authorship_output.get("status") == "COMPLETED":
        result["authorship_report_key"] = authorship_output.get("report_key")
    if summary_output.get("status") == "COMPLETED":
        result["document_summary_pdf_key"] = summary_output.get("summary_pdf_key")
        result["document_summary_report_key"] = summary_output.get("report_key")
    LOGGER.info("Finalize result: %s", json.dumps(result))
    return result


def lambda_handler(event: dict, context: object) -> dict[str, Any]:
    """Backward-compatible monolithic handler that composes the modular steps."""
    assembled = assemble_redaction_lambda_handler(event, context)
    capability_results = [
        authorship_artifact_lambda_handler(assembled, context),
        document_summary_artifact_lambda_handler(assembled, context),
    ]
    assembled_with_capabilities = dict(assembled)
    assembled_with_capabilities["capability_results"] = capability_results
    return finalize_report_lambda_handler(assembled_with_capabilities, context)
