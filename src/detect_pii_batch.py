import json
import logging
import os
from .detect_pii_settings import MIN_ENTITY_SCORE as DEFAULT_MIN_ENTITY_SCORE, COMPREHEND_LANGUAGE as DEFAULT_LANGUAGE_CODE, PII_DETECTION_API
from collections import Counter

import boto3


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

s3_client = boto3.client("s3")
comprehend_client = boto3.client("comprehend")


*** Removed: _env_float and local defaults, now using detect_pii_settings.py


def _load_json(bucket: str, key: str) -> dict:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def _write_json(bucket: str, key: str, payload: dict) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


def lambda_handler(event: dict, _context: object) -> dict:
    LOGGER.info("Detecting PII for batch event: %s", json.dumps(event))

    job_id = event["job_id"]
    work_bucket = event["work_bucket"]
    work_prefix = event["work_prefix"]
    batch_manifest_key = event["batch_manifest_key"]
    min_entity_score = float(event.get("min_entity_score", DEFAULT_MIN_ENTITY_SCORE))
    language_code = event.get("language_code", DEFAULT_LANGUAGE_CODE)

    batch_payload = _load_json(work_bucket, batch_manifest_key)
    chunks = batch_payload.get("chunks", [])

    findings: list[dict] = []
    entity_counts: Counter[str] = Counter()
    seen: set[tuple[int, str, str]] = set()

    if PII_DETECTION_API == "start_job":
        # Placeholder for StartPiiEntitiesDetectionJob logic
        raise NotImplementedError("StartPiiEntitiesDetectionJob is not yet implemented in detect_pii_batch.")
    else:
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue

            response = comprehend_client.detect_pii_entities(
                Text=text,
                LanguageCode=language_code,
            )

            for entity in response.get("Entities", []):
                score = float(entity.get("Score", 0.0))
                if score < min_entity_score:
                    continue

                begin = int(entity.get("BeginOffset", 0))
                end = int(entity.get("EndOffset", 0))
                if begin >= end or end > len(text):
                    continue

                phrase = text[begin:end].strip()
                if len(phrase) < 2:
                    continue

                entity_type = entity.get("Type", "UNKNOWN")
                dedupe_key = (int(chunk.get("page_number", 0)), phrase, entity_type)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                findings.append(
                    {
                        "page_number": int(chunk.get("page_number", 0)),
                        "phrase": phrase,
                        "entity_type": entity_type,
                        "score": round(score, 4),
                    }
                )
                entity_counts[entity_type] += 1

    batch_name = batch_manifest_key.rsplit("/", 1)[-1]
    finding_key = f"{work_prefix}{job_id}/findings/{batch_name}"
    finding_payload = {
        "job_id": job_id,
        "batch_manifest_key": batch_manifest_key,
        "findings": findings,
    }
    _write_json(work_bucket, finding_key, finding_payload)

    result = {
        "job_id": job_id,
        "finding_key": finding_key,
        "finding_count": len(findings),
        "entity_counts": dict(entity_counts),
    }
    LOGGER.info("Batch detection result: %s", json.dumps(result))
    return result
