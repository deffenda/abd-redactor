import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import boto3
import fitz  # PyMuPDF

from detect_pii_settings import COMPREHEND_LANGUAGE, MIN_ENTITY_SCORE
from prepare_chunks import chunk_text


SUPPORTED_REDACTION_EXTENSIONS = {".pdf"}
REDACTION_ENGINE_COMPREHEND = "comprehend"
REDACTION_ENGINE_MODEL = "model"
SUPPORTED_REDACTION_ENGINES = {REDACTION_ENGINE_COMPREHEND, REDACTION_ENGINE_MODEL}

DEFAULT_CHUNK_CHAR_LIMIT = int(os.getenv("CHUNK_CHAR_LIMIT", "4500"))
DEFAULT_REDACTION_MODEL = os.getenv("OPENAI_REDACTION_MODEL", os.getenv("OPENAI_AUTHORSHIP_MODEL", "gpt-4.1"))
MODEL_REDACTION_INPUT_CHAR_LIMIT = int(os.getenv("MODEL_REDACTION_INPUT_CHAR_LIMIT", "9000"))

comprehend_client = boto3.client("comprehend")


def build_redacted_filename(filename: str) -> str:
    source = Path(filename or "document.pdf")
    suffix = source.suffix or ".pdf"
    return f"{source.stem}-redacted{suffix}"


def normalize_redaction_engine(redaction_engine: str | None) -> str:
    normalized = (redaction_engine or REDACTION_ENGINE_COMPREHEND).strip().lower()
    if normalized not in SUPPORTED_REDACTION_ENGINES:
        supported = ", ".join(sorted(SUPPORTED_REDACTION_ENGINES))
        raise ValueError(f"Unsupported redaction engine '{normalized}'. Supported engines: {supported}.")
    return normalized


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_json_payload(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise RuntimeError("Model redaction detector returned an empty response.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first < 0 or last < 0 or last <= first:
            raise RuntimeError("Model redaction detector returned invalid JSON output.")
        try:
            payload = json.loads(text[first : last + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Model redaction detector returned unparsable JSON output.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Model redaction detector returned a non-object JSON payload.")
    return payload


def _call_model_redaction_detector(client: object, text: str, model_name: str) -> list[dict]:
    system_prompt = (
        "You are a PII extraction assistant for redaction. "
        "Return JSON only with key 'findings' that is a list of objects. "
        "Each finding object must include: phrase (exact text span copied from input), "
        "entity_type (EMAIL|PHONE|SSN|ADDRESS|NAME|CREDIT_CARD|DOB|IDENTIFICATION|UNKNOWN), "
        "score (0..1). "
        "Only include phrases that appear verbatim in the provided text."
    )
    user_prompt = (
        "Extract PII findings from this text for redaction.\n\n"
        "Return JSON only.\n\n"
        f"TEXT:\n{text}"
    )

    payload: dict[str, Any]

    try:
        response = client.responses.create(
            model=model_name,
            instructions=system_prompt,
            input=user_prompt,
            text={"format": {"type": "json_object"}},
        )
        raw = getattr(response, "output_text", "") or ""
        payload = _extract_json_payload(raw)
    except Exception:
        request_kwargs = {
            "model": model_name,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not model_name.startswith("gpt-5"):
            request_kwargs["temperature"] = 0

        try:
            response = client.chat.completions.create(**request_kwargs)
            raw = response.choices[0].message.content or ""
            payload = _extract_json_payload(raw)
        except Exception as chat_exc:
            raise RuntimeError(f"Model redaction request failed: {chat_exc}") from chat_exc

    findings_raw = payload.get("findings", [])
    if not isinstance(findings_raw, list):
        return []

    findings: list[dict] = []
    for item in findings_raw:
        if not isinstance(item, dict):
            continue
        phrase = str(item.get("phrase", "")).strip()
        if len(phrase) < 2:
            continue
        findings.append(
            {
                "phrase": phrase,
                "entity_type": str(item.get("entity_type", "UNKNOWN")).strip().upper() or "UNKNOWN",
                "score": _to_float(item.get("score"), 0.8),
            }
        )
    return findings


def _extract_findings_from_pdf_comprehend(
    input_path: Path,
    chunk_char_limit: int,
    min_entity_score: float,
    language_code: str,
) -> list[dict]:
    findings: list[dict] = []
    seen: set[tuple[int, str, str]] = set()

    with fitz.open(input_path) as document:
        for page in document:
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            for chunk in chunk_text(page_text, chunk_char_limit):
                if not chunk.strip():
                    continue

                response = comprehend_client.detect_pii_entities(
                    Text=chunk,
                    LanguageCode=language_code,
                )

                for entity in response.get("Entities", []):
                    score = float(entity.get("Score", 0.0))
                    if score < min_entity_score:
                        continue

                    begin = int(entity.get("BeginOffset", 0))
                    end = int(entity.get("EndOffset", 0))
                    if begin >= end or end > len(chunk):
                        continue

                    phrase = chunk[begin:end].strip()
                    if len(phrase) < 2:
                        continue

                    entity_type = str(entity.get("Type", "UNKNOWN"))
                    dedupe_key = (page.number + 1, phrase, entity_type)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)

                    findings.append(
                        {
                            "page_number": page.number + 1,
                            "phrase": phrase,
                            "entity_type": entity_type,
                            "score": round(score, 4),
                        }
                    )

    return findings


def _extract_findings_from_pdf_model(
    input_path: Path,
    chunk_char_limit: int,
    min_entity_score: float,
    model_name: str | None,
) -> tuple[list[dict], str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when redaction_engine='model'.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required when redaction_engine='model'.") from exc

    effective_model = (model_name or DEFAULT_REDACTION_MODEL).strip() or DEFAULT_REDACTION_MODEL
    client = OpenAI(api_key=api_key)

    findings: list[dict] = []
    seen: set[tuple[int, str, str]] = set()

    with fitz.open(input_path) as document:
        for page in document:
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            for chunk in chunk_text(page_text, chunk_char_limit):
                if not chunk.strip():
                    continue

                model_chunk = chunk[:MODEL_REDACTION_INPUT_CHAR_LIMIT]
                model_findings = _call_model_redaction_detector(client, model_chunk, effective_model)
                chunk_lower = model_chunk.lower()

                for finding in model_findings:
                    phrase = str(finding.get("phrase", "")).strip()
                    if len(phrase) < 2:
                        continue

                    if phrase not in model_chunk:
                        phrase_lower = phrase.lower()
                        index = chunk_lower.find(phrase_lower)
                        if index < 0:
                            continue
                        phrase = model_chunk[index : index + len(phrase)]

                    score = _to_float(finding.get("score"), 0.8)
                    if score < min_entity_score:
                        continue

                    entity_type = str(finding.get("entity_type", "UNKNOWN")).strip().upper() or "UNKNOWN"
                    dedupe_key = (page.number + 1, phrase, entity_type)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)

                    findings.append(
                        {
                            "page_number": page.number + 1,
                            "phrase": phrase,
                            "entity_type": entity_type,
                            "score": round(score, 4),
                        }
                    )

    return findings, effective_model


def _apply_findings_to_pdf(input_path: Path, output_path: Path, findings: list[dict]) -> dict:
    phrases_by_page: dict[int, set[str]] = defaultdict(set)
    entities_by_type: Counter[str] = Counter()

    for finding in findings:
        page_number = int(finding.get("page_number", 0))
        phrase = str(finding.get("phrase", "")).strip()
        if page_number <= 0 or len(phrase) < 2:
            continue
        phrases_by_page[page_number].add(phrase)
        entities_by_type[str(finding.get("entity_type", "UNKNOWN"))] += 1

    total_boxes = 0
    pages_with_redactions = 0
    redactions_per_page: dict[str, int] = {}

    with fitz.open(input_path) as document:
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

        document.save(str(output_path), garbage=4, deflate=True, clean=True)

    return {
        "total_findings_detected": len(findings),
        "unique_phrases_detected": sum(len(values) for values in phrases_by_page.values()),
        "total_redactions": total_boxes,
        "pages_with_redactions": pages_with_redactions,
        "redactions_per_page": redactions_per_page,
        "entities_by_type": dict(entities_by_type),
    }


def redact_uploaded_file(
    input_path: Path,
    filename: str,
    *,
    redaction_engine: str = REDACTION_ENGINE_COMPREHEND,
    model_name: str | None = None,
    chunk_char_limit: int = DEFAULT_CHUNK_CHAR_LIMIT,
    min_entity_score: float = MIN_ENTITY_SCORE,
    language_code: str = COMPREHEND_LANGUAGE,
) -> tuple[Path, dict]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_REDACTION_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_REDACTION_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}.")

    selected_engine = normalize_redaction_engine(redaction_engine)

    output_fd, output_name = tempfile.mkstemp(suffix=suffix)
    os.close(output_fd)
    Path(output_name).unlink(missing_ok=True)
    output_path = Path(output_name)

    selected_model_name: str | None = None
    if selected_engine == REDACTION_ENGINE_MODEL:
        findings, selected_model_name = _extract_findings_from_pdf_model(
            input_path=input_path,
            chunk_char_limit=chunk_char_limit,
            min_entity_score=min_entity_score,
            model_name=model_name,
        )
    else:
        findings = _extract_findings_from_pdf_comprehend(
            input_path=input_path,
            chunk_char_limit=chunk_char_limit,
            min_entity_score=min_entity_score,
            language_code=language_code,
        )

    metrics = _apply_findings_to_pdf(input_path=input_path, output_path=output_path, findings=findings)
    metrics["redaction_engine"] = selected_engine
    metrics["model_name_used"] = selected_model_name
    return output_path, metrics
