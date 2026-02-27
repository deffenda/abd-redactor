import json
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from ai_authorship import DEFAULT_OPENAI_MODEL, SUPPORTED_EXTENSIONS, extract_text_from_file, normalize_text
from model_inference import call_model_text
from prepare_chunks import chunk_text


SUPPORTED_DOCUMENT_SUMMARY_EXTENSIONS = set(SUPPORTED_EXTENSIONS)


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


DEFAULT_DOCUMENT_SUMMARY_MODEL = os.getenv(
    "OPENAI_DOCUMENT_SUMMARY_MODEL",
    os.getenv("OPENAI_AUTHORSHIP_MODEL", DEFAULT_OPENAI_MODEL),
)
DOCUMENT_SUMMARY_INPUT_CHAR_LIMIT = _read_env_int(
    "DOCUMENT_SUMMARY_INPUT_CHAR_LIMIT",
    16000,
)
DOCUMENT_SUMMARY_CHUNK_CHAR_LIMIT = _read_env_int(
    "DOCUMENT_SUMMARY_CHUNK_CHAR_LIMIT",
    _read_env_int("CHUNK_CHAR_LIMIT", 4500),
)
DOCUMENT_SUMMARY_MAX_CHUNKS = _read_env_int("DOCUMENT_SUMMARY_MAX_CHUNKS", 12)


def build_document_summary_filename(filename: str) -> str:
    source = Path(filename or "document")
    return f"{source.stem}-document-summary.pdf"


def _extract_json_payload(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("Document summary model returned an empty response.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first < 0 or last < 0 or last <= first:
            raise RuntimeError("Document summary model returned invalid JSON output.")
        try:
            payload = json.loads(text[first : last + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Document summary model returned unparsable JSON output.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Document summary model returned a non-object JSON payload.")
    return payload


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _string_value(value: Any, default: str = "Not provided") -> str:
    text = str(value or "").strip()
    return text or default


def _call_document_summary_model(text: str, model_name: str, additional_directions: str | None = None) -> dict[str, Any]:
    system_prompt = (
        "You are a document summarization assistant. Use only evidence explicitly present in the source text. "
        "Do not invent facts. Return JSON only with keys: "
        "summary, key_points, action_items, relevant_details, unanswered_questions, disclaimer. "
        "key_points/action_items/relevant_details/unanswered_questions must each be arrays of strings."
    )

    user_prompt_parts = [
        "Create a clear document summary from the text below.",
        "Return JSON only.",
    ]
    directions = (additional_directions or "").strip()
    if directions:
        user_prompt_parts.append("Additional user directions:")
        user_prompt_parts.append(directions)
    user_prompt_parts.append("")
    user_prompt_parts.append("DOCUMENT_TEXT:")
    user_prompt_parts.append(text)
    user_prompt = "\n".join(user_prompt_parts)

    # Provider routing is handled by `call_model_text`:
    # - OpenAI models require OPENAI_API_KEY
    # - `bedrock:*` models require AWS credentials + Bedrock access in-region
    # GovCloud/NIST note (SC-7/SA-9): external-provider usage may be disallowed in some enclaves.
    raw = call_model_text(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        require_json=True,
        temperature=0.0,
        max_tokens=1200,
        error_context="Document summary model request",
    )
    return _extract_json_payload(raw)


def _format_section(lines: list[str], title: str, values: list[str], empty_text: str = "Not identified.") -> None:
    lines.append(title)
    if values:
        for value in values:
            lines.append(f"- {value}")
    else:
        lines.append(f"- {empty_text}")
    lines.append("")


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(normalized)
    return values


def _merge_chunk_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise RuntimeError("Document summary model produced no chunk outputs.")

    summary_segments: list[str] = []
    key_points: list[str] = []
    action_items: list[str] = []
    relevant_details: list[str] = []
    unanswered_questions: list[str] = []
    disclaimer = ""

    for payload in payloads:
        summary = _string_value(payload.get("summary"), "")
        if summary:
            summary_segments.append(summary)
        key_points.extend(_string_list(payload.get("key_points")))
        action_items.extend(_string_list(payload.get("action_items")))
        relevant_details.extend(_string_list(payload.get("relevant_details")))
        unanswered_questions.extend(_string_list(payload.get("unanswered_questions")))
        if not disclaimer:
            disclaimer = _string_value(payload.get("disclaimer"), "")

    combined_summary = "\n\n".join(summary_segments).strip()
    if not combined_summary:
        combined_summary = "No summary was generated."

    return {
        "summary": combined_summary,
        "key_points": _dedupe_preserve(key_points),
        "action_items": _dedupe_preserve(action_items),
        "relevant_details": _dedupe_preserve(relevant_details),
        "unanswered_questions": _dedupe_preserve(unanswered_questions),
        "disclaimer": disclaimer or "AI-generated summary for review support only. Verify against the original document.",
    }


def _render_document_summary_text(
    payload: dict[str, Any],
    *,
    source_filename: str,
    model_name: str,
    analyzed_chars: int,
    original_chars: int,
    additional_directions: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("DOCUMENT SUMMARY")
    lines.append("")
    lines.append(f"- Source file: {source_filename}")
    lines.append(f"- Model: {model_name}")
    lines.append(f"- Characters analyzed: {analyzed_chars}")
    if analyzed_chars < original_chars:
        lines.append(f"- Note: truncated from {original_chars} total characters before model analysis.")
    lines.append("")

    if (additional_directions or "").strip():
        lines.append("USER DIRECTIONS")
        lines.append((additional_directions or "").strip())
        lines.append("")

    lines.append("SUMMARY")
    lines.append(_string_value(payload.get("summary"), "No summary was generated."))
    lines.append("")

    _format_section(lines, "KEY POINTS", _string_list(payload.get("key_points")))
    _format_section(lines, "ACTION ITEMS", _string_list(payload.get("action_items")))
    _format_section(lines, "RELEVANT DETAILS", _string_list(payload.get("relevant_details")))
    _format_section(lines, "UNANSWERED QUESTIONS", _string_list(payload.get("unanswered_questions")))

    lines.append("DISCLAIMER")
    lines.append(
        _string_value(
            payload.get("disclaimer"),
            "AI-generated summary for review support only. Verify against the original document.",
        )
    )
    lines.append("")

    return "\n".join(lines)


def _write_summary_text_to_pdf(summary_text: str, output_path: Path) -> None:
    wrapped_lines: list[str] = []
    for line in summary_text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=98, break_long_words=True, break_on_hyphens=False))

    if not wrapped_lines:
        wrapped_lines = ["Document summary was empty."]

    page_width = 612
    page_height = 792
    margin_x = 48
    margin_y = 48
    font_size = 10
    line_height = 13
    max_lines_per_page = max(1, int((page_height - (2 * margin_y)) / line_height))

    document = fitz.open()
    try:
        line_index = 0
        while line_index < len(wrapped_lines):
            page = document.new_page(width=page_width, height=page_height)
            y = margin_y
            for _ in range(max_lines_per_page):
                if line_index >= len(wrapped_lines):
                    break
                page.insert_text((margin_x, y), wrapped_lines[line_index], fontsize=font_size, fontname="helv")
                y += line_height
                line_index += 1
        document.save(str(output_path), garbage=4, deflate=True, clean=True)
    finally:
        document.close()


def generate_document_summary_file(
    input_path: Path,
    *,
    filename: str,
    model_name: str | None = None,
    additional_directions: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUMMARY_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_SUMMARY_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}.")

    extracted_text = extract_text_from_file(input_path, filename=filename)
    normalized = normalize_text(extracted_text)
    if not normalized:
        raise ValueError("No extractable text found in uploaded file.")

    effective_model = (model_name or DEFAULT_DOCUMENT_SUMMARY_MODEL).strip() or DEFAULT_DOCUMENT_SUMMARY_MODEL
    directions = (additional_directions or "").strip()
    all_chunks = chunk_text(normalized, DOCUMENT_SUMMARY_CHUNK_CHAR_LIMIT)
    if not all_chunks:
        raise ValueError("No extractable text chunks found in uploaded file.")

    if DOCUMENT_SUMMARY_MAX_CHUNKS > 0:
        selected_chunks = all_chunks[:DOCUMENT_SUMMARY_MAX_CHUNKS]
    else:
        selected_chunks = all_chunks

    payloads: list[dict[str, Any]] = []
    characters_analyzed = 0
    for chunk in selected_chunks:
        model_input = chunk[:DOCUMENT_SUMMARY_INPUT_CHAR_LIMIT].strip()
        if not model_input:
            continue
        payloads.append(
            _call_document_summary_model(
                model_input,
                effective_model,
                additional_directions=directions or None,
            )
        )
        characters_analyzed += len(model_input)

    payload = _merge_chunk_payloads(payloads)

    summary_text = _render_document_summary_text(
        payload,
        source_filename=filename,
        model_name=effective_model,
        analyzed_chars=characters_analyzed,
        original_chars=len(normalized),
        additional_directions=directions or None,
    )

    output_fd, output_name = tempfile.mkstemp(suffix=".pdf")
    os.close(output_fd)
    output_path = Path(output_name)
    _write_summary_text_to_pdf(summary_text, output_path)

    metrics = {
        "model_name_used": effective_model,
        "input_characters": len(normalized),
        "characters_analyzed": characters_analyzed,
        "total_chunks": len(all_chunks),
        "chunks_analyzed": len(payloads),
        "chunk_char_limit": DOCUMENT_SUMMARY_CHUNK_CHAR_LIMIT,
        "max_chunks": DOCUMENT_SUMMARY_MAX_CHUNKS,
        "key_point_count": len(payload.get("key_points", [])) if isinstance(payload.get("key_points", []), list) else 0,
        "additional_directions_provided": bool(directions),
        "additional_directions_length": len(directions),
    }
    return output_path, metrics
