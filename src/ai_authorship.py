import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from model_inference import call_model_text


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
DETECTOR_HEURISTIC = "heuristic"
DETECTOR_MODEL = "model"
SUPPORTED_DETECTORS = {DETECTOR_HEURISTIC, DETECTOR_MODEL}


def _read_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_AUTHORSHIP_MODEL", "gpt-4.1-mini")
MODEL_INPUT_CHAR_LIMIT = _read_env_int("MODEL_INPUT_CHAR_LIMIT", 12000)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE_RE = re.compile(r"[^.!?]+")
_PUNCTUATION_RE = re.compile(r"[,:;.!?]")
_SPACE_RE = re.compile(r"\s+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "which",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class AuthorshipFeatures:
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    sentence_length_stddev: float
    type_token_ratio: float
    repeated_bigram_ratio: float
    stopword_ratio: float
    punctuation_density: float


@dataclass(frozen=True)
class AuthorshipResult:
    label: str
    ai_probability: float
    confidence: float
    summary: str
    features: AuthorshipFeatures

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "ai_probability": self.ai_probability,
            "confidence": self.confidence,
            "summary": self.summary,
            "features": asdict(self.features),
            "disclaimer": (
                "This is a heuristic estimate. AI-authorship detection is not definitive and "
                "should be combined with manual review."
            ),
        }


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_detector(detector: str | None) -> str:
    normalized = (detector or DETECTOR_HEURISTIC).strip().lower()
    if normalized not in SUPPORTED_DETECTORS:
        supported = ", ".join(sorted(SUPPORTED_DETECTORS))
        raise ValueError(f"Unsupported detector '{normalized}'. Supported detectors: {supported}.")
    return normalized


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def extract_text_from_pdf(path: Path) -> str:
    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return normalize_text("\n".join(pages))


def extract_text_from_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required to analyze Word files.") from exc

    doc = Document(str(path))
    parts: list[str] = []
    for paragraph in doc.paragraphs:
        if paragraph.text and paragraph.text.strip():
            parts.append(paragraph.text)
    return normalize_text("\n".join(parts))


def extract_text_from_file(path: Path, filename: str | None = None) -> str:
    suffix = Path(filename).suffix.lower() if filename else path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def _split_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in _SENTENCE_RE.findall(text) if segment.strip()]
    return sentences or [text]


def _tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def compute_features(text: str) -> AuthorshipFeatures:
    words = _tokenize_words(text)
    if not words:
        return AuthorshipFeatures(
            word_count=0,
            sentence_count=0,
            avg_sentence_length=0.0,
            sentence_length_stddev=0.0,
            type_token_ratio=0.0,
            repeated_bigram_ratio=0.0,
            stopword_ratio=0.0,
            punctuation_density=0.0,
        )

    sentences = _split_sentences(text)
    sentence_lengths: list[int] = []
    for sentence in sentences:
        sentence_word_count = len(_tokenize_words(sentence))
        if sentence_word_count:
            sentence_lengths.append(sentence_word_count)
    if not sentence_lengths:
        sentence_lengths = [len(words)]

    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((length - avg_sentence_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
    sentence_stddev = math.sqrt(variance)

    unique_words = len(set(words))
    type_token_ratio = unique_words / len(words)

    bigrams = list(zip(words, words[1:]))
    repeated_bigram_count = sum(count - 1 for count in Counter(bigrams).values() if count > 1)
    repeated_bigram_ratio = repeated_bigram_count / max(1, len(bigrams))

    stopword_ratio = sum(1 for word in words if word in _STOPWORDS) / len(words)
    punctuation_density = len(_PUNCTUATION_RE.findall(text)) / max(1, len(text))

    return AuthorshipFeatures(
        word_count=len(words),
        sentence_count=len(sentence_lengths),
        avg_sentence_length=avg_sentence_length,
        sentence_length_stddev=sentence_stddev,
        type_token_ratio=type_token_ratio,
        repeated_bigram_ratio=repeated_bigram_ratio,
        stopword_ratio=stopword_ratio,
        punctuation_density=punctuation_density,
    )


def _estimate_probability(features: AuthorshipFeatures) -> float:
    low_burstiness = _clamp_01((14.0 - features.sentence_length_stddev) / 14.0)
    high_repetition = _clamp_01(features.repeated_bigram_ratio / 0.07)
    low_lexical_diversity = _clamp_01((0.62 - features.type_token_ratio) / 0.25)
    sentence_length_uniformity = _clamp_01(1.0 - abs(features.avg_sentence_length - 19.0) / 16.0)
    stopword_pattern = _clamp_01(1.0 - abs(features.stopword_ratio - 0.45) / 0.25)
    punctuation_pattern = _clamp_01(1.0 - abs(features.punctuation_density - 0.055) / 0.04)

    probability = (
        0.24 * low_burstiness
        + 0.22 * high_repetition
        + 0.20 * low_lexical_diversity
        + 0.14 * sentence_length_uniformity
        + 0.10 * stopword_pattern
        + 0.10 * punctuation_pattern
    )
    return round(_clamp_01(probability), 4)


def _estimate_confidence(features: AuthorshipFeatures, probability: float) -> float:
    length_factor = _clamp_01((features.word_count - 120) / 500)
    separation_factor = abs(probability - 0.5) * 2
    confidence = 0.35 + 0.4 * length_factor + 0.35 * separation_factor
    if features.word_count < 80:
        confidence *= 0.55
    if features.word_count < 40:
        confidence *= 0.5
    return round(_clamp_01(confidence), 4)


def _pick_label(probability: float, confidence: float) -> str:
    if confidence < 0.45:
        return "inconclusive"
    if probability >= 0.65:
        return "likely_ai"
    if probability <= 0.35:
        return "likely_human"
    return "inconclusive"


def analyze_text_authorship(text: str) -> AuthorshipResult:
    normalized = normalize_text(text)
    features = compute_features(normalized)
    probability = _estimate_probability(features)
    confidence = _estimate_confidence(features, probability)
    label = _pick_label(probability, confidence)

    summary = (
        f"Estimated AI authorship probability: {round(probability * 100, 1)}% "
        f"(confidence {round(confidence * 100, 1)}%)."
    )
    if features.word_count < 80:
        summary += " Document is short, so confidence is reduced."

    return AuthorshipResult(
        label=label,
        ai_probability=probability,
        confidence=confidence,
        summary=summary,
        features=features,
    )


def _extract_json_payload(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise RuntimeError("Model detector returned an empty response.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first < 0 or last < 0 or last <= first:
            raise RuntimeError("Model detector returned invalid JSON output.")
        try:
            payload = json.loads(text[first : last + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Model detector returned unparsable JSON output.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Model detector returned a non-object JSON payload.")
    return payload


def _call_openai_model_detector(text: str, model_name: str) -> dict[str, Any]:
    system_prompt = (
        "You are an AI-authorship analyst. Return JSON only with keys: "
        "ai_probability (0..1 float), confidence (0..1 float), "
        "label (likely_ai|likely_human|inconclusive), and summary."
    )
    user_prompt = (
        "Assess whether the following text was likely written by AI.\n\n"
        "Output JSON only.\n\n"
        f"TEXT:\n{text}"
    )

    # Provider routing is handled by `call_model_text`:
    # - OpenAI models -> OPENAI_API_KEY required
    # - `bedrock:*` models -> AWS credentials + Bedrock model access required
    # GovCloud/NIST note (SC-7/SA-9): disable external-provider options if your boundary requires
    # all inference to stay inside AWS-authorized services.
    raw = call_model_text(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        require_json=True,
        temperature=0.0,
        max_tokens=500,
        error_context="Model detector request",
    )
    return _extract_json_payload(raw)


def analyze_text_authorship_model(text: str, model_name: str | None = None) -> AuthorshipResult:
    normalized = normalize_text(text)
    features = compute_features(normalized)
    effective_model = (model_name or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL

    truncated = normalized[:MODEL_INPUT_CHAR_LIMIT]
    payload = _call_openai_model_detector(truncated, model_name=effective_model)

    probability = _clamp_01(_to_float(payload.get("ai_probability"), 0.5))
    confidence = _clamp_01(_to_float(payload.get("confidence"), 0.5))
    raw_label = str(payload.get("label", "")).strip().lower()
    label = raw_label if raw_label in {"likely_ai", "likely_human", "inconclusive"} else _pick_label(probability, confidence)

    summary = str(payload.get("summary", "")).strip()
    if not summary:
        summary = (
            f"Estimated AI authorship probability: {round(probability * 100, 1)}% "
            f"(confidence {round(confidence * 100, 1)}%) from model-backed analysis."
        )
    if len(normalized) > MODEL_INPUT_CHAR_LIMIT:
        summary += f" Only the first {MODEL_INPUT_CHAR_LIMIT} characters were analyzed by the model."
    if features.word_count < 80:
        summary += " Document is short, so confidence is reduced."

    return AuthorshipResult(
        label=label,
        ai_probability=round(probability, 4),
        confidence=round(confidence, 4),
        summary=summary,
        features=features,
    )


def analyze_file_authorship(
    path: Path,
    filename: str | None = None,
    detector: str = DETECTOR_HEURISTIC,
    model_name: str | None = None,
) -> tuple[AuthorshipResult, str]:
    selected_detector = normalize_detector(detector)
    text = extract_text_from_file(path, filename=filename)
    if selected_detector == DETECTOR_MODEL:
        result = analyze_text_authorship_model(text, model_name=model_name)
    else:
        result = analyze_text_authorship(text)
    return result, text
