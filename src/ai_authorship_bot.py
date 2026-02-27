import json
import os
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from ai_authorship import (
    DETECTOR_HEURISTIC,
    DETECTOR_MODEL,
    DEFAULT_OPENAI_MODEL,
    SUPPORTED_DETECTORS,
    SUPPORTED_EXTENSIONS,
    analyze_file_authorship,
    normalize_detector,
)
from document_summary import (
    DEFAULT_DOCUMENT_SUMMARY_MODEL,
    SUPPORTED_DOCUMENT_SUMMARY_EXTENSIONS,
    build_document_summary_filename,
    generate_document_summary_file,
)
from model_inference import MODEL_DROPDOWN_OPTIONS, SUPPORTED_MODEL_NAMES
from upload_redaction import (
    REDACTION_ENGINE_COMPREHEND,
    REDACTION_ENGINE_MODEL,
    SUPPORTED_REDACTION_ENGINES,
    SUPPORTED_REDACTION_EXTENSIONS,
    build_redacted_filename,
    normalize_redaction_engine,
    redact_uploaded_file,
)


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
TEXT_PREVIEW_LENGTH = int(os.getenv("TEXT_PREVIEW_LENGTH", "280"))
# MODEL_DROPDOWN_OPTIONS comes from `model_inference` and intentionally includes:
# - OpenAI model names
# - `bedrock:*` identifiers for AWS Bedrock models
# Keep the API validation list and UI dropdown in sync by editing that single source.
# GovCloud/NIST deployment tip: if your boundary disallows external providers, expose only
# `bedrock:*` options in `model_inference.py` and remove OpenAI options from that list.

app = FastAPI(
    title="AI Authorship + Redaction + Document Summary Bot",
    version="1.0.0",
    description=(
        "Upload documents for AI-authorship detection, S3-parity PDF redaction download, "
        "and model-generated document summaries."
    ),
)


def _validate_file(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")
    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Supported types: {supported}.",
        )
    return extension


def _validate_detector(detector: str | None) -> str:
    try:
        return normalize_detector(detector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_redaction_file(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")
    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_REDACTION_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_REDACTION_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Supported redaction types: {supported}.",
        )
    return extension


def _validate_redaction_engine(redaction_engine: str | None) -> str:
    try:
        return normalize_redaction_engine(redaction_engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_document_summary_file(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")
    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_SUMMARY_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_SUMMARY_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Supported document summary types: {supported}.",
        )
    return extension


def _validate_model_name(model_name: str | None, *, default_model: str = DEFAULT_OPENAI_MODEL) -> str:
    selected = (model_name or default_model).strip() or default_model
    if model_name and selected not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(model for model, _ in MODEL_DROPDOWN_OPTIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{selected}'. Supported models: {supported}.",
        )
    return selected


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "available_detectors": sorted(SUPPORTED_DETECTORS),
        "available_model_names": [model for model, _ in MODEL_DROPDOWN_OPTIONS],
        "available_redaction_engines": sorted(SUPPORTED_REDACTION_ENGINES),
        "default_redaction_engine": REDACTION_ENGINE_COMPREHEND,
        "default_model_name": DEFAULT_OPENAI_MODEL,
        "redaction_file_types": sorted(SUPPORTED_REDACTION_EXTENSIONS),
        "document_summary_file_types": sorted(SUPPORTED_DOCUMENT_SUMMARY_EXTENSIONS),
        "default_document_summary_model": DEFAULT_DOCUMENT_SUMMARY_MODEL,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    model_options = []
    for model_name, label in MODEL_DROPDOWN_OPTIONS:
        selected = " selected" if model_name == DEFAULT_OPENAI_MODEL else ""
        model_options.append(f'<option value="{model_name}"{selected}>{label}</option>')
    model_options_markup = "\n".join(model_options)

    document_summary_model_options = []
    for model_name, label in MODEL_DROPDOWN_OPTIONS:
        selected = " selected" if model_name == DEFAULT_DOCUMENT_SUMMARY_MODEL else ""
        document_summary_model_options.append(f'<option value="{model_name}"{selected}>{label}</option>')
    document_summary_model_options_markup = "\n".join(document_summary_model_options)

    return (
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Authorship + Redaction + Document Summary Bot</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 2rem; max-width: 920px; }
    h1 { margin-bottom: 0.4rem; }
    .layout { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
    .card { border: 1px solid #d6d6d6; border-radius: 8px; padding: 1rem; }
    h2 { margin-top: 0; }
    button { margin-top: 0.8rem; padding: 0.5rem 0.8rem; }
    select { margin-top: 0.4rem; margin-bottom: 0.4rem; }
    pre { background: #f5f5f5; border-radius: 6px; padding: 0.8rem; white-space: pre-wrap; }
    .download-link { display: inline-block; margin-top: 0.8rem; }
  </style>
</head>
<body>
  <h1>AI Authorship + Redaction + Document Summary Bot</h1>
  <p>Upload files for AI authorship analysis, PDF redaction download, and document summary generation.</p>
  <div class="layout">
    <div class="card">
      <h2>AI Detection</h2>
      <p>Upload a PDF or DOCX, choose a detector and model as needed, then click Analyze to view the AI authorship score.</p>
      <input id="file" type="file" accept=".pdf,.docx" />
      <br />
      <label for="detector">Detector:</label>
      <select id="detector">
        <option value="heuristic">Heuristic (current)</option>
        <option value="model">Model-backed</option>
      </select>
      <br />
      <label for="modelName">Model:</label>
      <select id="modelName">
        __MODEL_OPTIONS__
      </select>
      <br />
      <button id="submit">Analyze</button>
      <p id="status"></p>
      <pre id="result"></pre>
    </div>
    <div class="card">
      <h2>Redaction</h2>
      <p>Upload a PDF, select Comprehend or model redaction, then click Redact &amp; Download to get a redacted copy.</p>
      <input id="redactFile" type="file" accept=".pdf" />
      <br />
      <label for="redactionEngine">Engine:</label>
      <select id="redactionEngine">
        <option value="comprehend">Comprehend + chunking + dedupe</option>
        <option value="model">Model-based + chunking + dedupe</option>
      </select>
      <br />
      <label for="redactionModelName">Model:</label>
      <select id="redactionModelName">
        __MODEL_OPTIONS__
      </select>
      <br />
      <button id="redactSubmit">Redact &amp; Download</button>
      <p id="redactStatus"></p>
      <a id="downloadLink" class="download-link" hidden></a>
      <pre id="redactResult"></pre>
    </div>
    <div class="card">
      <h2>Document Summary</h2>
      <p>Upload a PDF or DOCX, optionally add extra directions, then click Generate Summary &amp; Download to receive a summary PDF.</p>
      <input id="summaryFile" type="file" accept=".pdf,.docx" />
      <br />
      <label for="summaryModelName">Model:</label>
      <select id="summaryModelName">
        __DOCUMENT_SUMMARY_MODEL_OPTIONS__
      </select>
      <br />
      <label for="summaryDirections">Additional Directions (optional):</label>
      <br />
      <textarea id="summaryDirections" rows="4" style="width: 100%;" placeholder="Example: Emphasize risks, deadlines, and action items for operations leadership."></textarea>
      <br />
      <button id="summarySubmit">Generate Summary &amp; Download</button>
      <p id="summaryStatus"></p>
      <a id="summaryDownloadLink" class="download-link" hidden></a>
      <pre id="summaryResult"></pre>
    </div>
  </div>
  <script>
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const detectorEl = document.getElementById("detector");
    const modelNameEl = document.getElementById("modelName");
    const redactionEngineEl = document.getElementById("redactionEngine");
    const redactionModelNameEl = document.getElementById("redactionModelName");
    const redactStatusEl = document.getElementById("redactStatus");
    const redactResultEl = document.getElementById("redactResult");
    const downloadLinkEl = document.getElementById("downloadLink");
    const summaryModelNameEl = document.getElementById("summaryModelName");
    const summaryDirectionsEl = document.getElementById("summaryDirections");
    const summaryStatusEl = document.getElementById("summaryStatus");
    const summaryResultEl = document.getElementById("summaryResult");
    const summaryDownloadLinkEl = document.getElementById("summaryDownloadLink");
    let lastDownloadUrl = null;
    let lastSummaryDownloadUrl = null;

    function syncModelSelectorState() {
      const useModel = detectorEl.value === "model";
      modelNameEl.disabled = !useModel;
    }

    detectorEl.addEventListener("change", syncModelSelectorState);
    syncModelSelectorState();

    function syncRedactionModelSelectorState() {
      const useModel = redactionEngineEl.value === "model";
      redactionModelNameEl.disabled = !useModel;
    }

    redactionEngineEl.addEventListener("change", syncRedactionModelSelectorState);
    syncRedactionModelSelectorState();

    document.getElementById("submit").addEventListener("click", async () => {
      const fileInput = document.getElementById("file");
      if (!fileInput.files.length) {
        statusEl.textContent = "Please choose a file first.";
        return;
      }
      statusEl.textContent = "Analyzing...";
      resultEl.textContent = "";
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      formData.append("detector", detectorEl.value);
      formData.append("model_name", modelNameEl.value);
      try {
        const response = await fetch("/analyze", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Request failed");
        }
        statusEl.textContent = "Analysis complete.";
        resultEl.textContent = JSON.stringify(payload, null, 2);
      } catch (error) {
        statusEl.textContent = "Analysis failed.";
        resultEl.textContent = String(error);
      }
    });

    document.getElementById("redactSubmit").addEventListener("click", async () => {
      const fileInput = document.getElementById("redactFile");
      if (!fileInput.files.length) {
        redactStatusEl.textContent = "Please choose a file first.";
        return;
      }

      if (lastDownloadUrl) {
        URL.revokeObjectURL(lastDownloadUrl);
        lastDownloadUrl = null;
      }

      redactStatusEl.textContent = "Redacting...";
      redactResultEl.textContent = "";
      downloadLinkEl.hidden = true;
      downloadLinkEl.removeAttribute("href");

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      formData.append("redaction_engine", redactionEngineEl.value);
      formData.append("model_name", redactionModelNameEl.value);

      try {
        const response = await fetch("/redact", { method: "POST", body: formData });
        if (!response.ok) {
          let detail = "Request failed";
          try {
            const payload = await response.json();
            detail = payload.detail || detail;
          } catch (_error) {}
          throw new Error(detail);
        }

        const blob = await response.blob();
        const statsHeader = response.headers.get("x-redaction-stats");
        let stats = {};
        if (statsHeader) {
          try {
            stats = JSON.parse(statsHeader);
          } catch (_error) {}
        }

        const contentDisposition = response.headers.get("content-disposition") || "";
        const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
        const filename = filenameMatch ? filenameMatch[1] : "redacted-document";

        const downloadUrl = URL.createObjectURL(blob);
        lastDownloadUrl = downloadUrl;
        downloadLinkEl.href = downloadUrl;
        downloadLinkEl.download = filename;
        downloadLinkEl.textContent = `Download ${filename}`;
        downloadLinkEl.hidden = false;

        redactStatusEl.textContent = "Redaction complete.";
        redactResultEl.textContent = JSON.stringify(stats, null, 2);
      } catch (error) {
        redactStatusEl.textContent = "Redaction failed.";
        redactResultEl.textContent = String(error);
      }
    });

    document.getElementById("summarySubmit").addEventListener("click", async () => {
      const fileInput = document.getElementById("summaryFile");
      if (!fileInput.files.length) {
        summaryStatusEl.textContent = "Please choose a file first.";
        return;
      }

      if (lastSummaryDownloadUrl) {
        URL.revokeObjectURL(lastSummaryDownloadUrl);
        lastSummaryDownloadUrl = null;
      }

      summaryStatusEl.textContent = "Generating document summary...";
      summaryResultEl.textContent = "";
      summaryDownloadLinkEl.hidden = true;
      summaryDownloadLinkEl.removeAttribute("href");

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      formData.append("model_name", summaryModelNameEl.value);
      formData.append("additional_directions", summaryDirectionsEl.value);

      try {
        const response = await fetch("/document-summary", { method: "POST", body: formData });
        if (!response.ok) {
          let detail = "Request failed";
          try {
            const payload = await response.json();
            detail = payload.detail || detail;
          } catch (_error) {}
          throw new Error(detail);
        }

        const blob = await response.blob();
        const statsHeader = response.headers.get("x-document-summary-stats");
        let stats = {};
        if (statsHeader) {
          try {
            stats = JSON.parse(statsHeader);
          } catch (_error) {}
        }

        const contentDisposition = response.headers.get("content-disposition") || "";
        const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
        const filename = filenameMatch ? filenameMatch[1] : "document-summary.pdf";

        const downloadUrl = URL.createObjectURL(blob);
        lastSummaryDownloadUrl = downloadUrl;
        summaryDownloadLinkEl.href = downloadUrl;
        summaryDownloadLinkEl.download = filename;
        summaryDownloadLinkEl.textContent = `Download ${filename}`;
        summaryDownloadLinkEl.hidden = false;

        summaryStatusEl.textContent = "Document summary complete.";
        summaryResultEl.textContent = JSON.stringify(stats, null, 2);
      } catch (error) {
        summaryStatusEl.textContent = "Document summary failed.";
        summaryResultEl.textContent = String(error);
      }
    });
  </script>
</body>
</html>
"""
        .replace("__MODEL_OPTIONS__", model_options_markup)
        .replace("__DOCUMENT_SUMMARY_MODEL_OPTIONS__", document_summary_model_options_markup)
    )


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    detector: str = Form(DETECTOR_HEURISTIC),
    model_name: str | None = Form(None),
) -> dict:
    extension = _validate_file(file)
    selected_detector = _validate_detector(detector)
    selected_model_name = None
    if selected_detector == DETECTOR_MODEL:
        selected_model_name = _validate_model_name(model_name)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {MAX_UPLOAD_BYTES} bytes.",
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)

        result, extracted_text = analyze_file_authorship(
            temp_path,
            filename=file.filename,
            detector=selected_detector,
            model_name=selected_model_name if selected_detector == DETECTOR_MODEL else None,
        )
        preview = extracted_text[:TEXT_PREVIEW_LENGTH]

        return {
            "filename": file.filename,
            "detector_requested": selected_detector,
            "available_detectors": sorted(SUPPORTED_DETECTORS),
            "model_name_requested": selected_model_name if selected_detector == DETECTOR_MODEL else None,
            "available_model_names": [model for model, _ in MODEL_DROPDOWN_OPTIONS],
            "label": result.label,
            "ai_probability": result.ai_probability,
            "ai_probability_percent": round(result.ai_probability * 100, 2),
            "confidence": result.confidence,
            "confidence_percent": round(result.confidence * 100, 2),
            "summary": result.summary,
            "features": result.features.__dict__,
            "text_preview": preview,
            "text_preview_truncated": len(extracted_text) > TEXT_PREVIEW_LENGTH,
            "disclaimer": (
                "This score is probabilistic and should be used with human review, "
                "metadata checks, and plagiarism/originality tools."
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze file: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/redact")
async def redact_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    redaction_engine: str = Form(REDACTION_ENGINE_COMPREHEND),
    model_name: str | None = Form(None),
) -> FileResponse:
    extension = _validate_redaction_file(file)
    selected_redaction_engine = _validate_redaction_engine(redaction_engine)
    selected_model_name = None
    if selected_redaction_engine == REDACTION_ENGINE_MODEL:
        selected_model_name = _validate_model_name(model_name)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {MAX_UPLOAD_BYTES} bytes.",
        )

    input_path: Path | None = None
    output_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(content)
            input_path = Path(tmp.name)

        output_path, metrics = redact_uploaded_file(
            input_path,
            filename=file.filename or f"document{extension}",
            redaction_engine=selected_redaction_engine,
            model_name=selected_model_name,
        )
        stats = {
            "redaction_engine": selected_redaction_engine,
            "model_name_requested": selected_model_name,
            "model_name_used": metrics.get("model_name_used"),
            "total_findings_detected": metrics.get("total_findings_detected", 0),
            "unique_phrases_detected": metrics.get("unique_phrases_detected", 0),
            "total_redactions": metrics.get("total_redactions", 0),
            "pages_with_redactions": metrics.get("pages_with_redactions"),
            "redactions_per_page": metrics.get("redactions_per_page", {}),
            "entities_by_type": metrics.get("entities_by_type", {}),
        }

        background_tasks.add_task(_safe_unlink, input_path)
        background_tasks.add_task(_safe_unlink, output_path)

        return FileResponse(
            path=str(output_path),
            media_type="application/pdf",
            filename=build_redacted_filename(file.filename or f"document{extension}"),
            background=background_tasks,
            headers={"X-Redaction-Stats": json.dumps(stats, separators=(",", ":"))},
        )
    except ValueError as exc:
        if input_path:
            _safe_unlink(input_path)
        if output_path:
            _safe_unlink(output_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if input_path:
            _safe_unlink(input_path)
        if output_path:
            _safe_unlink(output_path)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if input_path:
            _safe_unlink(input_path)
        if output_path:
            _safe_unlink(output_path)
        raise HTTPException(status_code=500, detail=f"Failed to redact file: {exc}") from exc


@app.post("/document-summary")
async def generate_document_summary(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str | None = Form(None),
    additional_directions: str | None = Form(None),
) -> FileResponse:
    extension = _validate_document_summary_file(file)
    selected_model_name = _validate_model_name(model_name, default_model=DEFAULT_DOCUMENT_SUMMARY_MODEL)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {MAX_UPLOAD_BYTES} bytes.",
        )

    input_path: Path | None = None
    output_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(content)
            input_path = Path(tmp.name)

        output_path, metrics = generate_document_summary_file(
            input_path,
            filename=file.filename or f"document{extension}",
            model_name=selected_model_name,
            additional_directions=additional_directions,
        )
        stats = {
            "model_name_requested": selected_model_name,
            "model_name_used": metrics.get("model_name_used"),
            "input_characters": metrics.get("input_characters", 0),
            "characters_analyzed": metrics.get("characters_analyzed", 0),
            "total_chunks": metrics.get("total_chunks", 0),
            "chunks_analyzed": metrics.get("chunks_analyzed", 0),
            "chunk_char_limit": metrics.get("chunk_char_limit"),
            "key_point_count": metrics.get("key_point_count", 0),
            "additional_directions_provided": metrics.get("additional_directions_provided", False),
            "additional_directions_length": metrics.get("additional_directions_length", 0),
            "disclaimer": (
                "AI-generated summary for review support only. "
                "Verify against the source document."
            ),
        }

        background_tasks.add_task(_safe_unlink, input_path)
        background_tasks.add_task(_safe_unlink, output_path)

        return FileResponse(
            path=str(output_path),
            media_type="application/pdf",
            filename=build_document_summary_filename(file.filename or f"document{extension}"),
            background=background_tasks,
            headers={"X-Document-Summary-Stats": json.dumps(stats, separators=(",", ":"))},
        )
    except ValueError as exc:
        if input_path:
            _safe_unlink(input_path)
        if output_path:
            _safe_unlink(output_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if input_path:
            _safe_unlink(input_path)
        if output_path:
            _safe_unlink(output_path)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if input_path:
            _safe_unlink(input_path)
        if output_path:
            _safe_unlink(output_path)
        raise HTTPException(status_code=500, detail=f"Failed to generate document summary: {exc}") from exc
