# ADB Document Manager (AI Detection / Redaction / Summary)

This project deploys an event-driven AWS pipeline that:

1. Accepts large PDFs uploaded to S3.
2. Chunks PDF text into Comprehend-safe segments.
3. Batches chunks and processes them in parallel with Amazon Comprehend PII detection.
4. Produces a fully redacted PDF.
5. Produces a JSON processing report with redaction quality metrics and capability statuses.
6. Produces an AI authorship report JSON artifact from the same S3-triggered document.
7. Produces a model-generated document summary PDF (chunked summarization path).

## Architecture

### Two Processing Modes

#### 1. Step Functions Pipeline (Production)
Event-driven orchestration for large-scale document processing:
- Auto-triggered by S3 uploads to `incoming/`
- Chunks large PDFs into Comprehend-safe segments
- Batches chunks for parallel processing
- Scales efficiently with Map states

**Flow:**
- `StartPipelineFunction` (Lambda) → triggered by S3 event
- `PdfRedactionStateMachine` (Step Functions) orchestrates
  - `PrepareChunksFunction` - downloads PDF, chunks text
  - `DetectBatchFunction` (Map state) - parallel Comprehend processing
  - `AssembleRedactionFunction` - aggregates batch findings and writes redacted PDF + base metrics
  - `GenerateCapabilities` (Parallel) 
    - `GenerateAuthorshipArtifactFunction` - writes S3 authorship report artifact
    - `GenerateDocumentSummaryArtifactFunction` - writes S3 document summary PDF artifact
  - `FinalizeReportFunction` - merges capability statuses and writes the final processing report

#### 2. Direct/Web Handlers (Quick Testing)
Fast local testing using the Lambda handler directly:
- `src/app.py` - standalone PDF redaction engine (used by `/redact`)
- `src/ai_authorship_bot.py` - local web app for AI detection, redaction download, and document summary PDF generation
- Process files immediately without Step Functions
- Ideal for development and quick tests
- Use `process_files.py` for redaction-engine tests or the web UI for all capabilities

### S3 Bucket Structure

- `PdfPipelineBucket` (S3)
  - Input prefix: `incoming/`
  - Redacted output prefix: `redacted/`
  - Report prefix: `reports/`
  - Intermediate work prefix: `work/`

## Files

### Core Application
- `template.yaml` - AWS SAM infrastructure and orchestration
- `layer/requirements.txt` - shared Lambda dependency layer package manifest
- `src/start_execution.py` - S3 event -> Step Functions starter
- `src/prepare_chunks.py` - chunking and batch manifest generation
- `src/detect_pii_batch.py` - per-batch Comprehend PII detection
- `src/assemble_redaction.py` - Lambda handler for redaction assembly step
- `src/generate_authorship_artifact.py` - Lambda handler for S3 authorship artifact generation
- `src/generate_document_summary_artifact.py` - Lambda handler for S3 summary artifact generation
- `src/finalize_report.py` - Lambda handler for final report merge + cleanup
- `src/assemble_output.py` - shared pipeline logic used by the modular Lambda handlers
- `src/app.py` - Direct PDF redaction handler (alternative to Step Functions)

### Helper Scripts
- `upload_user_file.py` - Upload your own PDF files to S3 for processing
- `upload_test_file.py` - Create and upload a test PDF with sample PII
- `check_s3_status.py` - Check bucket contents and processing status
- `process_files.py` - Directly invoke the redaction handler on uploaded files
- `test_app.py` - Test the Lambda handler functionality

### Configuration & Events
- `events/object-created.json` - sample EventBridge S3 event for local invoke
- `scripts/deploy.sh` - deploy helper script
- `samconfig.toml` - SAM configuration

## Setup

### Prerequisites

- AWS account + credentials configured (`aws configure`)
- Python 3.9+
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) (for deployment)
- Docker (for `sam build --use-container`)

### Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Packaging note:
- `requirements-local.txt` is for local development/testing.
- `src/requirements.txt` is intentionally minimal for Lambda artifact packaging.
- Heavy shared dependencies are built into `SharedDependenciesLayer` from `layer/requirements.txt`.

## ADB Document Manager (AI Detection / Redaction / Summary)

A lightweight upload API is available for:
- AI authorship likelihood analysis
- Direct upload redaction with downloadable output (without waiting for S3 trigger flow)
- Document summary PDF generation with optional user directions

The same three capabilities are also available in the S3-triggered pipeline:
- Upload a PDF to `s3://<BucketName>/incoming/`
- Step Functions automatically runs redaction, AI detection, and document summary generation
- Outputs are written to `redacted/` and `reports/`

- Entry point: `src/ai_authorship_bot.py`
- Supported uploads: `.pdf`, `.docx`
- Browser UI: `GET /`
- Selectable detectors: `heuristic` (default) and `model`
- Model dropdown for model-backed flows includes OpenAI and AWS Bedrock options (Claude 3.7 Sonnet, Claude 3.5 Sonnet v2, Nova Pro, Nova Lite)
- Detection endpoint: `POST /analyze` (multipart file upload)
- Redaction endpoint: `POST /redact` (PDF only; returns redacted file download)
- Redaction engines:
  - `comprehend` (default): Comprehend + chunking + dedupe
  - `model`: model-based findings + chunking + dedupe

Run locally:

```bash
source .venv/bin/activate
uvicorn ai_authorship_bot:app --app-dir src --reload --port 8000
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "detector=heuristic" \
  -F "file=@/path/to/document.pdf"
```

Model-backed request with explicit model:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "detector=model" \
  -F "model_name=gpt-5.2-pro" \
  -F "file=@/path/to/document.pdf"
```

Redaction request:

```bash
curl -X POST "http://127.0.0.1:8000/redact" \
  -F "redaction_engine=comprehend" \
  -F "file=@/path/to/document.pdf" \
  -o ./document-redacted.pdf
```

Model-based redaction request:

```bash
curl -X POST "http://127.0.0.1:8000/redact" \
  -F "redaction_engine=model" \
  -F "model_name=gpt-5" \
  -F "file=@/path/to/document.pdf" \
  -o ./document-redacted.pdf
```

Document summary request:

```bash
curl -X POST "http://127.0.0.1:8000/document-summary" \
  -F "model_name=gpt-5" \
  -F "additional_directions=Focus on risks, deadlines, and action items." \
  -F "file=@/path/to/document.pdf" \
  -o ./document-summary.pdf
```

Notes:
- `heuristic` is local and requires no external API.
- `model` supports OpenAI and AWS Bedrock model choices from the dropdown.
- OpenAI selections require `OPENAI_API_KEY`.
- AWS Bedrock selections require AWS credentials plus Bedrock model access in your configured region.
- Optional Bedrock setting:
  - `AWS_BEDROCK_REGION` (optional override; otherwise uses `AWS_REGION`/`AWS_DEFAULT_REGION`)
- Optional model settings:
  - `OPENAI_AUTHORSHIP_MODEL` (default: `gpt-4.1-mini`)
  - `MODEL_INPUT_CHAR_LIMIT` (default: `12000`)
- `/redact` uses the same Comprehend + chunking + dedupe logic as the S3 pipeline.
- Optional redaction model settings:
  - `OPENAI_REDACTION_MODEL` (default fallback: `OPENAI_AUTHORSHIP_MODEL`, then `gpt-4.1`)
  - `MODEL_REDACTION_INPUT_CHAR_LIMIT` (default: `9000`)
- `/document-summary` supports `.pdf` and `.docx`, and returns a downloadable `.pdf` file.
- `additional_directions` lets users tailor the summary from a textbox in the web UI.
- Optional document summary model settings:
  - `OPENAI_DOCUMENT_SUMMARY_MODEL` (fallback: `OPENAI_AUTHORSHIP_MODEL`, then `gpt-4.1-mini`)
  - `DOCUMENT_SUMMARY_INPUT_CHAR_LIMIT` (per-chunk model input cap, default: `16000`)
  - `DOCUMENT_SUMMARY_CHUNK_CHAR_LIMIT` (summary chunk size; fallback: `CHUNK_CHAR_LIMIT`, default: `4500`)
  - `DOCUMENT_SUMMARY_MAX_CHUNKS` (maximum chunks analyzed, default: `12`)
- Use it with manual review and document metadata checks.

## Deploy to AWS

### Full SAM Deployment

```bash
sam build --use-container
sam deploy --guided
```

The SAM template includes a shared dependency Lambda Layer (`SharedDependenciesLayer`) built from `layer/requirements.txt` and attached to all pipeline functions.

Or use script:

```bash
./scripts/deploy.sh adb-document-manager
```

After deploy, note outputs:

- `BucketName`
- `InputPrefix`
- `OutputPrefix`
- `ReportPrefix`
- `StateMachineArn`

S3-trigger capability settings (SAM parameters / Lambda env):

- `EnableS3Authorship` (`true`/`false`, default `true`)
- `EnableS3DocumentSummary` (`true`/`false`, default `true`)
- `RequireS3Capabilities` (`true`/`false`, default `false`)
- `S3AuthorshipDetector` (`heuristic` or `model`, default `heuristic`)
- `S3AuthorshipModel` (optional model name when detector=`model`)
- `S3SummaryModel` (optional summary model override)
- `S3SummaryDirections` (optional extra summary prompt instructions)
- `OpenAIApiKey` (optional, required for model-backed S3 capabilities)

Capability availability:
- `S3 trigger (incoming/)`: redacted PDF + redaction report + authorship report + document summary PDF
- `Web upload API`: `/analyze` (AI detection), `/redact` (redaction), `/document-summary` (summary)

Reporting differences by capability (S3-triggered):
- `Redaction`: writes `reports/<name>-redaction-report.json` (primary processing report) plus `redacted/<name>-redacted.pdf`.
- `AI detection (authorship)`: writes `reports/<name>-authorship-report.json` with detector/model, label, probability, confidence, features, and text preview.
- `Document summary`: writes `reports/<name>-document-summary.pdf` and stores summary run metrics/status under `capabilities.document_summary` in the final processing report.
- Final report merge: `reports/<name>-redaction-report.json` includes `capabilities.authorship` and `capabilities.document_summary` with `status` values (`COMPLETED`, `FAILED`, or `SKIPPED`) and output keys when completed.

## Quick Start (Local Testing)

### 1. Test the Handler
```bash
source .venv/bin/activate
python3 test_app.py
```

### 2. Upload a PDF
```bash
# Upload your own PDF
python3 upload_user_file.py "/path/to/your/document.pdf"

# Or create and upload a test PDF with sample PII
python3 upload_test_file.py
```

### 3. Wait for Automatic S3 Processing
After upload, the S3 trigger automatically starts Step Functions processing for redaction, AI detection, and document summary.

### 4. Check Upload Status
```bash
python3 check_s3_status.py
```

### 5. Optional: Run Direct Local Redaction Path
This is a local quick-test path for redaction only (not the full S3-triggered capability flow).
```bash
python3 process_files.py
```

### 6. Check Results
```bash
python3 check_s3_status.py
```

The redacted PDFs will be in `redacted/`, and capability artifacts/reports will be in `reports/` in S3.

## Use It (AWS Deployment)

### Upload a PDF

#### Option A: Using Helper Script
```bash
source .venv/bin/activate
python3 upload_user_file.py "/path/to/your/document.pdf"
```

#### Option B: AWS CLI
```bash
aws s3 cp ./sample.pdf s3://<BucketName>/incoming/sample.pdf
```

After uploading, the pipeline automatically starts processing your PDF through:
- **Step Functions**: Full chunking/batched Comprehend processing plus authorship and document summary artifact generation
- **Web API / Local Paths**: Immediate AI detection, redaction download, and document summary generation for manual uploads (`src/ai_authorship_bot.py`); `process_files.py` remains a fast redaction-engine test path
- AI detection (`sample-authorship-report.json`) and document summary (`sample-document-summary.pdf`) are generated by the S3-triggered path by default when enabled.

### Pipeline Outputs

- `s3://<BucketName>/redacted/sample-redacted.pdf`
- `s3://<BucketName>/reports/sample-redaction-report.json`
- `s3://<BucketName>/reports/sample-authorship-report.json`
- `s3://<BucketName>/reports/sample-document-summary.pdf`

Download results:

```bash
aws s3 cp s3://<BucketName>/redacted/sample-redacted.pdf ./sample-redacted.pdf
aws s3 cp s3://<BucketName>/reports/sample-redaction-report.json ./sample-redaction-report.json
aws s3 cp s3://<BucketName>/reports/sample-authorship-report.json ./sample-authorship-report.json
aws s3 cp s3://<BucketName>/reports/sample-document-summary.pdf ./sample-document-summary.pdf
```

## Report Content

### Redaction Processing Report (`*-redaction-report.json`)
- Primary orchestration report for each S3-triggered document.
- Includes processing timings, file metrics, pipeline/chunking metadata, PII findings, quality/compliance metrics, and output keys.
- Includes capability merge results under:
  - `capabilities.authorship`
  - `capabilities.document_summary`
- When completed, `output.authorship_report_key` and/or `output.document_summary_pdf_key` are added.

### Authorship Artifact (`*-authorship-report.json`)
- Separate JSON artifact focused on AI detection output.
- Includes:
  - `detector_used`, `model_name_used`
  - `label`, `ai_probability`, `confidence`, `summary`
  - `features` (authorship feature vector)
  - `text_preview` and truncation flag

### Document Summary Artifact (`*-document-summary.pdf`)
- Separate PDF artifact containing the generated summary text.
- Summary generation metrics are tracked in the final processing report under `capabilities.document_summary` (for example: chunk counts, analyzed characters, model used, and direction flags).

### Web API Reporting (direct upload)
- `POST /analyze`: returns JSON body with detection result + feature payload.
- `POST /redact`: returns redacted PDF file and `X-Redaction-Stats` response header (JSON stats).
- `POST /document-summary`: returns summary PDF file and `X-Document-Summary-Stats` response header (JSON stats).

## Lambda Handlers

- `src/start_execution.py`: Handles S3 object-created events and starts the Step Functions execution.
- `src/prepare_chunks.py`: Extracts and chunks PDF text, then writes batch manifests to S3.
- `src/detect_pii_batch.py`: Processes chunk batches with Comprehend for PII detection.
- `src/assemble_redaction.py` and `src/assemble_output.py`: Applies redactions, writes the redacted PDF, and builds base report data.
- `src/generate_authorship_artifact.py`: Generates the AI-authorship analysis artifact for the same document.
- `src/generate_document_summary_artifact.py`: Generates the document summary PDF artifact for the same document.
- `src/finalize_report.py`: Merges redaction and capability statuses into the final processing report.

This modular split improves scalability and maintainability. Each Lambda can be deployed and scaled independently.

## Local Testing with SAM (Optional)

For direct Step Functions testing, invoke with sample EventBridge event:

```bash
sam build --use-container
sam local invoke StartPipelineFunction --event events/object-created.json
```

Update `events/object-created.json` with your bucket name and file path first.

Or use the helper scripts for end-to-end local testing (see Quick Start above).

## Notes

- Best with text-based PDFs. Image-only/scanned PDFs need OCR before reliable redaction.
- Large documents are handled by chunking and Step Functions parallel Map processing.
- Intermediate `work/` artifacts are deleted after successful output assembly.

## Cleanup

```bash
sam delete
```
