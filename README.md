# AWS PDF Redaction Pipeline (Chunked + Batched Comprehend)

This project deploys an event-driven AWS pipeline that:

1. Accepts large PDFs uploaded to S3.
2. Chunks PDF text into Comprehend-safe segments.
3. Batches chunks and processes them in parallel with Amazon Comprehend PII detection.
4. Produces a fully redacted PDF.
5. Produces a JSON report with total redaction changes made.

## Architecture

### Two Processing Modes

#### 1. Step Functions Pipeline (Production)
Event-driven orchestration for large-scale processing:
- Auto-triggered by S3 uploads to `incoming/`
- Chunks large PDFs into Comprehend-safe segments
- Batches chunks for parallel processing
- Scales efficiently with Map states

**Flow:**
- `StartPipelineFunction` (Lambda) → triggered by S3 event
- `PdfRedactionStateMachine` (Step Functions) orchestrates
  - `PrepareChunksFunction` - downloads PDF, chunks text
  - `DetectBatchFunction` (Map state) - parallel Comprehend processing
  - `AssembleOutputFunction` - aggregates results, redacts PDF

#### 2. Direct Handler (Quick Testing)
Fast local testing using the Lambda handler directly:
- `src/app.py` - standalone PDF redaction
- Process files immediately without Step Functions
- Ideal for development and quick tests
- Use `process_files.py` to invoke

### S3 Bucket Structure

- `PdfPipelineBucket` (S3)
  - Input prefix: `incoming/`
  - Redacted output prefix: `redacted/`
  - Report prefix: `reports/`
  - Intermediate work prefix: `work/`

## Files

### Core Application
- `template.yaml` - AWS SAM infrastructure and orchestration
- `src/start_execution.py` - S3 event -> Step Functions starter
- `src/prepare_chunks.py` - chunking and batch manifest generation
- `src/detect_pii_batch.py` - per-batch Comprehend PII detection
- `src/assemble_output.py` - final redacted PDF + report generation
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
pip install -r src/requirements.txt
```

## Deploy to AWS

### Full SAM Deployment

```bash
sam build --use-container
sam deploy --guided
```

Or use script:

```bash
./scripts/deploy.sh pdf-redaction-pipeline
```

After deploy, note outputs:

- `BucketName`
- `InputPrefix`
- `OutputPrefix`
- `ReportPrefix`
- `StateMachineArn`

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

### 3. Check Upload Status
```bash
python3 check_s3_status.py
```

### 4. Process the Files
```bash
python3 process_files.py
```

### 5. Check Results
```bash
python3 check_s3_status.py
```

The redacted PDFs will be in the `redacted/` folder in S3.

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
- **Step Functions**: Full chunking and batched Comprehend processing
- **Direct Handler**: Immediate redaction (see `process_files.py`)

### Pipeline Outputs

- `s3://<BucketName>/redacted/sample-redacted.pdf`
- `s3://<BucketName>/reports/sample-redaction-report.json`

Download results:

```bash
aws s3 cp s3://<BucketName>/redacted/sample-redacted.pdf ./sample-redacted.pdf
aws s3 cp s3://<BucketName>/reports/sample-redaction-report.json ./sample-redaction-report.json
```

## Report Content

The JSON report includes:

- Input/output object locations
- Total pages/chunks/batches processed
- PII findings detected
- `redaction_boxes_applied`
- `changes_made` (same as boxes applied)
- Redactions per page
- Entity counts by PII type

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
