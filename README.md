# AWS PDF Redaction Pipeline (Chunked + Batched Comprehend)

This project deploys an event-driven AWS pipeline that:

1. Accepts large PDFs uploaded to S3.
2. Chunks PDF text into Comprehend-safe segments.
3. Batches chunks and processes them in parallel with Amazon Comprehend PII detection.
4. Produces a fully redacted PDF.
5. Produces a JSON report with total redaction changes made.

## Architecture

- `PdfPipelineBucket` (S3)
  - Input prefix: `incoming/`
  - Redacted output prefix: `redacted/`
  - Report prefix: `reports/`
  - Intermediate work prefix: `work/`
- `StartPipelineFunction` (Lambda)
  - Triggered by EventBridge on S3 Object Created events.
  - Starts Step Functions execution per input PDF.
- `PdfRedactionStateMachine` (Step Functions)
  - `PrepareChunksFunction`:
    - Downloads source PDF
    - Extracts text by page
    - Chunks text and writes batch manifests to S3
  - `DetectBatchFunction` (Map state, parallel):
    - Reads a batch manifest
    - Calls `comprehend:DetectPiiEntities` per chunk
    - Writes batch findings to S3
  - `AssembleOutputFunction`:
    - Aggregates findings
    - Applies redactions to PDF
    - Uploads redacted PDF and JSON report

## Files

- `template.yaml` - AWS SAM infrastructure and orchestration
- `src/start_execution.py` - S3 event -> Step Functions starter
- `src/prepare_chunks.py` - chunking and batch manifest generation
- `src/detect_pii_batch.py` - per-batch Comprehend PII detection
- `src/assemble_output.py` - final redacted PDF + report generation
- `events/object-created.json` - sample EventBridge S3 event for local invoke
- `scripts/deploy.sh` - deploy helper script

## Prerequisites

- AWS account + credentials configured (`aws configure`)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- Docker (for `sam build --use-container`)

## Deploy

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

## Use It

Upload a PDF:

```bash
aws s3 cp ./sample.pdf s3://<BucketName>/incoming/sample.pdf
```

Pipeline outputs:

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

## Local Test (Optional)

Invoke the start Lambda with a sample EventBridge event:

```bash
sam build --use-container
sam local invoke StartPipelineFunction --event events/object-created.json
```

Replace `REPLACE_WITH_BUCKET_NAME` and key in `events/object-created.json` first.

## Notes

- Best with text-based PDFs. Image-only/scanned PDFs need OCR before reliable redaction.
- Large documents are handled by chunking and Step Functions parallel Map processing.
- Intermediate `work/` artifacts are deleted after successful output assembly.

## Cleanup

```bash
sam delete
```
