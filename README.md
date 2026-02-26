# AWS PDF Redaction Pipeline (Python Lambda)

This project deploys an AWS Lambda that is triggered when a PDF is uploaded to an S3 "directory" (prefix).  
The Lambda uses Amazon Comprehend (AI PII detection) + PyMuPDF to apply black-box redactions, then writes a new PDF back to S3.

## What Gets Deployed

- `PdfPipelineBucket` (S3 bucket)
  - Input prefix (default): `incoming/`
  - Output prefix (default): `redacted/`
- `PdfRedactionFunction` (Lambda, Python 3.12)
- IAM permissions for:
  - Read/write S3 objects in this bucket
  - `comprehend:DetectPiiEntities`

## Project Structure

- `template.yaml` - AWS SAM infrastructure definition
- `src/app.py` - Lambda redaction logic
- `src/requirements.txt` - Python dependencies
- `events/s3-put.json` - sample event for local Lambda invoke

## Prerequisites

- AWS account + credentials configured (`aws configure`)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- Docker (required for `sam build --use-container`)

## Deploy

```bash
sam build --use-container
sam deploy --guided
```

When prompted during `sam deploy --guided`:

- Allow SAM to create IAM roles/policies.
- Keep defaults unless you want custom prefixes.
- Example custom values:
  - `InputPrefix`: `incoming/`
  - `OutputPrefix`: `redacted/`
  - `MinEntityScore`: `0.8`
  - `ComprehendLanguage`: `en`

After deploy, note stack outputs:

- `BucketName`
- `InputPrefix`
- `OutputPrefix`

Or use the included script:

```bash
./scripts/deploy.sh pdf-redaction-pipeline
```

## Use It

Upload a PDF:

```bash
aws s3 cp ./sample.pdf s3://<BucketName>/incoming/sample.pdf
```

The Lambda will create:

```text
s3://<BucketName>/redacted/sample-redacted.pdf
```

Download result:

```bash
aws s3 cp s3://<BucketName>/redacted/sample-redacted.pdf ./sample-redacted.pdf
```

## Local Test (Optional)

Build once:

```bash
sam build --use-container
```

Update `events/s3-put.json` with your bucket/key and run:

```bash
sam local invoke PdfRedactionFunction --event events/s3-put.json
```

## Notes / Limits

- Best on text-based PDFs. Scanned/image-only PDFs need OCR before reliable redaction.
- Redaction quality depends on Comprehend confidence threshold (`MinEntityScore`).
- Output files are written to `redacted/`, avoiding recursive reprocessing.

## Cleanup

```bash
sam delete
```
