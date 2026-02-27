"""
Lambda handler for PDF ingestion and preprocessing.
Triggered by S3 events. Extracts text, chunks PDF, and writes batch manifests to S3.
"""
import json
import logging
import os
from pathlib import Path
import boto3
import fitz
from .prepare_chunks import chunk_text

def lambda_handler(event, _context):
    logger = logging.getLogger()
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    s3_client = boto3.client("s3")

    # Parse S3 event
    records = event.get("Records", [])
    for record in records:
        if record.get("eventSource") != "aws:s3":
            continue
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        if not key.lower().endswith(".pdf"):
            continue
        job_id = os.urandom(8).hex()
        local_pdf = Path(f"/tmp/{job_id}-input.pdf")
        s3_client.download_file(bucket, key, str(local_pdf))
        # Extract and chunk text
        with fitz.open(local_pdf) as doc:
            for page in doc:
                page_text = page.get_text("text")
                for chunk in chunk_text(page_text, 4500):
                    # ...write chunk manifest logic here...
                    pass
        # ...write manifest to S3...
    return {"status": "preprocessing complete"}
