#!/usr/bin/env python3
"""Upload a user's PDF file to the S3 bucket for processing."""

import boto3
import sys
from pathlib import Path

# Get the file path from command line or use the default
file_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/deffenda/Downloads/health_report.pdf"
file_path = Path(file_path)

if not file_path.exists():
    print(f"✗ File not found: {file_path}")
    sys.exit(1)

if not file_path.suffix.lower() == '.pdf':
    print(f"✗ File must be a PDF. Got: {file_path.suffix}")
    sys.exit(1)

# S3 configuration
s3_client = boto3.client('s3', region_name='us-east-1')
bucket = 'abd-redactor-20260226135602-8965'
key = f'incoming/{file_path.name}'

print(f"Uploading {file_path.name}...")
print(f"File size: {file_path.stat().st_size} bytes\n")

try:
    s3_client.upload_file(str(file_path), bucket, key)
    print(f"✓ Successfully uploaded to S3!")
    print(f"  Bucket: {bucket}")
    print(f"  Key: {key}")
    print(f"  S3 URI: s3://{bucket}/{key}")
    print("\n" + "="*60)
    print("File is ready for processing!")
    print("="*60)
    print("\nThe Lambda function will:")
    print("  1. Detect your file in the incoming/ folder")
    print("  2. Extract text from the PDF")
    print("  3. Use AWS Comprehend to identify PII")
    print("  4. Redact sensitive information")
    print(f"  5. Save the redacted PDF to redacted/{file_path.name}")
except Exception as e:
    print(f"✗ Upload failed: {e}")
    sys.exit(1)
