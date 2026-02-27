#!/usr/bin/env python3
"""Create and upload a test PDF for the document manager app to process."""

import sys
sys.path.insert(0, 'src')
import fitz
import boto3
from pathlib import Path

# Create a test PDF with PII content
pdf_path = Path("/tmp/test_document.pdf")
doc = fitz.open()
page = doc.new_page()

# Add text with various PII
text = """CONFIDENTIAL DOCUMENT

Employee Information:
Name: John Smith
Email: john.smith@company.com
Phone: +1-555-123-4567
SSN: 123-45-6789

Medical Records:
Patient: Jane Doe
Date of Birth: 1990-05-15
Doctor: Dr. Robert Johnson
Diagnosis: Hypertension treatment

Financial Information:
Credit Card: 4532-1111-2222-3333
Account Number: 98765432

Address: 123 Main Street, New York, NY 10001
"""

# Insert text with some margin
text_rect = fitz.Rect(36, 36, page.rect.width - 36, page.rect.height - 36)
page.insert_textbox(text_rect, text, fontsize=11, color=(0, 0, 0))
doc.save(str(pdf_path))
doc.close()

print(f"✓ Created test PDF: {pdf_path}")
print(f"  File size: {pdf_path.stat().st_size} bytes\n")

# Upload to S3
s3_client = boto3.client('s3', region_name='us-east-1')
bucket = 'abd-redactor-20260226135602-8965'
key = 'incoming/test_document.pdf'

try:
    s3_client.upload_file(str(pdf_path), bucket, key)
    print(f"✓ Successfully uploaded to S3!")
    print(f"  Bucket: {bucket}")
    print(f"  Key: {key}")
    print(f"  S3 URI: s3://{bucket}/{key}")
    print("\n" + "="*60)
    print("File is now ready for processing!")
    print("="*60)
    print("\nThe pipeline will:")
    print("  1. Detect the uploaded file")
    print("  2. Extract text from the PDF")
    print("  3. Use AWS Comprehend to identify PII")
    print("  4. Redact the PII entities")
    print(f"  5. Upload the redacted PDF to s3://{bucket}/redacted/test_document.pdf")
    print(f"  6. Generate authorship and document summary artifacts under s3://{bucket}/reports/")
except Exception as e:
    print(f"✗ Upload failed: {e}")
