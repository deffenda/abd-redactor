#!/usr/bin/env python3
"""Invoke the Lambda handler to process the uploaded PDF files."""

import sys
sys.path.insert(0, 'src')
import json
from app import lambda_handler

# Create an S3 event for the uploaded files
s3_event = {
    "Records": [
        {
            "eventSource": "aws:s3",
            "s3": {
                "bucket": {"name": "abd-redactor-20260226135602-8965"},
                "object": {"key": "incoming/health_report.pdf"}
            }
        },
        {
            "eventSource": "aws:s3",
            "s3": {
                "bucket": {"name": "abd-redactor-20260226135602-8965"},
                "object": {"key": "incoming/test_document.pdf"}
            }
        }
    ]
}

print("Processing PDFs with Lambda handler...\n")
print("="*60)

try:
    result = lambda_handler(s3_event, None)
    print("✓ Processing completed!")
    print("\nResults:")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
