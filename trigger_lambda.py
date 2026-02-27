#!/usr/bin/env python3
"""Trigger the Lambda handler on the uploaded health_report.pdf file."""

import sys
sys.path.insert(0, 'src')
import json
from app import lambda_handler

# Create an S3 event for the health_report.pdf file
s3_event = {
    "Records": [
        {
            "eventSource": "aws:s3",
            "s3": {
                "bucket": {"name": "abd-redactor-20260226135602-8965"},
                "object": {"key": "incoming/health_report.pdf"}
            }
        }
    ]
}

print("Triggering Lambda on health_report.pdf...")
print("=" * 60)

try:
    result = lambda_handler(s3_event, None)
    print("✓ Lambda execution successful!\n")
    print("Results:")
    print(json.dumps(result, indent=2))
    
    if result.get("processed", 0) > 0:
        for res in result.get("results", []):
            print("\n" + "-" * 60)
            print(f"File: {res['input_key']}")
            print(f"Redaction hits: {res['redaction_hits']}")
            print(f"Effectiveness score: {res['effectiveness_score']}")
            print(f"Processing time: {res['processing_time_seconds']}s")
            print(f"Report: {res['report_key']}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
