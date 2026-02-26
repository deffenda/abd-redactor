#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import json
import os

# Set dummy AWS credentials for testing
os.environ['AWS_ACCESS_KEY_ID'] = 'test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'

print("="*60)
print("PDF Redaction Quality App - Test Run")
print("="*60 + "\n")

try:
    from app import lambda_handler, parse_s3_records
    
    # Create a test S3 event
    s3_event = {
        "Records": [{
            "eventSource": "aws:s3",
            "s3": {
                "bucket": {"name": "test-bucket"},
                "object": {"key": "incoming/sample.pdf"}
            }
        }]
    }
    
    print("✓ Successfully imported lambda_handler")
    print("✓ Handler is ready to process S3 events\n")
    
    # Parse the event
    objects = parse_s3_records(s3_event)
    print(f"✓ Parsed S3 event: {len(objects)} object(s) to process")
    for obj in objects:
        print(f"  - Bucket: {obj.bucket}, Key: {obj.key}\n")
    
    print("-"*60)
    print("App Status: Ready")
    print("-"*60)
    print("\nNote: Full execution requires:")
    print("  • AWS credentials configured")
    print("  • S3 buckets and PDF files")
    print("  • AWS Comprehend service access")
    print("\nThe app will:")
    print("  1. Download PDFs from S3")
    print("  2. Redact PII using AWS Comprehend")
    print("  3. Upload redacted PDFs to output bucket")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
