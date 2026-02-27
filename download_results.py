#!/usr/bin/env python3
"""Download redacted PDF and display report summary."""

import boto3
from pathlib import Path
import json

s3_client = boto3.client('s3', region_name='us-east-1')
bucket = 'abd-redactor-20260226135602-8965'

# Download redacted PDF
redacted_key = 'redacted/health_report-redacted.pdf'
local_path = Path.home() / 'Desktop' / 'health_report-redacted.pdf'
print("Downloading redacted PDF...")
s3_client.download_file(bucket, redacted_key, str(local_path))
print(f"✓ Saved to: {local_path}\n")

# Get report
report_key = 'reports/health_report-redaction-report.json'
response = s3_client.get_object(Bucket=bucket, Key=report_key)
report = json.loads(response['Body'].read())

print("=" * 70)
print("REDACTION REPORT SUMMARY")
print("=" * 70)
print(f"\n📊 Processing Time: {report['processing']['total_duration_seconds']}s")
print(f"📄 Total Pages: {report['processing']['total_pages_processed']}")
print(f"✅ Pages with Redactions: {report['processing']['pages_with_redactions']}")
print(f"📍 Coverage: {report['processing']['coverage_percentage']}%")
print(f"\n🔍 PII Detected: {report['pii_detection']['redaction_boxes_applied']} entities")
print(f"⚡ Effectiveness: {report['quality_metrics']['redaction_effectiveness_score']}%")
print(f"\n💾 Original Size: {report['file_metrics']['original_file_size_kb']} KB")
print(f"📦 Redacted Size: {report['file_metrics']['redacted_file_size_kb']} KB")
print(f"📉 Compression: {report['file_metrics']['compression_ratio_percent']}%")

print("\n📄 REDACTIONS PER PAGE:")
for page, count in report['pii_detection']['redactions_per_page'].items():
    print(f"  • Page {page}: {count} redactions")

print("\n" + "=" * 70)
