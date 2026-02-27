#!/usr/bin/env python3
"""Display the enhanced redaction report."""

import boto3
import json

s3_client = boto3.client('s3', region_name='us-east-1')
bucket = 'abd-redactor-20260226135602-8965'
key = 'reports/health_report-redaction-report.json'

response = s3_client.get_object(Bucket=bucket, Key=key)
report = json.loads(response['Body'].read())

print("\n" + "=" * 70)
print("ENHANCED REDACTION REPORT - health_report.pdf")
print("=" * 70 + "\n")

print("📊 PROCESSING:")
for k, v in report.get('processing', {}).items():
    print(f"  • {k.replace('_', ' ').title()}: {v}")

print("\n⚡ PERFORMANCE METRICS:")
for k, v in report.get('performance_metrics', {}).items():
    print(f"  • {k.replace('_', ' ').title()}: {v}")

print("\n📁 FILE METRICS:")
for k, v in report.get('file_metrics', {}).items():
    print(f"  • {k.replace('_', ' ').title()}: {v}")

print("\n🔍 PII DETECTION:")
for k, v in report.get('pii_detection', {}).items():
    if k != 'redactions_per_page':
        print(f"  • {k.replace('_', ' ').title()}: {v}")

print("\n📄 REDACTIONS PER PAGE:")
redactions_per_page = report.get('pii_detection', {}).get('redactions_per_page', {})
if redactions_per_page:
    for page, count in redactions_per_page.items():
        print(f"  • Page {page}: {count} redactions")
else:
    print("  • No per-page data available")

print("\n✅ QUALITY METRICS:")
for k, v in report.get('quality_metrics', {}).items():
    print(f"  • {k.replace('_', ' ').title()}: {v}")

print("\n⚖️  COMPLIANCE:")
for k, v in report.get('compliance', {}).items():
    print(f"  • {k.replace('_', ' ').title()}: {v}")

print("\n" + "=" * 70)
