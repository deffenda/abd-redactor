#!/usr/bin/env python3
"""Display comprehensive redaction quality report with all metrics."""

import boto3
import json

s3_client = boto3.client('s3', region_name='us-east-1')
bucket = 'abd-redactor-20260226135602-8965'
key = 'reports/health_report-redaction-report.json'

response = s3_client.get_object(Bucket=bucket, Key=key)
report = json.loads(response['Body'].read())

print("\n" + "=" * 80)
print("COMPREHENSIVE REDACTION QUALITY REPORT")
print("=" * 80)

sections = [
    ("📊 PROCESSING", "processing"),
    ("⚡ PERFORMANCE", "performance_metrics"),
    ("📁 FILE METRICS", "file_metrics"),
    ("🔍 PII DETECTION", "pii_detection"),
    ("🎯 DETECTION ACCURACY", "detection_accuracy"),
    ("📄 DOCUMENT QUALITY", "document_quality"),
    ("⚖️ RISK ASSESSMENT", "risk_assessment"),
    ("✅ QUALITY METRICS", "quality_metrics"),
    ("📋 COMPLIANCE", "compliance"),
]

for title, section_key in sections:
    section_data = report.get(section_key, {})
    if section_data:
        print(f"\n{title}:")
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                if key not in ['entity_distribution_percentage', 'redactions_per_page', 'pages_with_high_pii_concentration', 'top_redacted_phrases', 'entity_specific_metrics', 'pages_flagged_for_review']:
                    if isinstance(value, (dict, list)):
                        print(f"  • {key.replace('_', ' ').title()}:")
                        if isinstance(value, dict):
                            for k, v in value.items():
                                print(f"      - {k.replace('_', ' ').title()}: {v}")
                        else:
                            for item in value:
                                print(f"      - {item}")
                    else:
                        print(f"  • {key.replace('_', ' ').title()}: {value}")

# Special handling for nested structures
print(f"\n🔗 ADVANCED METRICS:")
if "quality_metrics" in report and "performance" in report.get("quality_metrics", {}):
    perf = report["quality_metrics"]["performance"]
    for k, v in perf.items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

print("\n" + "=" * 80)
print("Report Summary")
print("=" * 80)
print(f"✅ Audit Readiness: {report.get('compliance', {}).get('audit_readiness', 'N/A')}%")
print(f"⚠️ Manual Review Recommended: {report.get('compliance', {}).get('manual_review_recommended', 'N/A')}")
print("\n" + "=" * 80)
