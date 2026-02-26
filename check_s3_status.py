#!/usr/bin/env python3
"""Check S3 bucket for redacted files."""

import boto3

s3_client = boto3.client('s3', region_name='us-east-1')
bucket = 'abd-redactor-20260226135602-8965'

print("Checking S3 bucket for redacted files...\n")
print(f"Bucket: {bucket}\n")

# Check incoming folder
print("incoming/ folder:")
response = s3_client.list_objects_v2(Bucket=bucket, Prefix='incoming/')
if 'Contents' in response:
    for obj in response['Contents']:
        print(f"  - {obj['Key']} ({obj['Size']} bytes)")
else:
    print("  (empty)")

# Check redacted folder
print("\nredacted/ folder:")
response = s3_client.list_objects_v2(Bucket=bucket, Prefix='redacted/')
if 'Contents' in response:
    for obj in response['Contents']:
        size_kb = obj['Size'] / 1024
        print(f"  - {obj['Key']} ({size_kb:.1f} KB)")
        print(f"    S3 URI: s3://{bucket}/{obj['Key']}")
else:
    print("  (empty)")
    print("\n  The Lambda function hasn't processed the file yet.")
    print("  It's triggered by S3 events when files are uploaded.")
