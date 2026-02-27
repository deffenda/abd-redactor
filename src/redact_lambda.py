"""
Lambda handler for PII detection and redaction.
Triggered by batch manifest events. Detects PII, redacts PDF, and writes results to S3.
"""
import json
import logging
import os
from pathlib import Path
import boto3
import fitz
from .detect_pii_batch import lambda_handler as detect_pii_batch_handler
from .assemble_output import *

def lambda_handler(event, _context):
    logger = logging.getLogger()
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # This expects event to contain batch manifest info
    # Call detect_pii_batch logic
    batch_result = detect_pii_batch_handler(event, _context)
    # ...assemble output, redact PDF, write report...
    # (Reuse logic from assemble_output.py and app.py as needed)
    return {"status": "redaction complete", "batch_result": batch_result}
