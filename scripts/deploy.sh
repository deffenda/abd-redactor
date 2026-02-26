#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="${1:-pdf-redaction-pipeline}"
REGION="${AWS_REGION:-us-east-1}"
INPUT_PREFIX="${INPUT_PREFIX:-incoming/}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-redacted/}"
MIN_ENTITY_SCORE="${MIN_ENTITY_SCORE:-0.8}"
COMPREHEND_LANGUAGE="${COMPREHEND_LANGUAGE:-en}"

sam build --use-container
sam deploy \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides \
    InputPrefix="${INPUT_PREFIX}" \
    OutputPrefix="${OUTPUT_PREFIX}" \
    MinEntityScore="${MIN_ENTITY_SCORE}" \
    ComprehendLanguage="${COMPREHEND_LANGUAGE}"
