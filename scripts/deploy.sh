#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="${1:-adb-document-manager}"
REGION="${AWS_REGION:-us-east-1}"
INPUT_PREFIX="${INPUT_PREFIX:-incoming/}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-redacted/}"
REPORT_PREFIX="${REPORT_PREFIX:-reports/}"
WORK_PREFIX="${WORK_PREFIX:-work/}"
MIN_ENTITY_SCORE="${MIN_ENTITY_SCORE:-0.8}"
COMPREHEND_LANGUAGE="${COMPREHEND_LANGUAGE:-en}"
CHUNK_CHAR_LIMIT="${CHUNK_CHAR_LIMIT:-4500}"
CHUNKS_PER_BATCH="${CHUNKS_PER_BATCH:-20}"
MAP_MAX_CONCURRENCY="${MAP_MAX_CONCURRENCY:-5}"

sam build --use-container
sam deploy \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides \
    InputPrefix="${INPUT_PREFIX}" \
    OutputPrefix="${OUTPUT_PREFIX}" \
    ReportPrefix="${REPORT_PREFIX}" \
    WorkPrefix="${WORK_PREFIX}" \
    MinEntityScore="${MIN_ENTITY_SCORE}" \
    ComprehendLanguage="${COMPREHEND_LANGUAGE}" \
    ChunkCharLimit="${CHUNK_CHAR_LIMIT}" \
    ChunksPerBatch="${CHUNKS_PER_BATCH}" \
    MapMaxConcurrency="${MAP_MAX_CONCURRENCY}"
