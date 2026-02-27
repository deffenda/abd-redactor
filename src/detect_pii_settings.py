"""
Central configuration for DetectPiiEntities settings.
Edit this file to customize detection behavior globally.
"""
import os

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Main settings for DetectPiiEntities
MIN_ENTITY_SCORE = _env_float("MIN_ENTITY_SCORE", 0.8)
COMPREHEND_LANGUAGE = os.getenv("COMPREHEND_LANGUAGE", "en")
MAX_COMPREHEND_TEXT_LEN = int(os.getenv("MAX_COMPREHEND_TEXT_LEN", "4500"))

# API selection: 'detect' (default) or 'start_job'
PII_DETECTION_API = os.getenv("PII_DETECTION_API", "detect")  # 'detect' or 'start_job'

# Add more settings as needed
