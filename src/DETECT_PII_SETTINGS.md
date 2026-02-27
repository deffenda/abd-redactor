# DetectPiiEntities Settings Reference

This file centralizes all settings for AWS Comprehend's DetectPiiEntities API usage in this project. Edit `src/detect_pii_settings.py` to customize detection globally.

## Settings

- **MIN_ENTITY_SCORE**: Minimum confidence score for detected PII entities to be considered valid. (Default: 0.8)
- **COMPREHEND_LANGUAGE**: Language code for Comprehend detection (e.g., 'en' for English). (Default: 'en')
- **MAX_COMPREHEND_TEXT_LEN**: Maximum number of characters per text chunk sent to Comprehend. (Default: 4500)

## Usage

Import these settings in your modules:

```python
from .detect_pii_settings import MIN_ENTITY_SCORE, COMPREHEND_LANGUAGE, MAX_COMPREHEND_TEXT_LEN
```

All modules should use these settings for consistent and easily customizable PII detection behavior.
