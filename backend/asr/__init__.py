"""
ASR module boundary for Evidentia backend.

Design intent:
- Centralize streaming/chunked transcription adapters.
- Keep provider-specific complexity out of API handlers.
- Return timestamped segments for downstream evidence tracing.
"""
