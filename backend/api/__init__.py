"""
API orchestration boundary for Evidentia backend.

Design intent:
- Expose thin, typed endpoints for transcript/event/risk/note flows.
- Keep request validation explicit and failure modes predictable.
- Orchestrate modules without embedding domain logic in routers.
"""
