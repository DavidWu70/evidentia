"""
Event extraction boundary for Evidentia backend.

Design intent:
- Extract minimal evidence-backed clinical events.
- Enforce segment_id/t0/t1/quote on each event payload.
- Prevent non-traceable summaries from reaching risk/note layers.
"""
