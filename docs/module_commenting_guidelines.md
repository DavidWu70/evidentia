# Module Commenting Guidelines (MVP)

Use comments to explain **design intent** and **safety constraints**, not obvious syntax.

## Required style for core modules

For each core module (`asr/events/risk/note/api`), add a short top-level docstring that explains:

1. Why this module exists
2. What contract it enforces
3. What failure mode it prevents

## Example (good)

```python
"""
Extract minimal evidence-backed clinical events.
Each event must contain:
- segment_id
- t0/t1
- quote (verbatim)
This ensures traceability and prevents hallucinated summaries.
"""
```

## Example (bad)

```python
# Set variable x
x = 1
```

## Rule of thumb

If a future reader asks "why was this designed this way?", comments should answer that.
If comments only restate the code, delete them.
