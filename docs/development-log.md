# Development log

Record meaningful attempts as they occur, including failures. Identify contributors by GitHub handle or member shorthand. Link each entry to a commit, PR, run record, or supporting file.

Use the following format for each entry:

### YYYY-MM-DD - Attempt title

- Contributor:
- Attempt:
- Outcome:
- Lesson and next action:
- Evidence:

### 2026-11-22 - Component C first runnable implementation

- Contributor: xiangyi
- Attempt: Implement adult reference targets, correction-safe local intake records, Chicago-date summaries, and dated Gordon Nutrislice menu normalization/cache.
- Outcome: Added a standard-library implementation and deterministic tests covering official formula coefficients, editable targets, consumed fractions, corrections, incomplete logs, date rollover, meal-specific caching, and nutrient validation.
- Lesson and next action: Live Nutrislice availability is external and should be verified separately before integration; connect Component B output and Component D's planner against the documented dictionaries next.
- Evidence: `tests/test_daily_tracking.py`; command `python -m unittest tests.test_daily_tracking -v`.

### 2026-11-22 - Component C/D interface integration

- Contributor: xiangyi
- Attempt: Merge the current C and D branches from `origin/main`, then pass C's normalized menu and daily summary directly into D.
- Outcome: The initial review found a `nutrients`/`nutrition` field mismatch. Added a narrow adapter and integration tests without changing D's ranking algorithm.
- Lesson and next action: Maintain the C output dictionaries as the integration contract and run the same test whenever either component changes.
- Evidence: `tests/test_cd_integration.py`; command `python -m unittest discover -s tests -v`.
