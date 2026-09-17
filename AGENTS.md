# Collaboration preferences

Jacob welcomes suggestions for better implementations, even when he proposes a
specific approach. Treat proposed implementation details as open to discussion
unless he explicitly makes them a requirement.

- Before implementing, flag materially better alternatives and briefly explain
  their practical tradeoffs: simplicity, performance, reliability, and maintenance.
- Make a clear recommendation rather than silently following a less suitable
  approach. Focus on meaningful choices, not exhaustive alternatives or minor nits.
- Preserve the user's underlying goal and respect explicit decisions. This
  preference does not authorize unrelated changes or overriding requirements.
- Example: for a consolidated, append-only training event log, recommend JSON
  Lines (`.jsonl`) over rewriting a single JSON document on every event. Explain
  append efficiency, machine readability, and interrupted-write recovery before
  implementing the format.

This is a collaboration preference, not a claim that the example migration has
already been implemented.

## Exploratory-phase compatibility policy

Until serious training work has begun, Jacob does not require backward
compatibility. Prioritize clarity, simplicity, and unambiguous behavior over
preserving old interfaces, configuration names, file formats, or checkpoints.

- Prefer a clean replacement over compatibility aliases, legacy fallbacks,
  migration machinery, or support for multiple historical formats.
- Clearly report breaking changes and reject obsolete inputs with useful errors
  rather than silently guessing their meaning.
- This policy does not authorize deleting existing data or training artifacts.
- Revisit compatibility requirements when serious training begins; do not assume
  this exploratory-phase policy applies indefinitely.
