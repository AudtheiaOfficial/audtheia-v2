# `config/`

| File | Builds in | Role |
|---|---|---|
| `settings.json` | Session 3 | Every configurable value: stations, sensors, model paths, schedules, credentials, capture tuning, analysis-location toggle, embedding-forwarding toggle. No species names, model names, API keys, or paths are ever hardcoded elsewhere in the codebase — they all live here. |
| `config/README.md` (full version) | Session 3 | Every setting documented with examples. |

**Contracts-lock gate:** after Session 3, `schema.sql` and `settings.json` are frozen — later sessions conform to them rather than triggering schema rework.

This is a placeholder stub for the pre-Session-3 repository skeleton.
