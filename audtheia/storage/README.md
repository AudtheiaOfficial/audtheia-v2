# `audtheia/storage/` — Database contract and access layer

| File | Builds in | Role |
|---|---|---|
| `schema.sql` | **Session 1 (this session)** | Every table in the system. UUID PKs, `station_id`, `trigger_source` + event fields, `data_source`, missing-data/QC status, `qc_state`, salience (provisional + authoritative slots), full model/data version provenance, UTC timestamps, `synced_at`, desktop-owned ownership separation, `station_telemetry`, `dream_passes` / `patterns` / `pattern_observations`. |
| `database.py` | Session 2 | All read/write functions; checkpointed, UUID-idempotent append-only Pi→desktop sync. |

`schema.sql` is the contract every other file in the repository conforms to — see its header comment for the full decision-by-decision mapping, and `audtheia-v2-decisions-log.md` #9–#52.

**Local-only files, never committed** (see `.gitignore`): `database/audtheia.db`.
