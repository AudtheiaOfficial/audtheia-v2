# audtheia/storage

The database contract and the code that reads and writes it. Every other part of
the system conforms to the schema defined here.

| File | Role |
|------|------|
| schema.sql | Defines every table in the system: globally unique identifiers, the station and event fields, the source and quality-control status carried by every value, the provisional and authoritative salience slots, full model and data version provenance, coordinated-universal-time timestamps, the field-to-desktop sync marker, the separation between field-owned and desktop-owned tables, and the station telemetry and pattern-discovery tables. |
| database.py | All read and write functions, plus the checkpointed, append-only synchronization that carries records from a field station up to the desktop hub without ever duplicating one. |

The same schema and access layer run on both the field station and the desktop
hub. The schema is the contract the rest of the repository depends on; its header
comment documents each table in detail.

Local-only files, never committed (see .gitignore): database/audtheia.db.
