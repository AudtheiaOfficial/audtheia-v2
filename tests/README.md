# `tests/`

Unit, integration, and mocked-hardware smoke tests (added to the build plan as its own top-level folder, alongside the canonical repository structure in the Master Concept).

Sessions 1–18 are designed to build and verify entirely on mocked hardware — no Pi 5 / AI HAT+ 2 / camera / hydrophone required until ~Session 19. Each session's own verification work (e.g. this session's `schema.sql` good/bad-insert tests, run 5× for determinism) graduates into this folder as the corresponding module is built, rather than living only in chat history.

Nothing here yet — populated starting with Session 2 (`database.py` unit tests).
