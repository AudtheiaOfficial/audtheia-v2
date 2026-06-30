# audtheia/analysis

Quality control, verification, and longitudinal pattern discovery. These modules
take the raw observations a field station records and turn them into a verified,
interpreted, research-grade record.

| File | Runs on | Role |
|------|---------|------|
| observation.py | Field station (Pi) | A deterministic predict, compare, correct quality-control engine, not a language model. It validates each observation, completes it with controlled-vocabulary status values, and consolidates it. It runs only the field-tier skills and routes anything structurally novel for later review. |
| verify.py | Desktop hub | Re-verifies each detection with the high-accuracy RF-DETR model through ONNX Runtime, adds the ecological interpretation (role, rarity, cross-modal attribution) clearly labeled as inferred, computes the authoritative salience, and sets the verified flag. |
| dream.py | Desktop hub | Runs the longitudinal pattern-discovery pass over the verified record and writes candidate hypotheses, each with an effect size and the span of data behind it. |

The separation between what was measured and what was inferred is structural: the
field-station quality-control step never writes to a desktop-owned table, so
measured fact and downstream interpretation can never be confused.
