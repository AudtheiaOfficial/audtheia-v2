"""What a stored verification verdict means, said the same way everywhere.

Path: audtheia/analysis/verdicts.py

Two different gates write to observation_verification, and they are not
equivalent:

  The two-model check, for an event with frames. The desktop verifier re-scores
  the saved frames and the event clears only when it recognises the same taxon as
  the field call, confidently enough.

  The acoustic confidence gate, for a pure sound event. There is no frame for a
  second model to re-score, so the event clears when its strongest call meets a
  configured floor. It is honest and documented, but weaker: no second model ever
  checked it.

A third case is not a verdict at all. When no desktop verification model was
configured, earlier versions recorded a stand-in result in which every frame was
"scored" and nothing was recognised, with no verifier version. That reads like a
model that looked and saw nothing, which is false: no model looked. Such a row is
recognised here and shown as "did not run".

Every reader, the report, its CSV, the interface and the audit, classifies a
verdict through this module, so the two gates are never pooled into one number
and a missing check is never presented as a failed one. Nothing here writes.
"""

from __future__ import annotations

from typing import Iterable, Optional

__all__ = [
    "GATE_TWO_MODEL",
    "GATE_ACOUSTIC",
    "is_stand_in",
    "classify",
    "split_counts",
    "describe",
    "summary_sentence",
    "floors",
]

GATE_TWO_MODEL = "two_model_check"
GATE_ACOUSTIC = "acoustic_confidence_gate"

# States. A visual event is verified, not verified, not run (no model checked it)
# or pending (no verdict yet). A sound event is cleared or not cleared by its gate,
# or pending.
VERIFIED = "verified"
NOT_VERIFIED = "not_verified"
NOT_RUN = "not_run"
PENDING = "pending"
CLEARED = "cleared"
NOT_CLEARED = "not_cleared"


def floors(settings) -> dict:
    """The clearing floors in force, from the settings."""
    try:
        v = settings.thresholds_config()["verification"]
    except Exception:  # noqa: BLE001 - a settings object without thresholds uses the documented defaults
        v = {}
    return {"clear_confidence": _float(v.get("clear_confidence"), 0.5),
            "acoustic_clear_confidence": _float(v.get("acoustic_clear_confidence"), 0.7)}


def _float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_stand_in(observation: dict, verification: Optional[dict]) -> bool:
    """Whether a verdict was recorded while no verification model was configured.

    Recognised by its shape: an event with frames, no verifier version, nothing
    resolved, no confidence and nothing cleared. A real verifier always stamps a
    version, so from now on no real verdict has this shape.
    """
    if not verification or str(observation.get("trigger_source")) == "audio":
        return False
    return (not verification.get("verified")
            and not verification.get("rfdetr_version")
            and not verification.get("rfdetr_scientific_name")
            and not verification.get("rfdetr_gbif_usage_key")
            and verification.get("rfdetr_confidence") is None
            and not verification.get("frames_in_agreement"))


def classify(observation: dict, verification: Optional[dict]) -> dict:
    """The gate and the state of one event's verification.

    Returns gate (two_model_check or acoustic_confidence_gate), state, and for a
    visual verdict that did not clear, reason: recognized_none, disagrees,
    below_floor or no_field_call.
    """
    audio = str(observation.get("trigger_source")) == "audio"
    gate = GATE_ACOUSTIC if audio else GATE_TWO_MODEL
    if not verification:
        return {"gate": gate, "state": PENDING, "reason": None}
    if audio:
        return {"gate": gate, "state": CLEARED if verification.get("verified") else NOT_CLEARED, "reason": None}
    if is_stand_in(observation, verification):
        return {"gate": gate, "state": NOT_RUN, "reason": "no_model"}
    if verification.get("verified"):
        return {"gate": gate, "state": VERIFIED, "reason": None}
    resolved = verification.get("rfdetr_scientific_name") or verification.get("rfdetr_gbif_usage_key")
    if not resolved:
        reason = "recognized_none"
    elif verification.get("rfdetr_agrees_with_field") == 0:
        reason = "disagrees"
    elif verification.get("rfdetr_agrees_with_field") is None:
        reason = "no_field_call"
    else:
        reason = "below_floor"
    return {"gate": gate, "state": NOT_VERIFIED, "reason": reason}


def split_counts(pairs: Iterable[tuple]) -> dict:
    """Verification counted separately per gate, from (observation, verification) pairs.

    visual: events with frames and the two-model check; audio: pure sound events
    and the acoustic confidence gate. The two are never added together.
    """
    out = {
        "visual": {"total": 0, VERIFIED: 0, NOT_VERIFIED: 0, NOT_RUN: 0, PENDING: 0},
        "audio": {"total": 0, CLEARED: 0, NOT_CLEARED: 0, PENDING: 0},
    }
    for observation, verification in pairs:
        c = classify(observation, verification)
        bucket = out["audio"] if c["gate"] == GATE_ACOUSTIC else out["visual"]
        bucket["total"] += 1
        bucket[c["state"]] += 1
    return out


def summary_sentence(split: dict) -> str:
    """One line that states both gates separately, for summaries and tables."""
    v = split.get("visual") or {}
    a = split.get("audio") or {}
    parts = []
    if v.get("total"):
        text = (f"{v.get(VERIFIED, 0)} of {v['total']} visual "
                f"{'event' if v['total'] == 1 else 'events'} by the two-model check")
        unchecked = v.get(NOT_RUN, 0) + v.get(PENDING, 0)
        if unchecked:
            text += f" ({unchecked} not checked by a second model)"
        parts.append(text)
    if a.get("total"):
        parts.append(f"{a.get(CLEARED, 0)} of {a['total']} audio "
                     f"{'event' if a['total'] == 1 else 'events'} by the acoustic confidence gate")
    return "; ".join(parts) if parts else "no events in scope"


def _num(value, digits: int = 2) -> str:
    if value is None:
        return "not recorded"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def describe(observation: dict, verification: Optional[dict], *, children: Iterable[dict] = (),
             floors_in_force: Optional[dict] = None, verifier_configured: Optional[bool] = None,
             verifier_version: Optional[str] = None, verifier_identities: Optional[set] = None,
             book=None) -> str:
    """One plain sentence for an event's verification, naming its gate.

    verifier_configured and verifier_version describe the desktop verifier in force
    now; verifier_identities, when known, is the set of identity keys its labels
    resolve to, used to say when the field call's taxon is not among them. book is a
    taxa.TaxonBook used to name taxa by their accepted identity.
    """
    f = floors_in_force or {"clear_confidence": 0.5, "acoustic_clear_confidence": 0.7}
    children = list(children or ())
    c = classify(observation, verification)

    if c["gate"] == GATE_ACOUSTIC:
        calls = [float(ch["confidence"]) for ch in children
                 if ch.get("modality") == "audio" and ch.get("confidence") is not None]
        strongest = max(calls) if calls else None
        floor = _num(f["acoustic_clear_confidence"])
        if c["state"] == PENDING:
            return "not yet checked by the acoustic confidence gate"
        if c["state"] == CLEARED:
            return (f"cleared by the acoustic confidence gate: strongest call {_num(strongest, 3)} against a "
                    f"floor of {floor}; not cross-checked by a second model")
        return (f"not cleared by the acoustic confidence gate: strongest call {_num(strongest, 3)} is below the "
                f"floor of {floor}; not cross-checked by a second model")

    floor = _num(f["clear_confidence"])
    if c["state"] == PENDING:
        if verifier_configured is False:
            return "desktop verification not run: no verification model is configured"
        return "awaiting desktop verification"
    if c["state"] == NOT_RUN:
        return ("desktop verification did not run: no verification model was configured when this event was "
                "processed, so no frame was checked by a second model")

    v = verification or {}
    who = v.get("rfdetr_version") or "the verifier (version not recorded)"
    scored = v.get("frames_scored") or 0
    agreeing = v.get("frames_in_agreement") or 0
    resolved = v.get("rfdetr_scientific_name") or v.get("rfdetr_gbif_usage_key")
    taxon = resolved
    if book is not None and resolved:
        taxon = book.display(book.identify_label(v.get("rfdetr_scientific_name"), v.get("rfdetr_gbif_usage_key")))
    frames = f"{agreeing} of {scored} {'frame' if scored == 1 else 'frames'}"
    # A verifier that is the screening model itself is not an independent check,
    # and the wording says so rather than calling it a second model.
    same = ""
    if v.get("rfdetr_version") and v.get("rfdetr_version") == observation.get("screening_model_version"):
        same = "; the verifier is the same model that screened the event, so this is not an independent check"

    if c["state"] == VERIFIED:
        return (f"verified by the two-model check: {who} recognized {taxon} in {frames}, mean confidence "
                f"{_num(v.get('rfdetr_confidence'))}, at or above the floor of {floor}, agreeing with the field call"
                f"{same}")

    reason = c["reason"]
    if reason == "recognized_none":
        text = (f"not verified: {who} scored {scored} {'frame' if scored == 1 else 'frames'} "
                f"and recognized none of them")
        field_ident = _field_identity(children, book)
        if (field_ident is not None and verifier_identities is not None and verifier_version
                and v.get("rfdetr_version") == verifier_version and field_ident["key"] not in verifier_identities):
            field_name = book.display(field_ident) if book is not None else field_ident.get("label")
            text += f"; its label set does not include the field call's taxon, {field_name}"
        return text + same
    if reason == "disagrees":
        field_ident = _field_identity(children, book)
        field_name = (book.display(field_ident) if (book is not None and field_ident) else
                      (field_ident or {}).get("label") or "the field call")
        return (f"not verified: {who} recognized {taxon} in {frames}, which differs from the field call, "
                f"{field_name}{same}")
    if reason == "no_field_call":
        # No agreement was stored with this verdict. When the record does hold a
        # field call, the two are compared here by identity and the comparison is
        # stated as made at reading time; the stored verdict is not changed.
        field_ident = _field_identity(children, book)
        if field_ident is not None:
            verifier_ident = (book.identify_label(v.get("rfdetr_scientific_name"), v.get("rfdetr_gbif_usage_key"))
                              if book is not None else
                              {"key": str(v.get("rfdetr_scientific_name") or v.get("rfdetr_gbif_usage_key"))})
            field_name = (book.display(field_ident) if book is not None else field_ident.get("label"))
            if verifier_ident is not None and str(verifier_ident.get("key")).lower() == str(field_ident["key"]).lower():
                return (f"not verified: {who} recognized {taxon} in {frames}, the same taxon as the field call "
                        f"(compared when read; no agreement was stored with the verdict), but its mean confidence "
                        f"{_num(v.get('rfdetr_confidence'))} is below the floor of {floor}{same}")
            return (f"not verified: {who} recognized {taxon} in {frames}, which differs from the field call, "
                    f"{field_name} (compared when read; no agreement was stored with the verdict){same}")
        return (f"not verified: {who} recognized {taxon} in {frames}, mean confidence "
                f"{_num(v.get('rfdetr_confidence'))}, but there is no field call to compare it with{same}")
    return (f"not verified: {who} recognized {taxon} in {frames}, agreeing with the field call, but its mean "
            f"confidence {_num(v.get('rfdetr_confidence'))} is below the floor of {floor}{same}")


def _field_identity(children: list, book) -> Optional[dict]:
    """The identity of the field call: the strongest visual detection."""
    best = None
    for ch in children:
        if ch.get("modality") != "vision" or ch.get("confidence") is None:
            continue
        if best is None or ch["confidence"] > best["confidence"]:
            best = ch
    if best is None:
        return None
    if book is None:
        from audtheia.analysis.taxa import raw_label

        label = raw_label(best)
        return {"key": label, "label": label} if label else None
    return book.identify(best)
