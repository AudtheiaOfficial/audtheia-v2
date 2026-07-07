#!/usr/bin/env python3
"""Audtheia per-species reference fetch.

Path: scripts/bootstrap_fetch_species.py

For each species a deployment cares about, this fetches a small, citable set of
reference facts once and stores them locally, so the desktop can name and rank a
detection with no internet at run time:

  - the GBIF taxonomic match (its usage key, accepted scientific name, rank, and
    an English common name where one exists),
  - the GBIF global occurrence count, a worldwide aggregate across many datasets
    and locations that is a reference figure only, never a local abundance, and
  - the IUCN Red List category, when a Red List token is available.

Every stored record carries the date it was fetched and the date of each source
snapshot, so a report can always disclose how current its reference data is. A
species that GBIF cannot match is reported and skipped; nothing is ever stored
under a made-up key. The fetch is done once and cached: a species already on file
is left as it is unless a refresh is asked for.

The species to fetch come from the deployment itself: the union of every
station's target species, which the person running Audtheia sets, plus anything
named on the command line. Audtheia never invents a species list; the expert
running it decides which species matter.

It uses the pinned requests library through a small client seam, so the network
calls are easy to see and the whole fetch is testable without touching the
internet. It runs in the desktop environment created by setup.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# GBIF's open web services. The name match and the occurrence count are public
# and need no account; only GBIF's bulk download service needs a login, and this
# fetch deliberately does not use it, so no GBIF credentials are required.
GBIF_BASE = "https://api.gbif.org/v1"

# The IUCN Red List API version four. A personal token is required and is read
# from the local secrets file; without one the Red List status is left blank and
# clearly reported, rather than the fetch failing. The token is sent as a bearer
# credential, the scheme this version of the API uses.
IUCN_BASE = "https://api.iucnredlist.org/api/v4"

# The controlled set of Red List category codes, used to recognize the category
# in the assessment response wherever the field sits, so a change in the shape of
# the response around it does not silently store the wrong thing.
RED_LIST_CODES = {
    "DD", "LC", "NT", "VU", "EN", "CR", "EW", "EX",
    "LR/lc", "LR/nt", "LR/cd", "NE",
}

# A courteous pause between calls. The Red List service explicitly asks callers
# to space their requests, and it costs a multi-species fetch very little.
DEFAULT_REQUEST_DELAY_SECONDS = 1.0

HTTP_TIMEOUT_SECONDS = 30


def _info(message: str) -> None:
    print(f"    {message}", flush=True)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class FetchError(Exception):
    """A problem that should stop the whole fetch with a clear message."""


# ---------------------------------------------------------------------------
# The HTTP client seam. The real one uses requests; a test injects a client that
# returns canned responses, so the fetch logic is exercised with no network.
# ---------------------------------------------------------------------------


class RequestsClient:
    """A thin wrapper over requests, the one place the network is touched."""

    def __init__(self) -> None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - the environment guarantees it
            raise FetchError(
                "the 'requests' package is not installed; run the desktop setup first."
            ) from exc
        self._requests = requests

    def get_json(self, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None):
        """Return the parsed JSON body, or None when the resource is absent.

        A not-found is returned as None so a missing species is a normal outcome,
        not an error. Any other failure raises, so a real network or credential
        problem is visible rather than silently swallowed.
        """
        response = self._requests.get(
            url, params=params, headers=headers, timeout=HTTP_TIMEOUT_SECONDS
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()


# ---------------------------------------------------------------------------
# GBIF.
# ---------------------------------------------------------------------------


def gbif_match(client, name: str) -> Optional[dict]:
    """Match a scientific name to the GBIF backbone.

    Returns the match when GBIF resolves the name to a usage key, or None when it
    cannot, so a name that does not resolve is skipped rather than stored under a
    guessed key.
    """
    result = client.get_json(f"{GBIF_BASE}/species/match", params={"name": name})
    if not result or result.get("matchType") in (None, "NONE") or not result.get("usageKey"):
        return None
    return result


def gbif_vernacular(client, usage_key: int) -> Optional[str]:
    """An English common name for a taxon, when GBIF has one."""
    result = client.get_json(
        f"{GBIF_BASE}/species/{usage_key}/vernacularNames", params={"limit": 100}
    )
    if not result:
        return None
    names = result.get("results", [])
    for entry in names:
        if entry.get("language") == "eng" and entry.get("vernacularName"):
            return entry["vernacularName"]
    for entry in names:
        if entry.get("vernacularName"):
            return entry["vernacularName"]
    return None


def gbif_occurrence_count(client, usage_key: int) -> Optional[int]:
    """GBIF's worldwide occurrence count for a taxon.

    This is a global aggregate across every dataset GBIF holds and every place a
    record came from. It is a reference figure only and must never be read as a
    local abundance; local rarity is a separate, effort-normalized statistic
    computed from a station's own record.
    """
    result = client.get_json(f"{GBIF_BASE}/occurrence/count", params={"taxonKey": usage_key})
    if result is None:
        return None
    try:
        return int(result)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# IUCN Red List.
# ---------------------------------------------------------------------------


def _extract_category_code(payload) -> Optional[str]:
    """Find a Red List category code anywhere reasonable in an assessment body.

    The category is read defensively: whichever field carries it, only a value in
    the controlled set of Red List codes is accepted, so a change around it never
    stores a wrong or invented status.
    """
    if not isinstance(payload, dict):
        return None
    candidates = []
    rlc = payload.get("red_list_category")
    if isinstance(rlc, dict):
        candidates += [rlc.get("code"), rlc.get("category")]
    elif isinstance(rlc, str):
        candidates.append(rlc)
    candidates += [payload.get("red_list_category_code"), payload.get("category"), payload.get("code")]
    for value in candidates:
        if isinstance(value, str) and value.strip() in RED_LIST_CODES:
            return value.strip()
    return None


def iucn_status(client, name: str, token: Optional[str]) -> Optional[str]:
    """The latest IUCN Red List category for a species, when a token is set.

    With no token this returns None so the fetch continues with the status left
    blank. With a token it looks the species up, takes its latest assessment, and
    reads the category from that assessment, returning None on any absence rather
    than guessing.
    """
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    taxon = client.get_json(f"{IUCN_BASE}/taxa/scientific_name/{name}", headers=headers)
    if not isinstance(taxon, dict):
        return None
    assessments = taxon.get("assessments") or []
    if not assessments:
        return None

    latest = next((a for a in assessments if a.get("latest")), assessments[0])
    # Some responses carry the code on the summary already; prefer it when valid.
    summary_code = _extract_category_code(latest)
    if summary_code:
        return summary_code

    assessment_id = latest.get("assessment_id") or latest.get("id")
    if assessment_id is None:
        return None
    detail = client.get_json(f"{IUCN_BASE}/assessment/{assessment_id}", headers=headers)
    return _extract_category_code(detail)


# ---------------------------------------------------------------------------
# One species.
# ---------------------------------------------------------------------------


def fetch_one(client, name: str, token: Optional[str]):
    """Fetch the reference facts for one species, or None when GBIF cannot match it."""
    from audtheia.storage.database import SpeciesReference, utc_now_iso

    match = gbif_match(client, name)
    if match is None:
        return None

    usage_key = match["usageKey"]
    scientific_name = match.get("canonicalName") or match.get("scientificName") or name
    rank = match.get("rank")
    match_type = match.get("matchType")
    confidence = match.get("confidence")

    common_name = gbif_vernacular(client, usage_key)
    count = gbif_occurrence_count(client, usage_key)
    status = iucn_status(client, scientific_name, token)

    today = _today()
    record = SpeciesReference(
        gbif_usage_key=str(usage_key),
        scientific_name=scientific_name,
        fetched_at=utc_now_iso(),
        common_name=common_name,
        taxonomic_rank=rank,
        iucn_status=status,
        iucn_fetch_date=today if status is not None else None,
        gbif_occurrence_count=count,
        gbif_snapshot_date=today if count is not None else None,
    )
    return record, {"match_type": match_type, "confidence": confidence}


# ---------------------------------------------------------------------------
# The species list.
# ---------------------------------------------------------------------------


def resolve_species(settings, args) -> list[str]:
    """The names to fetch: the deployment's target species plus any given by hand.

    Names are gathered from the stations the deployment already defines (all of
    them, or one when a station is named) and from the command line, then
    deduplicated while keeping their order. Audtheia never adds a species the
    deployment did not ask for.
    """
    names: list[str] = []

    stations = settings.stations()
    if args.station_id:
        stations = [settings.station(args.station_id)]
    for station in stations:
        for name in station.get("target_species", []) or []:
            if isinstance(name, str) and name.strip():
                names.append(name.strip())

    for name in args.species or []:
        if name.strip():
            names.append(name.strip())

    if args.from_file:
        path = Path(args.from_file)
        if not path.exists():
            raise FetchError(f"species file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                names.append(text)

    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(name)
    return ordered


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------


def run(
    settings=None,
    *,
    client=None,
    args=None,
    delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
) -> dict:
    """Fetch and store reference data for the resolved species list."""
    from audtheia.config import load_settings
    from audtheia.storage.database import Database

    if settings is None:
        settings = load_settings()
    if client is None:
        client = RequestsClient()

    token = settings.secrets.get("iucn_api_key") or None
    if not token:
        _info("no IUCN Red List token set; conservation status will be left blank.")

    db = Database(settings.db_path(), **settings.database_kwargs())

    names = resolve_species(settings, args)
    if not names:
        _info("no target species are configured. Add species to a station's "
              "target_species, or pass --species or --from-file.")
        return {"fetched": 0, "cached": 0, "unmatched": 0, "failed": 0}

    outcome = {"fetched": 0, "cached": 0, "unmatched": 0, "failed": 0}
    _info(f"{len(names)} species to consider.")

    for name in names:
        try:
            existing = _cached_key(db, client, name)
            if existing is not None and not args.refresh:
                _info(f"{name}: already on file (key {existing}); skipping. Use --refresh to update.")
                outcome["cached"] += 1
                continue

            result = fetch_one(client, name, token)
            if result is None:
                _info(f"{name}: GBIF could not match this name; skipped. Check the spelling or add it by hand.")
                outcome["unmatched"] += 1
                continue

            record, meta = result
            db.upsert_species_reference(record)
            note = f"key {record.gbif_usage_key}, {record.taxonomic_rank}"
            if meta["match_type"] and meta["match_type"] != "EXACT":
                note += f", {meta['match_type']} match confidence {meta['confidence']}, please verify"
            if record.iucn_status:
                note += f", IUCN {record.iucn_status}"
            if record.gbif_occurrence_count is not None:
                note += f", GBIF global occurrences {record.gbif_occurrence_count}"
            _info(f"{name}: stored ({note}).")
            outcome["fetched"] += 1
        except Exception as exc:  # noqa: BLE001 - one species must never abort the batch
            _info(f"{name}: fetch failed ({exc}); left unchanged, re-run to retry.")
            outcome["failed"] += 1
        finally:
            if delay_seconds:
                time.sleep(delay_seconds)

    return outcome


def _cached_key(db, client, name: str) -> Optional[str]:
    """The stored usage key for a name, if it is already on file.

    A match is needed to know the key a name resolves to, but the match is cheap
    and lets the cache be keyed by the stable GBIF key rather than by raw text.
    """
    match = gbif_match(client, name)
    if match is None:
        return None
    key = str(match["usageKey"])
    return key if db.get_species_reference(key) is not None else None


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch-species-data",
        description=(
            "Fetch each target species' GBIF taxonomy and occurrence count and its "
            "IUCN Red List status, once, and store it locally for offline use."
        ),
    )
    parser.add_argument(
        "--species", action="append", default=[],
        help="A scientific name to fetch. May be given more than once.",
    )
    parser.add_argument(
        "--from-file", default=None,
        help="A file of scientific names, one per line (lines starting with # are ignored).",
    )
    parser.add_argument(
        "--station-id", default=None,
        help="Fetch only one station's target species instead of every station's.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch and update species already on file.",
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Seconds to wait between species, to be polite to the services.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    print("Audtheia species reference fetch")
    try:
        outcome = run(args=args, delay_seconds=args.delay_seconds)
        print(
            f"\nDone. Stored {outcome['fetched']}, already on file {outcome['cached']}, "
            f"unmatched {outcome['unmatched']}, failed {outcome['failed']}."
        )
        return 0
    except FetchError as exc:
        print(f"\nFetch stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nFetch interrupted. It is safe to run again.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
