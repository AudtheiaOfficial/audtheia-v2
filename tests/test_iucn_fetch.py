"""The IUCN Red List v4 lookup uses the right endpoint shape and parses a status.

Path: tests/test_iucn_fetch.py

The v4 taxa endpoint takes the genus and species as separate query parameters,
not the binomial as one path segment, and the assessment detail lives at the
plural /assessments/{id} path. An earlier form put the full name in the path and
used the singular detail path, so every lookup returned nothing and the
conservation status stayed blank. This proves the corrected request shape and the
defensive parsing against a scripted client, with no token and no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bootstrap_setup_pi import SetupPiError  # noqa: E402,F401  (ensure scripts importable)
from scripts.bootstrap_fetch_species import iucn_status, IUCN_BASE  # noqa: E402


CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


class StubClient:
    """Returns a canned JSON body per URL and records how it was called."""

    def __init__(self, by_url):
        self._by_url = by_url
        self.calls = []

    def get_json(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        return self._by_url.get(url)


def test_summary_code_and_query_params():
    print("\n[1] The taxa lookup uses genus_name/species_name and reads the summary code")
    taxa_url = f"{IUCN_BASE}/taxa/scientific_name"
    client = StubClient({
        taxa_url: {"assessments": [{"latest": True, "red_list_category_code": "LC", "assessment_id": 1}]}
    })
    status = iucn_status(client, "Cyanocitta cristata", token="fake-token")
    check("the status is read from the summary", status == "LC")
    first = client.calls[0]
    check("the taxa endpoint is the query form, not a path segment", first["url"] == taxa_url)
    check("the genus and species are sent as separate params",
          first["params"].get("genus_name") == "Cyanocitta" and first["params"].get("species_name") == "cristata")
    check("the token is sent as a bearer header", "Bearer fake-token" in str(first["headers"].get("Authorization")))


def test_falls_back_to_plural_assessment_detail():
    print("\n[2] With no summary code, it reads the plural /assessments/{id} detail")
    taxa_url = f"{IUCN_BASE}/taxa/scientific_name"
    detail_url = f"{IUCN_BASE}/assessments/456"
    client = StubClient({
        taxa_url: {"assessments": [{"latest": True, "assessment_id": 456}]},
        detail_url: {"red_list_category": {"code": "EN"}},
    })
    status = iucn_status(client, "Panthera tigris", token="fake-token")
    check("the status comes from the assessment detail", status == "EN")
    check("the detail path is plural /assessments/{id}", any(c["url"] == detail_url for c in client.calls))


def test_genus_only_and_no_token():
    print("\n[3] A genus-only name and a missing token both yield no status, no call")
    client = StubClient({})
    check("a one-word (genus) name returns None", iucn_status(client, "Aplysina", token="fake-token") is None)
    check("a genus-only name makes no request", client.calls == [])
    client2 = StubClient({})
    check("no token returns None", iucn_status(client2, "Cyanocitta cristata", token=None) is None)
    check("no token makes no request", client2.calls == [])


def main() -> int:
    test_summary_code_and_query_params()
    test_falls_back_to_plural_assessment_detail()
    test_genus_only_and_no_token()
    print(f"\n==== IUCN fetch: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
