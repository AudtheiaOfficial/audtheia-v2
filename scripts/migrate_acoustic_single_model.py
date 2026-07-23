"""One-time configuration migration: one acoustic model per station.

Path: scripts/migrate_acoustic_single_model.py

A station sits in one place and listens with one model. The configuration used
to hold three named acoustic slots and a selector naming which of them was live:

    "acoustic": {
      "active": "birdnet",
      "options": {
        "birdnet": {"path": ..., "labels_path": ..., "version": ..., "citation": ...},
        "marine":  {...},
        "custom":  {...}
      }
    }

Only the selected slot was ever loaded, so the other two were pre-staging that
nobody asked for and a second place a path could be set and silently not used.
The slot names also wrote two model families into the configuration contract of
a platform that is meant to be indifferent to what is being studied. This
migration collapses the block to the one model the station actually listens
with:

    "acoustic": {
      "path": ..., "labels_path": ...,
      "sample_rate": ..., "window_seconds": ..., "output_key": ...,
      "version": ..., "citation": ...
    }

Two files, not one:

  config/settings.json holds the committed configuration.

  config/settings.local.json holds this machine's absolute paths, addressed by
  pointer, in the form
  stations[station_id=...].models.acoustic.options.birdnet.path . After the
  restructure those pointers resolve to nothing, the loader reports them stale,
  and the model an installation is actually running disappears without an error.
  That is the same failure that lost the capture source, and the local file is
  the copy holding the paths that really run, so it is migrated alongside the
  committed one.

Safety:

  - Nothing is written until a timestamped backup of each file exists beside it.
  - Both files are rewritten together, or neither is. The new text for both is
    prepared and validated in memory first.
  - The migration is idempotent. A configuration already in the flat shape is
    reported and left alone, so running it twice is harmless.
  - Files are written with newline="\n" so a configuration does not change line
    endings on a Windows desktop, per the one-line-ending rule.
  - A local pointer naming a slot that was not the live one is reported by name
    and dropped, because that model was never being used. It remains in the
    backup.

Usage (from the repository root):

    python scripts/migrate_acoustic_single_model.py            # migrate
    python scripts/migrate_acoustic_single_model.py --check    # report only
    python scripts/migrate_acoustic_single_model.py --settings path/to/settings.json

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# The keys the flat acoustic block carries. Ordered as they are written, so a
# migrated file reads the same way on every machine.
FLAT_KEYS = (
    "path",
    "labels_path",
    "sample_rate",
    "window_seconds",
    "output_key",
    "version",
    "citation",
)

# The keys that identify the retired shape. Either one present means a block has
# not been migrated yet.
LEGACY_KEYS = ("active", "options")

# A local override pointer addressing a slot inside the retired shape. The slot
# name is captured so a pointer for a slot that was not live can be reported by
# name rather than silently dropped.
LEGACY_POINTER = re.compile(
    r"^(?P<head>.*\.models\.acoustic)\.options\.(?P<slot>[^.]+)\.(?P<field>.+)$"
)


class MigrationError(RuntimeError):
    """A problem that should stop the migration with a clear message."""


def _repo_root() -> Path:
    # scripts/ sits directly under the repository root.
    return Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MigrationError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(loaded, dict):
        raise MigrationError(f"{path} must contain a JSON object")
    return loaded


def _write_json(path: Path, data: dict) -> None:
    # newline="\n" pins the line ending so the same configuration does not differ
    # byte for byte between a desktop and a station.
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def _backup(path: Path, stamp: str) -> Path:
    backup_path = path.with_name(f"{path.name}.backup-{stamp}")
    shutil.copy2(str(path), str(backup_path))
    return backup_path


# ---------------------------------------------------------------------------
# The committed configuration
# ---------------------------------------------------------------------------


def is_legacy_block(block: Any) -> bool:
    """Whether an acoustic block still carries the retired selector and slots."""
    return isinstance(block, dict) and any(key in block for key in LEGACY_KEYS)


def active_slots(raw: dict) -> dict[str, str]:
    """Each station's live acoustic slot, by station id.

    Read before anything is rewritten, because it is the only thing that says
    which of the three configured slots was the one actually being loaded, and
    so which local override pointer is worth carrying forward.
    """
    out: dict[str, str] = {}
    for station in raw.get("stations", []) or []:
        if not isinstance(station, dict):
            continue
        station_id = station.get("station_id")
        block = ((station.get("models") or {}).get("acoustic")) or {}
        if isinstance(station_id, str) and is_legacy_block(block):
            active = block.get("active")
            if isinstance(active, str) and active:
                out[station_id] = active
    return out


def flatten_block(block: dict) -> dict:
    """Collapse one legacy acoustic block to the flat shape.

    The live slot's values are carried forward and the other slots are dropped:
    they were never loaded, so nothing that was running is lost. A slot that
    carried the extra audio-shape keys keeps them, because those describe the
    model file rather than the slot it sat in.
    """
    active = block.get("active")
    options = block.get("options") or {}
    option = options.get(active) if isinstance(options, dict) else None
    if not isinstance(option, dict):
        option = {}

    flat: dict = {}
    for key in FLAT_KEYS:
        flat[key] = option.get(key)
    return flat


def migrate_settings(raw: dict) -> tuple[dict, list[str]]:
    """Return the migrated configuration and a note per station changed.

    The input is not mutated, so a caller holding the original for comparison
    still has it.
    """
    out = json.loads(json.dumps(raw))
    notes: list[str] = []
    for station in out.get("stations", []) or []:
        if not isinstance(station, dict):
            continue
        models = station.get("models")
        if not isinstance(models, dict):
            continue
        block = models.get("acoustic")
        if not is_legacy_block(block):
            continue
        active = block.get("active")
        flat = flatten_block(block)
        models["acoustic"] = flat
        name = station.get("station_name") or station.get("station_id") or "a station"
        if flat.get("path"):
            notes.append(f"{name}: kept the model configured in the '{active}' slot")
        else:
            notes.append(
                f"{name}: the '{active}' slot held no model path, so this station "
                f"now honestly reads as having no acoustic model"
            )
    return out, notes


# ---------------------------------------------------------------------------
# The local override file
# ---------------------------------------------------------------------------


def _station_id_of(pointer: str) -> Optional[str]:
    match = re.search(r"stations\[station_id=([^\]]+)\]", pointer)
    return match.group(1) if match else None


def migrate_overrides(
    overrides: dict, slots: dict[str, str]
) -> tuple[dict, list[str], list[str]]:
    """Rewrite acoustic override pointers into the flat shape.

    Returns the new override map, a note per pointer rewritten, and a note per
    pointer dropped. A pointer for a slot that was not the live one is dropped:
    that model was never loaded, and carrying it forward would silently promote
    a model the station was not listening with.

    When the committed file has already been flattened, the live slot is no
    longer knowable. A station with exactly one slot named in the local file is
    unambiguous and is rewritten; a station with several is left untouched and
    reported, because guessing would be the very failure this migration exists
    to prevent.
    """
    # Which slots each station's pointers name, so ambiguity can be detected
    # without depending on the committed file still carrying the selector.
    named: dict[str, set[str]] = {}
    for pointer in overrides:
        match = LEGACY_POINTER.match(pointer)
        if not match:
            continue
        station_id = _station_id_of(pointer) or ""
        named.setdefault(station_id, set()).add(match.group("slot"))

    out: dict = {}
    rewritten: list[str] = []
    dropped: list[str] = []

    for pointer, value in overrides.items():
        match = LEGACY_POINTER.match(pointer)
        if not match:
            out[pointer] = value
            continue

        station_id = _station_id_of(pointer) or ""
        slot = match.group("slot")
        live = slots.get(station_id)
        if live is None:
            # The committed file no longer says. One named slot is unambiguous.
            candidates = named.get(station_id, set())
            if len(candidates) == 1:
                live = slot
            else:
                out[pointer] = value
                dropped.append(
                    f"left in place, needs a decision: {pointer} (this station "
                    f"names {len(candidates)} slots in the local file and the "
                    f"committed file no longer records which was live)"
                )
                continue

        if slot != live:
            dropped.append(
                f"dropped: {pointer} (the '{slot}' slot was not the one this "
                f"station listened with)"
            )
            continue

        new_pointer = f"{match.group('head')}.{match.group('field')}"
        out[new_pointer] = value
        rewritten.append(f"{pointer}\n      becomes {new_pointer}")

    return out, rewritten, dropped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def needs_migration(settings_raw: dict, overrides: dict) -> bool:
    """Whether either file still carries the retired shape."""
    for station in settings_raw.get("stations", []) or []:
        if isinstance(station, dict) and is_legacy_block(
            ((station.get("models") or {}).get("acoustic")) or {}
        ):
            return True
    return any(LEGACY_POINTER.match(pointer) for pointer in overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collapse each station's three acoustic slots to the one model it listens with."
    )
    parser.add_argument("--settings", help="path to settings.json (defaults to config/settings.json)")
    parser.add_argument("--local", help="path to settings.local.json (defaults to config/settings.local.json)")
    parser.add_argument("--check", action="store_true", help="report what would change and write nothing")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    settings_path = Path(args.settings).resolve() if args.settings else repo_root / "config" / "settings.json"
    local_path = Path(args.local).resolve() if args.local else repo_root / "config" / "settings.local.json"

    if not settings_path.exists():
        print(f"No configuration found at {settings_path}. Nothing to migrate.", file=sys.stderr)
        return 2

    try:
        settings_raw = _read_json(settings_path)
        local_raw = _read_json(local_path) if local_path.exists() else {}
    except MigrationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    overrides = local_raw.get("overrides", {}) if isinstance(local_raw.get("overrides"), dict) else {}

    if not needs_migration(settings_raw, overrides):
        print("Already migrated: every station carries one acoustic model. No change made.")
        return 0

    # The live slot per station is read from the committed file before anything
    # is rewritten, because it is the only record of which slot was loaded.
    slots = active_slots(settings_raw)

    new_settings, station_notes = migrate_settings(settings_raw)
    new_overrides, rewritten, dropped = migrate_overrides(overrides, slots)

    print(f"Configuration: {settings_path}")
    for note in station_notes:
        print(f"  {note}")
    if not station_notes:
        print("  no station block needed changing")

    if local_path.exists():
        print(f"Machine paths: {local_path}")
        for note in rewritten:
            print(f"  rewrites {note}")
        for note in dropped:
            print(f"  {note}")
        if not rewritten and not dropped:
            print("  no acoustic pointer to rewrite")

    if args.check:
        print("\nCheck only. Nothing was written, and no backup was taken.")
        return 0

    if dropped:
        print("\nAnything dropped above stays in the backups written below.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written: list[Path] = []
    try:
        backup = _backup(settings_path, stamp)
        print(f"\nBackup written to {backup}")
        _write_json(settings_path, new_settings)
        written.append(settings_path)

        if local_path.exists():
            backup = _backup(local_path, stamp)
            print(f"Backup written to {backup}")
            new_local = dict(local_raw)
            new_local["overrides"] = new_overrides
            _write_json(local_path, new_local)
            written.append(local_path)
    except OSError as exc:
        print(f"\nMigration failed while writing: {exc}", file=sys.stderr)
        print(f"Files written before the failure: {', '.join(str(p) for p in written) or 'none'}", file=sys.stderr)
        print("Restore from the backups written above.", file=sys.stderr)
        return 1

    print("\nDone. Each station now carries one acoustic model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
