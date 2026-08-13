"""SSH-pull station-to-desktop sync: the desktop side, and a station-side CLI.

The desktop hub pulls a field station's captured record across the network, so
the effort stays on the powered hub and the station stays lean and spends its
energy on capture. This reuses, without changing, the append-only sync the
storage layer already defines (audtheia/storage/database.py): the station exports
its unconfirmed rows, the desktop imports them, and the station marks them
confirmed. A pull can therefore never overwrite a desktop-owned value, and
re-delivering a batch changes nothing, so an interrupted pull is safe to repeat.

Transport is injected, so the sync logic is tested with no network. A command
runner runs one station-side command and returns its output; the real runner
drives ssh against a connected Pi, and a test runner runs the same station CLI
against an in-process station database. The station CLI has two verbs:

    python -m audtheia.sync export --batch N     print the next unconfirmed batch as JSON
    python -m audtheia.sync confirm              read confirmed ids as JSON on stdin, stamp them

This module carries the data transport only. Pulling the media files an event
references, the automatic reachable-station loop, and the interface control are
separate, layered on top of this core.
"""
from __future__ import annotations

import json
import shlex
import subprocess  # noqa: S404 - used only to drive ssh with a fixed argument vector
import sys
from typing import Optional, Protocol

from audtheia.storage.database import DEFAULT_BATCH_SIZE, SYNCABLE_TABLES, Database

__all__ = [
    "SyncTransportError",
    "CommandRunner",
    "SshCommandRunner",
    "pull_once",
    "pull_all",
    "do_export",
    "do_confirm",
    "main",
]


class SyncTransportError(RuntimeError):
    """A station-side command could not be run or returned a failure."""


class CommandRunner(Protocol):
    """Runs one station-side sync command and returns its stdout.

    ``args`` are the CLI arguments after ``-m audtheia.sync`` (for example
    ``["export", "--batch", "500"]``). ``input_text``, when given, is written to
    the command's stdin (the confirm verb reads its ids there). A non-zero exit is
    raised as a SyncTransportError.
    """

    def run(self, args: list, *, input_text: Optional[str] = None) -> str: ...


# ===========================================================================
# The station CLI verbs, factored so both the real CLI and the test runner
# share one implementation of each.
# ===========================================================================


def do_export(db: Database, batch_size: int) -> str:
    """Return the station's next unconfirmed batch as a JSON string."""
    return json.dumps(db.export_unsynced_batch(batch_size=batch_size))


def do_confirm(db: Database, input_text: str) -> str:
    """Stamp the confirmed ids on the station and return the counts as JSON.

    ``input_text`` is the desktop's confirmation: a JSON object mapping each
    syncable table to the ids the desktop now holds. Only those ids are stamped,
    and an already-stamped row is left untouched, so a repeated confirmation is
    harmless.
    """
    confirmed = json.loads(input_text) if input_text else {}
    counts = {t: db.mark_synced(t, confirmed.get(t, []) or []) for t in SYNCABLE_TABLES}
    return json.dumps(counts)


# ===========================================================================
# The desktop side: pull rounds over an injected runner.
# ===========================================================================


def _batch_is_empty(batch: dict) -> bool:
    return sum(len(batch.get(t, []) or []) for t in SYNCABLE_TABLES) == 0


def pull_once(desktop: Database, runner: CommandRunner, *, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Pull one batch: export on the station, import on the desktop, confirm.

    Returns a dict with the per-table confirmed counts for this round and whether
    the station reported nothing left (``empty``). The three steps are the exact
    append-only cycle the storage layer defines, run across the runner: the
    station never has a row removed until the desktop holds it.
    """
    try:
        exported = runner.run(["export", "--batch", str(int(batch_size))])
    except SyncTransportError:
        raise
    except Exception as exc:  # noqa: BLE001 - any transport failure is surfaced as one error type
        raise SyncTransportError(f"the station export command failed: {exc}") from exc

    try:
        batch = json.loads(exported) if exported.strip() else {}
    except json.JSONDecodeError as exc:
        raise SyncTransportError(f"the station export did not return valid JSON: {exc}") from exc

    if _batch_is_empty(batch):
        return {"confirmed": {t: 0 for t in SYNCABLE_TABLES}, "empty": True}

    confirmed = desktop.import_batch(batch)
    runner.run(["confirm"], input_text=json.dumps(confirmed))
    return {"confirmed": {t: len(confirmed.get(t, [])) for t in SYNCABLE_TABLES}, "empty": False}


def pull_all(desktop: Database, runner: CommandRunner, *, batch_size: int = DEFAULT_BATCH_SIZE,
             max_rounds: int = 10000) -> dict:
    """Pull batches until the station has nothing left, or a round cap is hit.

    Returns the total confirmed per table and the number of rounds run. The cap is
    a safety bound only; a normal pull ends when the station reports an empty
    batch. Because every step is idempotent, a pull interrupted at any point is
    simply resumed on the next call.
    """
    totals = {t: 0 for t in SYNCABLE_TABLES}
    rounds = 0
    while rounds < max_rounds:
        result = pull_once(desktop, runner, batch_size=batch_size)
        if result["empty"]:
            break
        for t in SYNCABLE_TABLES:
            totals[t] += result["confirmed"][t]
        rounds += 1
    return {"rounds": rounds, "confirmed": totals, "total": sum(totals.values())}


class SshCommandRunner:
    """A CommandRunner that drives ssh against a connected station. PENDING
    ON-DEVICE VALIDATION.

    Builds one ssh invocation per command from the station's stored connection
    target (host, user, port, key) and its remote repository directory and Python.
    The station verb runs under the station's own environment, so it reads the
    station's configured database. Only the transport is here; it is exercised on
    a real Pi during field validation, while the sync logic above is tested with a
    local runner.
    """

    def __init__(self, *, host: str, user: str, repo_dir: str, python: str = "python3",
                 port: int = 22, key_path: Optional[str] = None, timeout: int = 60) -> None:
        self._host = host
        self._user = user
        self._repo_dir = repo_dir
        self._python = python
        self._port = int(port)
        self._key_path = key_path
        self._timeout = int(timeout)

    def _ssh_prefix(self) -> list:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", str(self._port)]
        if self._key_path:
            cmd += ["-i", self._key_path]
        cmd.append(f"{self._user}@{self._host}")
        return cmd

    def run(self, args: list, *, input_text: Optional[str] = None) -> str:
        remote = (
            f"cd {shlex.quote(self._repo_dir)} && "
            f"{shlex.quote(self._python)} -m audtheia.sync "
            + " ".join(shlex.quote(str(a)) for a in args)
        )
        cmd = self._ssh_prefix() + [remote]
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argument vector; remote string is quoted
                cmd, input=input_text, capture_output=True, text=True, timeout=self._timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SyncTransportError(f"ssh to the station failed: {exc}") from exc
        if proc.returncode != 0:
            raise SyncTransportError(
                f"the station sync command exited {proc.returncode}: {proc.stderr.strip() or 'no error output'}"
            )
        return proc.stdout


# ===========================================================================
# The station-side CLI entry point.
# ===========================================================================


def main(argv: Optional[list] = None) -> int:
    """Run one station-side sync verb against the station's configured database.

    Invoked on the station by the desktop over ssh. It loads the station's
    settings, opens its database, and either prints the next unconfirmed batch
    (``export``) or stamps confirmed ids read from stdin (``confirm``). It writes
    only JSON to stdout, so the desktop can parse it, and nothing else.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="audtheia.sync", description="Station-to-desktop sync (station side).")
    parser.add_argument("--settings", default=None, help="path to a settings file; the default loader is used when omitted")
    sub = parser.add_subparsers(dest="verb", required=True)
    exp = sub.add_parser("export", help="print the next unconfirmed batch as JSON")
    exp.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    sub.add_parser("confirm", help="read confirmed ids as JSON on stdin and stamp them")

    args = parser.parse_args(argv)

    from audtheia.config import load_settings  # noqa: PLC0415 - deferred so importing this module is cheap

    settings = load_settings(args.settings) if args.settings else load_settings()
    db = Database(settings.db_path())

    if args.verb == "export":
        sys.stdout.write(do_export(db, args.batch))
    else:
        sys.stdout.write(do_confirm(db, sys.stdin.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
