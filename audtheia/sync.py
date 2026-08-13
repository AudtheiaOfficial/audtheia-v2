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
import logging
import shlex
import subprocess  # noqa: S404 - used only to drive ssh with a fixed argument vector
import sys
import threading
from typing import Optional, Protocol

from audtheia.storage.database import DEFAULT_BATCH_SIZE, SYNCABLE_TABLES, Database

logger = logging.getLogger("audtheia.sync")

__all__ = [
    "SyncTransportError",
    "CommandRunner",
    "SshCommandRunner",
    "MediaFetcher",
    "ScpMediaFetcher",
    "pull_once",
    "pull_all",
    "pull_media",
    "do_export",
    "do_confirm",
    "is_reachable",
    "station_ssh_runner",
    "station_media_fetcher",
    "sync_station",
    "sync_reachable_stations",
    "run_sync_loop",
    "start_sync_loop",
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


def pull_once(desktop: Database, runner: CommandRunner, *, batch_size: int = DEFAULT_BATCH_SIZE,
              media_fetcher: Optional["MediaFetcher"] = None, repo_root: Optional[str] = None) -> dict:
    """Pull one batch: export on the station, import on the desktop, fetch media, confirm.

    Returns a dict with the per-table confirmed counts for this round, the media
    counts, and whether the station reported nothing left (``empty``). The order is
    deliberate: the database rows are imported, then the media files an event
    references are fetched, and only then are the rows confirmed on the station.
    So if a media fetch cannot even reach the station, nothing is confirmed and the
    same rows are retried on the next pull, and the station never drops a row the
    desktop does not yet hold. A single missing file is best-effort and never
    stalls the record, since the authoritative row is what the firewall protects.
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
        return {"confirmed": {t: 0 for t in SYNCABLE_TABLES}, "media": {"fetched": 0, "failed": 0}, "empty": True}

    confirmed = desktop.import_batch(batch)
    media = {"fetched": 0, "failed": 0}
    if media_fetcher is not None:
        media = pull_media(batch.get("observations", []), fetcher=media_fetcher, repo_root=repo_root or ".")
    runner.run(["confirm"], input_text=json.dumps(confirmed))
    return {"confirmed": {t: len(confirmed.get(t, [])) for t in SYNCABLE_TABLES}, "media": media, "empty": False}


def pull_all(desktop: Database, runner: CommandRunner, *, batch_size: int = DEFAULT_BATCH_SIZE,
             media_fetcher: Optional["MediaFetcher"] = None, repo_root: Optional[str] = None,
             max_rounds: int = 10000) -> dict:
    """Pull batches until the station has nothing left, or a round cap is hit.

    Returns the total confirmed per table, the media totals, and the number of
    rounds run. The cap is a safety bound only; a normal pull ends when the station
    reports an empty batch. Because every step is idempotent, a pull interrupted at
    any point is simply resumed on the next call.
    """
    totals = {t: 0 for t in SYNCABLE_TABLES}
    media = {"fetched": 0, "failed": 0}
    rounds = 0
    while rounds < max_rounds:
        result = pull_once(desktop, runner, batch_size=batch_size, media_fetcher=media_fetcher, repo_root=repo_root)
        if result["empty"]:
            break
        for t in SYNCABLE_TABLES:
            totals[t] += result["confirmed"][t]
        media["fetched"] += result["media"]["fetched"]
        media["failed"] += result["media"]["failed"]
        rounds += 1
    return {"rounds": rounds, "confirmed": totals, "media": media, "total": sum(totals.values())}


# ===========================================================================
# Media: the frame and clip files an event references live under data/ on the
# station and must be fetched so the desktop can show them.
# ===========================================================================


class MediaFetcher(Protocol):
    """Fetches one media path from the station to the desktop.

    ``remote_rel`` is the repository-root-relative path the record stored (for
    example ``data/detections/visual/<event>/``); ``local_abs`` is where it must
    land on the desktop so the same stored path resolves. ``is_dir`` fetches a
    whole event directory (its frames and annotations) rather than a single file.
    """

    def fetch(self, remote_rel: str, local_abs, *, is_dir: bool) -> None: ...


def pull_media(observations: list, *, fetcher: "MediaFetcher", repo_root: str) -> dict:
    """Fetch the frames and clips the given observations reference. Best-effort.

    For each event, the whole representative-frame directory is fetched (so the
    frame strip and the event playback have every saved frame), and the audio clip
    if present. Each fetch is independent: a single file that cannot be fetched is
    counted as a failure and logged, but never raises, so one missing frame never
    stalls a sync. The authoritative record is the database row, which is already
    imported; media is the recoverable copy of what the row points at.
    """
    from pathlib import Path, PurePosixPath  # noqa: PLC0415

    root = Path(repo_root)
    fetched = 0
    failed = 0
    seen_dirs = set()

    def _try(remote_rel: str, local_abs, is_dir: bool) -> None:
        nonlocal fetched, failed
        try:
            fetcher.fetch(remote_rel, local_abs, is_dir=is_dir)
            fetched += 1
        except Exception as exc:  # noqa: BLE001 - a missing file is logged, never fatal
            failed += 1
            logger.warning("could not fetch station media %s (%s: %s)", remote_rel, type(exc).__name__, exc)

    for obs in observations:
        rep = obs.get("representative_frame")
        if rep:
            event_dir = str(PurePosixPath(rep).parent)
            if event_dir and event_dir not in ("", ".") and event_dir not in seen_dirs:
                seen_dirs.add(event_dir)
                _try(event_dir, root / event_dir, True)
        clip = obs.get("audio_clip_path")
        if clip:
            _try(clip, root / clip, False)
    return {"fetched": fetched, "failed": failed}


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


# The Pi conventions the desktop drives against: the station's repository sits at
# ~/audtheia and runs its own virtual environment. These match scripts/setup-pi.sh.
REMOTE_ROOT = "audtheia"
REMOTE_PYTHON = ".venv/bin/python"


class ScpMediaFetcher:
    """A MediaFetcher that copies station media over scp. PENDING ON-DEVICE VALIDATION.

    Builds one scp per fetch from the station's connection target. A directory is
    copied recursively into the local parent, so the event folder lands beside the
    record that references it; a clip is copied to its exact path. Only the
    transport is here; it is confirmed on a real Pi during field validation.
    """

    def __init__(self, *, host: str, user: str, repo_dir: str = REMOTE_ROOT, port: int = 22,
                 key_path: Optional[str] = None, timeout: int = 120) -> None:
        self._host = host
        self._user = user
        self._repo_dir = repo_dir
        self._port = int(port)
        self._key_path = key_path
        self._timeout = int(timeout)

    def _scp_prefix(self, recursive: bool) -> list:
        cmd = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", "-P", str(self._port)]
        if recursive:
            cmd.append("-r")
        if self._key_path:
            cmd += ["-i", self._key_path]
        return cmd

    def fetch(self, remote_rel: str, local_abs, *, is_dir: bool) -> None:
        from pathlib import Path  # noqa: PLC0415

        local_abs = Path(local_abs)
        remote = f"{self._repo_dir}/{remote_rel}"  # relative to the station user's home
        if is_dir:
            local_abs.parent.mkdir(parents=True, exist_ok=True)
            dest = str(local_abs.parent)
        else:
            local_abs.parent.mkdir(parents=True, exist_ok=True)
            dest = str(local_abs)
        cmd = self._scp_prefix(is_dir) + [f"{self._user}@{self._host}:{remote}", dest]
        try:
            proc = subprocess.run(  # noqa: S603 - fixed vector; remote is a stored record path
                cmd, capture_output=True, text=True, timeout=self._timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SyncTransportError(f"scp of {remote} failed: {exc}") from exc
        if proc.returncode != 0:
            raise SyncTransportError(f"scp of {remote} failed: {proc.stderr.strip()[:200] or 'no error output'}")


def is_reachable(host: str, port: int = 22, *, timeout: float = 3.0) -> bool:
    """Whether a TCP connection to the station opens, a cheap reachability probe."""
    import socket  # noqa: PLC0415

    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _station_target(station: dict) -> Optional[tuple]:
    prov = station.get("provisioning") or {}
    host = (prov.get("host") or "").strip()
    user = (prov.get("user") or "").strip()
    if not host or not user:
        return None
    return host, user, int(prov.get("port", 22))


def station_ssh_runner(settings, station: dict):
    """A real SshCommandRunner for a station, or None when it has no connection target."""
    from pathlib import Path  # noqa: PLC0415

    target = _station_target(station)
    if target is None:
        return None
    host, user, port = target
    key = Path(settings.repo_root) / ".keys" / ("station_" + str(station.get("station_id", "")))
    return SshCommandRunner(host=host, user=user, repo_dir=REMOTE_ROOT, python=REMOTE_PYTHON,
                            port=port, key_path=str(key) if key.exists() else None)


def station_media_fetcher(settings, station: dict):
    """A real ScpMediaFetcher for a station, or None when it has no connection target."""
    from pathlib import Path  # noqa: PLC0415

    target = _station_target(station)
    if target is None:
        return None
    host, user, port = target
    key = Path(settings.repo_root) / ".keys" / ("station_" + str(station.get("station_id", "")))
    return ScpMediaFetcher(host=host, user=user, repo_dir=REMOTE_ROOT, port=port,
                           key_path=str(key) if key.exists() else None)


def sync_station(settings, desktop: Database, station: dict, *, runner: Optional[CommandRunner] = None,
                 fetcher: Optional["MediaFetcher"] = None, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Pull one station's full record and media to the desktop.

    Builds the real ssh runner and media fetcher from the station's stored
    connection target when they are not supplied (they are supplied in tests). A
    station with no connection target is skipped with a plain reason.
    """
    runner = runner or station_ssh_runner(settings, station)
    if runner is None:
        return {"skipped": True, "reason": "this station has no stored connection target; connect its Pi first."}
    if fetcher is None:
        fetcher = station_media_fetcher(settings, station)
    result = pull_all(desktop, runner, batch_size=batch_size, media_fetcher=fetcher,
                      repo_root=str(settings.repo_root))
    result["station_id"] = station.get("station_id")
    return result


def sync_reachable_stations(settings, desktop: Database, *, runner_factory=None, fetcher_factory=None,
                            reachable=None) -> dict:
    """Sync every station that has a connection target and is reachable now.

    The factories and the reachability predicate are injected, so this is tested
    with no network; the desktop supplies the real ones. A per-station transport
    failure is captured and never stops the other stations. Whether to run this
    automatically is the caller's decision (the desktop's background loop gates it
    on the auto-sync setting); a manual sync calls it for one station regardless.
    """
    runner_factory = runner_factory or (lambda s: station_ssh_runner(settings, s))
    fetcher_factory = fetcher_factory or (lambda s: station_media_fetcher(settings, s))
    reachable = reachable or (lambda s: is_reachable((s.get("provisioning") or {}).get("host", ""),
                                                     int((s.get("provisioning") or {}).get("port", 22))))
    results = {}
    for station in settings.stations():
        if _station_target(station) is None:
            continue
        sid = station.get("station_id")
        if not reachable(station):
            results[sid] = {"skipped": True, "reason": "not reachable"}
            continue
        runner = runner_factory(station)
        if runner is None:
            continue
        try:
            results[sid] = pull_all(desktop, runner, media_fetcher=fetcher_factory(station),
                                    repo_root=str(settings.repo_root))
        except SyncTransportError as exc:
            results[sid] = {"error": str(exc)}
    return results


# The desktop's automatic sync cadence: how often it looks for reachable stations
# to pull. A property of the powered hub, not of a deployment, so it lives here.
DEFAULT_SYNC_INTERVAL_SECONDS = 60.0


def run_sync_loop(settings, desktop, stop_event, *, interval: float = DEFAULT_SYNC_INTERVAL_SECONDS,
                  sync=sync_reachable_stations) -> None:
    """Pull reachable stations on a cadence until stopped.

    Runs one sync pass, then waits the interval, and repeats until ``stop_event``
    is set. A pass that fails is logged and the loop continues, so a station that
    is briefly unreachable, or a transient network error, never stops the loop.
    The pass function is injected so the loop is tested without a network.
    """
    while not stop_event.is_set():
        try:
            sync(settings, desktop)
        except Exception as exc:  # noqa: BLE001 - a failed round is logged, the loop continues
            logger.warning("automatic station sync round failed: %s", exc)
        if stop_event.wait(interval):
            break


def start_sync_loop(settings, desktop, *, interval: float = DEFAULT_SYNC_INTERVAL_SECONDS):
    """Start the automatic pull loop in a daemon thread, or nothing when off.

    Started only on the desktop and only when the buffer's auto-sync setting is on,
    so a station-only or opted-out deployment does no automatic pulling. Returns
    the stop event to set on shutdown, or None when no loop was started.
    """
    if getattr(settings, "node_role", None) != "desktop":
        return None
    if not (settings.raw.get("buffer", {}) or {}).get("auto_sync_when_reachable", False):
        return None
    stop = threading.Event()
    threading.Thread(
        target=run_sync_loop, args=(settings, desktop, stop), kwargs={"interval": interval},
        name="audtheia-station-sync", daemon=True,
    ).start()
    logger.info("automatic station sync loop started (every %.0f s)", interval)
    return stop


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
