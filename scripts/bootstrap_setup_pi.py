#!/usr/bin/env python3
"""Audtheia field-station provisioning, driven from the desktop.

Path: scripts/bootstrap_setup_pi.py

This runs on the desktop and stands up a Raspberry Pi field station over the
network, so the person setting up a station never needs a keyboard or a screen on
the Pi. Given a station defined in the desktop configuration and the address of a
Pi that has booted with SSH reachable, it:

  1. Confirms it can reach the Pi.
  2. Installs a per-station SSH key on the Pi, so every later connection is
     key-based rather than password-based.
  3. Sends the application code, the station's own configuration (marked as a
     field node), and the station's models.
  4. Runs the Pi-side setup script on the Pi, which installs the field
     dependencies, wires the network hotspot, and enables the boot service.

It uses only the Python standard library and drives the system's own SSH client,
so it works on Windows, macOS, and Linux with nothing extra to install. Every SSH
action goes through one small runner object; a dry run swaps in a runner that
prints what it would do instead of doing it, and a test swaps in a runner that
records the calls, so the whole flow is verifiable without a real Pi.

The one step this cannot do for the user is the very first one: a bare Pi has no
software at all, so it must first be flashed with Raspberry Pi OS with SSH and
the network enabled (Raspberry Pi Imager writes all of that headlessly). After
that first boot, everything here is automatic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

# The Pi-side script this orchestrator sends and runs. It lives beside this file.
PI_PAYLOAD = REPO_ROOT / "scripts" / "setup-pi.sh"

# Where the sent files land on the Pi, under the login user's home directory.
REMOTE_ROOT = "audtheia"

# What never travels to a field station: version-control internals, the desktop's
# own environment and database, and local data and reports. The Pi builds its own.
ARCHIVE_EXCLUDES = {".git", ".venv", "venv", "env", "__pycache__", "data", "database", "reports"}


def _step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def _info(message: str) -> None:
    print(f"    {message}", flush=True)


class SetupPiError(Exception):
    """A provisioning step failed in a way that should stop with a clear message."""


# ---------------------------------------------------------------------------
# The SSH runner seam. The real one drives the system ssh and scp; a recording
# runner captures calls for a test; a logging runner prints them for a dry run.
# ---------------------------------------------------------------------------


class SshRunner:
    """Runs commands on the Pi and copies files to it through the system client."""

    def __init__(self, host: str, user: str, port: int = 22, key_path: Optional[Path] = None) -> None:
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

    def _ssh_base(self, use_key: bool) -> list[str]:
        cmd = ["ssh", "-p", str(self.port)]
        if use_key and self.key_path is not None:
            cmd += ["-i", str(self.key_path), "-o", "BatchMode=yes"]
        cmd += [f"{self.user}@{self.host}"]
        return cmd

    def check_tools(self) -> None:
        for tool in ("ssh", "scp"):
            if shutil.which(tool) is None:
                raise SetupPiError(
                    f"the '{tool}' command was not found. On Windows enable the OpenSSH "
                    f"Client (Settings, Optional Features); it is already present on macOS "
                    f"and Linux."
                )

    def run(self, remote_command: str, *, use_key: bool = False, check: bool = True) -> int:
        proc = subprocess.run(self._ssh_base(use_key) + [remote_command])
        if check and proc.returncode != 0:
            raise SetupPiError(f"remote command failed ({proc.returncode}): {remote_command}")
        return proc.returncode

    def put(self, local: Path, remote: str, *, use_key: bool = False) -> None:
        cmd = ["scp", "-P", str(self.port)]
        if use_key and self.key_path is not None:
            cmd += ["-i", str(self.key_path), "-o", "BatchMode=yes"]
        cmd += [str(local), f"{self.user}@{self.host}:{remote}"]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SetupPiError(f"copy to the Pi failed ({proc.returncode}): {local} -> {remote}")


class LoggingRunner:
    """A runner that prints each action instead of performing it, for a dry run."""

    def __init__(self, host: str, user: str, port: int = 22) -> None:
        self.host = host
        self.user = user
        self.port = port
        self.key_path = None
        self.calls: list[tuple] = []

    def check_tools(self) -> None:
        self.calls.append(("check_tools",))
        _info("would confirm ssh and scp are available")

    def run(self, remote_command: str, *, use_key: bool = False, check: bool = True) -> int:
        self.calls.append(("run", remote_command, use_key))
        _info(f"would run on the Pi ({'key' if use_key else 'password'}): {remote_command}")
        return 0

    def put(self, local: Path, remote: str, *, use_key: bool = False) -> None:
        self.calls.append(("put", str(local), remote, use_key))
        _info(f"would copy {Path(local).name} to {remote}")


# ---------------------------------------------------------------------------
# Building the artifacts that get sent to the Pi.
# ---------------------------------------------------------------------------


def build_code_archive(work_dir: Path) -> Path:
    """A tar.gz of the application code, without the desktop's own local state."""
    archive = work_dir / "audtheia-code.tar.gz"

    def _filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        parts = Path(info.name).parts
        if any(part in ARCHIVE_EXCLUDES for part in parts):
            return None
        return info

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(REPO_ROOT / "audtheia", arcname="audtheia", filter=_filter)
        tar.add(REPO_ROOT / "requirements.txt", arcname="requirements.txt")
    return archive


def build_pi_settings(settings, station_id: str, work_dir: Path) -> Path:
    """The station's own configuration, marked as a field node.

    The desktop's configuration is the source of truth. This derives the Pi's copy
    from it: the same shared settings, but with this node's role set to field and
    its one active station selected, so the loader on the Pi validates cleanly and
    the field runner knows which station it is. Nothing is invented; only the node
    block is set and the station list is narrowed to the one being deployed.
    """
    station = settings.station(station_id)
    raw = json.loads(json.dumps(settings.raw))  # a deep copy, so the desktop's is untouched
    raw["node"] = {"role": "pi", "active_station_id": station_id}
    raw["stations"] = [station]

    out = work_dir / "settings.json"
    out.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return out


def station_model_files(settings, station_id: str) -> list[tuple[Path, str]]:
    """The model files for one station that exist locally and can be staged.

    Each entry pairs the local file with the repository-relative path it belongs
    at, so the Pi receives every model at exactly the path its configuration
    already names. A model whose file is not present is skipped rather than
    failing the push, so a station can be provisioned before every model is in
    place and the missing ones sent later.
    """
    station = settings.station(station_id)
    files: list[tuple[Path, str]] = []

    visual = station.get("models", {}).get("visual_pi", {}).get("path")
    if visual:
        p = settings.resolve_path(visual)
        if p.exists():
            files.append((p, visual))

    acoustic = station.get("models", {}).get("acoustic", {})
    active = acoustic.get("active")
    option = acoustic.get("options", {}).get(active, {}) if active else {}
    apath = option.get("path")
    if apath:
        p = settings.resolve_path(apath)
        if p.exists():
            files.append((p, apath))

    return files


def build_pi_secrets(settings, work_dir: Path) -> Optional[Path]:
    """A least-privilege secrets file for the Pi, or nothing when there is none.

    A field station needs only its own hotspot key, never the desktop's data
    credentials, so only that one value travels. When it is not set, nothing is
    sent and the hotspot step on the Pi is skipped with a clear note rather than
    failing.
    """
    hotspot = settings.secrets.get("hotspot_password")
    if not hotspot:
        return None
    out = work_dir / "secrets.json"
    out.write_text(json.dumps({"hotspot_password": hotspot}, indent=2), encoding="utf-8")
    return out


def ensure_station_key(settings, station_id: str) -> tuple[Path, Path]:
    """A per-station SSH key pair on the desktop, created once and reused.

    Key-based access replaces the password after the first connection, so a
    station is administered with its own credential rather than a shared secret.
    """
    key_dir = REPO_ROOT / ".keys"
    key_dir.mkdir(parents=True, exist_ok=True)
    private = key_dir / f"station_{station_id}"
    public = key_dir / f"station_{station_id}.pub"
    if not private.exists():
        if shutil.which("ssh-keygen") is None:
            raise SetupPiError("ssh-keygen was not found; it ships with the OpenSSH client.")
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(private), "-C", f"audtheia-{station_id}"],
            check=True,
        )
    return private, public


# ---------------------------------------------------------------------------
# The orchestration itself.
# ---------------------------------------------------------------------------


def provision(
    settings,
    station_id: str,
    *,
    runner,
    work_dir: Path,
    make_key: bool = True,
    generate_key: bool = True,
    preauthorized_key: Optional[Path] = None,
) -> None:
    """Send everything a station needs and run the Pi-side setup, through the runner.

    When a preauthorized key is given, the desktop's key is already trusted by the
    Pi (it was added when the card was flashed), so the connection is key-based
    from the first step and no password is ever needed, which is what lets the
    guided flow run without a terminal prompt.
    """
    station = settings.station(station_id)
    _info(f"Station: {station.get('station_name')} ({station_id})")

    runner.check_tools()

    if preauthorized_key is not None:
        # Key-first: the desktop's key is already authorized on the Pi, so every
        # step uses the key and no password is asked for.
        if hasattr(runner, "key_path"):
            runner.key_path = preauthorized_key
        send_with_key = True
        _step("Confirming the Pi is reachable")
        runner.run("echo audtheia-reachable", use_key=True)
    else:
        send_with_key = make_key
        _step("Confirming the Pi is reachable")
        runner.run("echo audtheia-reachable", use_key=False)
        if make_key:
            _step("Installing a per-station key on the Pi")
            if generate_key:
                private, public = ensure_station_key(settings, station_id)
            else:
                # A preview creates no real key on disk; a nominal path stands in so
                # the steps that would send and authorize it can still be shown.
                private, public = None, work_dir / "station_key.pub"
                public.write_text("(preview)\n", encoding="utf-8")
            runner.put(public, f"{REMOTE_ROOT}/station_key.pub", use_key=False)
            # Append the key to the authorized set, once, over the first password
            # connection; later steps use the key.
            runner.run(
                f"mkdir -p ~/.ssh {REMOTE_ROOT} && "
                f"cat ~/{REMOTE_ROOT}/station_key.pub >> ~/.ssh/authorized_keys && "
                f"chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys",
                use_key=False,
            )
            if private is not None and hasattr(runner, "key_path"):
                runner.key_path = private

    _step("Preparing the files to send")
    archive = build_code_archive(work_dir)
    pi_settings = build_pi_settings(settings, station_id, work_dir)
    pi_secrets = build_pi_secrets(settings, work_dir)
    models = station_model_files(settings, station_id)
    _info(f"code archive, station configuration, and {len(models)} model file(s)")

    _step("Sending the files to the Pi")
    runner.run(f"mkdir -p ~/{REMOTE_ROOT}", use_key=send_with_key)
    runner.put(archive, f"{REMOTE_ROOT}/audtheia-code.tar.gz", use_key=send_with_key)
    runner.put(pi_settings, f"{REMOTE_ROOT}/settings.json", use_key=send_with_key)
    if pi_secrets is not None:
        runner.put(pi_secrets, f"{REMOTE_ROOT}/secrets.json", use_key=send_with_key)
    runner.put(PI_PAYLOAD, f"{REMOTE_ROOT}/setup-pi.sh", use_key=send_with_key)
    for local, rel in models:
        parent = str(Path(rel).parent.as_posix())
        runner.run(f"mkdir -p ~/{REMOTE_ROOT}/{parent}", use_key=send_with_key)
        runner.put(local, f"{REMOTE_ROOT}/{rel}", use_key=send_with_key)

    _step("Running the Pi-side setup")
    runner.run(f"bash ~/{REMOTE_ROOT}/setup-pi.sh", use_key=send_with_key)

    _step("Done")
    _info(f"The station is configured. It answers on the network at {station.get('station_name')}.")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _resolve_target(settings, args) -> tuple[str, str, int]:
    ssh_cfg = settings.raw.get("network", {}).get("ssh", {})
    host = args.host or ssh_cfg.get("host")
    user = args.user or ssh_cfg.get("username")
    port = args.port or ssh_cfg.get("port") or 22
    if not host:
        raise SetupPiError("no Pi address given; pass --host or set network.ssh.host in settings.")
    if not user:
        raise SetupPiError("no Pi user given; pass --user or set network.ssh.username in settings.")
    return host, user, int(port)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="connect-pi",
        description="Provision a Raspberry Pi field station over SSH from the desktop.",
    )
    parser.add_argument("--station-id", required=True, help="The station to deploy, by its id.")
    parser.add_argument("--host", default=None, help="The Pi's address (overrides settings).")
    parser.add_argument("--user", default=None, help="The Pi's login user (overrides settings).")
    parser.add_argument("--port", type=int, default=None, help="The Pi's SSH port (overrides settings).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every action without contacting a Pi, to preview a deployment.",
    )
    parser.add_argument(
        "--no-key",
        action="store_true",
        help="Skip installing a per-station key and use the password path throughout.",
    )
    parser.add_argument(
        "--show-key",
        action="store_true",
        help="Print the desktop's public key for this station and exit, for authorizing it on the Pi at flash time.",
    )
    parser.add_argument(
        "--key-auth",
        action="store_true",
        help="Connect using the station's already-authorized key, with no password (the guided flow uses this).",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        from audtheia.config import load_settings

        settings = load_settings()

        if args.show_key:
            _private, public = ensure_station_key(settings, args.station_id)
            sys.stdout.write(public.read_text(encoding="utf-8"))
            return 0

        host, user, port = _resolve_target(settings, args)

        print("Audtheia field-station provisioning")
        print(f"Target: {user}@{host}:{port}")

        preauthorized = None
        if args.key_auth:
            if args.dry_run:
                preauthorized = REPO_ROOT / ".keys" / f"station_{args.station_id}"
            else:
                private, _public = ensure_station_key(settings, args.station_id)
                preauthorized = private

        runner = LoggingRunner(host, user, port) if args.dry_run else SshRunner(host, user, port, key_path=preauthorized)

        with tempfile.TemporaryDirectory(prefix="audtheia-pi-") as tmp:
            provision(
                settings,
                args.station_id,
                runner=runner,
                work_dir=Path(tmp),
                make_key=not (args.no_key or args.key_auth),
                generate_key=(not args.no_key) and (not args.dry_run) and (not args.key_auth),
                preauthorized_key=preauthorized,
            )
        return 0

    except SetupPiError as exc:
        print(f"\nProvisioning stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nProvisioning interrupted. It is safe to run again.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    # Run from the repository root so the configuration import resolves.
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
