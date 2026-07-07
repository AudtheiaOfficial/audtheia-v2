#!/usr/bin/env python3
"""Audtheia desktop launcher.

Path: scripts/bootstrap_start.py

This is the daily entry point. It starts the desktop application and takes the
person straight to the interface, so running Audtheia is one step and needs no
commands to be remembered.

What it does:

  - Finds the isolated environment that setup created, and explains how to run
    setup if it is not there yet.
  - Starts the local web backend from that environment.
  - Waits until the backend is answering, then shows the local web address and
    offers to open it in the default browser. The address is always printed, so
    a person who prefers to open it themselves simply clicks or copies it.
  - Optionally shows a small system-tray icon with Open and Quit, for a person
    who would rather not keep a terminal window in view. The tray needs two extra
    packages; when they are missing the launcher explains this and offers to
    install them with a single confirmation, then continues without them if
    declined. The daily launch never depends on the tray.

It uses only the standard library for the core launch, so it always runs. The
tray is the one optional part, and it degrades gracefully. The interface is
served on the loopback address, reachable only from the desktop it runs on.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"
SERVER_MODULE = "audtheia.app.server"

# Where the launcher looks for a logo to use as the tray icon. When a file is
# present here it is used; otherwise a simple placeholder icon is drawn. This is
# how a real logo can be added later with no change to any code: drop it in.
LOGO_CANDIDATES = (
    REPO_ROOT / "audtheia" / "app" / "static" / "logo.png",
    REPO_ROOT / "audtheia" / "app" / "static" / "favicon.png",
)

# The optional packages that draw the system-tray icon. They are installed only
# when a person asks for the tray, never as part of the ordinary install.
TRAY_PACKAGES = ("pystray>=0.19", "Pillow>=10")

# How long to wait for the backend to start answering before giving up.
STARTUP_TIMEOUT_SECONDS = 30


def _info(message: str) -> None:
    print(f"    {message}", flush=True)


class LaunchError(Exception):
    """A problem that should stop the launch with a clear message."""


# ---------------------------------------------------------------------------
# Environment and server.
# ---------------------------------------------------------------------------


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def require_environment() -> Path:
    interpreter = venv_python()
    if not interpreter.exists():
        raise LaunchError(
            "the Audtheia environment was not found. Run setup first: "
            "setup.bat on Windows, or ./setup.sh on macOS and Linux."
        )
    return interpreter


def server_address(settings) -> tuple[str, int]:
    server = settings.raw.get("server", {})
    return server.get("host", "127.0.0.1"), int(server.get("port", 8000))


def display_host(host: str) -> str:
    # A bind address of all-interfaces is shown as loopback, which is the address
    # a person actually opens in a browser on the same machine.
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def start_server(interpreter: Path):
    """Start the backend as a child process that inherits this terminal."""
    return subprocess.Popen([str(interpreter), "-m", SERVER_MODULE], cwd=str(REPO_ROOT))


def wait_until_reachable(host: str, port: int, timeout: int = STARTUP_TIMEOUT_SECONDS) -> bool:
    """Return once the backend answers on host and port, or False on timeout."""
    connect_host = display_host(host)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# Browser.
# ---------------------------------------------------------------------------


def _ask_yes_no(question: str, default: bool) -> bool:
    """Ask a yes or no question, defaulting without blocking when there is no
    interactive terminal (for example when launched from a service or a pipe)."""
    if not sys.stdin or not sys.stdin.isatty():
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(question + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def offer_browser(url: str, *, mode: str) -> None:
    """Show the address and, depending on mode, open it or offer to.

    mode is 'ask' (offer, default yes), 'open' (open without asking), or 'none'
    (only show the address).
    """
    print("")
    print(f"    Audtheia is running at:  {url}")
    print("")
    if mode == "none":
        return
    if mode == "open" or _ask_yes_no("    Open Audtheia in your browser now?", default=True):
        webbrowser.open(url)


# ---------------------------------------------------------------------------
# The optional system tray.
# ---------------------------------------------------------------------------


def _tray_available() -> bool:
    try:
        import pystray  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import problem means the tray is unavailable
        return False


def install_tray(interpreter: Path) -> bool:
    """Install the tray packages into the environment. Returns success."""
    _info("Installing the tray packages: " + ", ".join(TRAY_PACKAGES))
    try:
        subprocess.run(
            [str(interpreter), "-m", "pip", "install", *TRAY_PACKAGES],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        _info("the tray packages could not be installed; continuing without the tray.")
        return False


def _tray_image():
    """The tray icon: the logo when one is present, otherwise a simple mark."""
    from PIL import Image, ImageDraw

    for candidate in LOGO_CANDIDATES:
        if candidate.exists():
            try:
                return Image.open(candidate)
            except Exception:  # noqa: BLE001 - a bad image file falls back to the drawn mark
                break

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 6, 58, 58), fill=(15, 118, 130, 255))
    draw.text((26, 22), "A", fill=(255, 255, 255, 255))
    return image


def run_with_tray(interpreter: Path, url: str, proc) -> None:
    """Show a tray icon that opens the interface or quits the application."""
    import pystray

    def _open(icon, item):  # noqa: ANN001 - pystray callback signature
        webbrowser.open(url)

    def _quit(icon, item):  # noqa: ANN001 - pystray callback signature
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open Audtheia", _open, default=True),
        pystray.MenuItem("Quit", _quit),
    )
    icon = pystray.Icon("audtheia", _tray_image(), "Audtheia", menu)

    try:
        icon.run()
    finally:
        _stop_process(proc)


# ---------------------------------------------------------------------------
# Shutdown.
# ---------------------------------------------------------------------------


def _stop_process(proc) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def launch(args) -> int:
    from audtheia.config import load_settings

    interpreter = require_environment()
    settings = load_settings()
    host, port = server_address(settings)
    url = f"http://{display_host(host)}:{port}"

    want_tray = args.tray or args.install_tray
    if want_tray and not _tray_available():
        print("")
        _info("The system tray needs two extra packages that are not installed yet.")
        _info("With them, Audtheia shows a small icon with Open and Quit, so you do")
        _info("not need to keep this window in view.")
        if args.install_tray or _ask_yes_no("    Install the tray now?", default=False):
            if install_tray(interpreter):
                # The freshly installed packages live in the environment, not in
                # this interpreter, so the tray runs in a follow-up launch.
                _info("Tray installed. Start Audtheia again with the tray option to use it.")
            want_tray = False
        else:
            _info("Continuing without the tray.")
            want_tray = False

    print("Starting Audtheia")
    proc = start_server(interpreter)

    if not wait_until_reachable(host, port):
        _stop_process(proc)
        raise LaunchError(
            "the interface did not start in time. Check the messages above, and "
            "make sure setup completed and the database was initialized."
        )

    if want_tray and _tray_available():
        offer_browser(url, mode="none")
        _info("Audtheia is in the system tray. Use its icon to open or quit.")
        run_with_tray(interpreter, url, proc)
        return 0

    browser_mode = "none" if args.no_browser else ("open" if args.open else "ask")
    offer_browser(url, mode=browser_mode)
    _info("Audtheia is running. Press Ctrl-C here to stop it.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("")
        _info("Stopping Audtheia.")
        _stop_process(proc)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="start",
        description="Start the Audtheia desktop application and open the interface.",
    )
    parser.add_argument("--tray", action="store_true", help="Run with a system-tray icon when available.")
    parser.add_argument(
        "--install-tray", action="store_true",
        help="Install the optional tray packages, then run with the tray.",
    )
    parser.add_argument("--open", action="store_true", help="Open the browser without asking.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open or offer the browser.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        return launch(args)
    except LaunchError as exc:
        print(f"\nCould not start Audtheia: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
