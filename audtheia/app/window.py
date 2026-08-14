"""Audtheia desktop app window.

Path: audtheia/app/window.py

Opens the local interface in its own desktop window, so Audtheia runs as an
application rather than a tab inside a web browser. The window is a thin frame
around the same local address the browser would show, so nothing about the
interface or the server changes: this only chooses how the interface is
presented.

The window is drawn by pywebview, which wraps the operating system's own web
view (Edge WebView2 on Windows, WebKit on macOS, and the system web view on
Linux). That package is optional and is imported lazily inside the function that
opens the window, so importing this module never requires it. When it is not
installed, opening a window raises a clear message telling the person how to add
it, and the launcher treats that as a reason to fall back to the browser rather
than a failure.

Run it directly once the server is up:

    python -m audtheia.app.window                 (address read from settings)
    python -m audtheia.app.window http://127.0.0.1:8000

pywebview must run on its process's main thread, which is exactly where this runs
when the launcher starts it, so the window owns the main thread and closing it
returns control cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# The window title and a comfortable default size for a data interface. These are
# presentation constants for the desktop frame, not deployment settings, so they
# live here rather than in the configuration.
WINDOW_TITLE = "Audtheia"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 820
MIN_WIDTH = 900
MIN_HEIGHT = 600


def resolve_url(argv: Optional[list] = None) -> str:
    """The local address to show: the one given, or the configured server address.

    A caller (the launcher) passes the address it already knows. Run directly with
    no address, the same value is derived from the configuration, so the window
    opens on the right port without repeating it. An all-interfaces bind address is
    shown as loopback, which is the address a person actually opens on the same
    machine.
    """
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        return argv[0]

    from audtheia.config import load_settings

    settings = load_settings()
    server = settings.raw.get("server", {})
    host = server.get("host", "127.0.0.1")
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    port = int(server.get("port", 8000))
    return f"http://{host}:{port}"


def open_window(url: str, *, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> None:
    """Open a desktop window on the given address and run until it is closed.

    pywebview is imported here so this module loads without it. When it is absent
    the call stops with a clear, actionable message rather than a stack trace, so
    the launcher can fall back to the browser.
    """
    try:
        import webview
    except Exception as exc:  # noqa: BLE001 - any import problem means the window cannot open
        raise SystemExit(
            "The desktop app window needs the pywebview package, which is not "
            "installed. Install it with: pip install pywebview"
        ) from exc

    webview.create_window(
        WINDOW_TITLE,
        url,
        width=width,
        height=height,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
    )

    # Brand the window with the Audtheia mark rather than the interpreter's own
    # icon. The icon ships next to this package, so it resolves the same way on
    # every machine. pywebview added the start(icon=...) argument in a later
    # release; on an older build that argument is absent, so fall back to a
    # plain start() instead of failing to open the window at all.
    icon_path = Path(__file__).resolve().parent / "static" / "favicon.png"
    try:
        webview.start(icon=str(icon_path))
    except TypeError:
        webview.start()


def main(argv: Optional[list] = None) -> int:
    open_window(resolve_url(argv))
    return 0


if __name__ == "__main__":
    # Allow running as a plain script from the repository root, not only as a
    # module, by making the package importable first.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    raise SystemExit(main())
