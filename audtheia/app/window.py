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

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("audtheia.app.window")

# The window title and a comfortable default size for a data interface. These are
# presentation constants for the desktop frame, not deployment settings, so they
# live here rather than in the configuration.
WINDOW_TITLE = "Audtheia"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 820
MIN_WIDTH = 900
MIN_HEIGHT = 600

# The identity Windows uses to group this program's taskbar button and to choose
# its icon. Without one, Windows treats the window as belonging to the Python
# interpreter that runs it and shows the interpreter's own logo on the taskbar,
# whatever icon the window itself carries. It is a fixed, non-secret label.
WINDOWS_APP_ID = "Audtheia.Desktop"

# How long to wait for the window to appear on screen before giving up on
# setting its icon by hand, and how often to look.
ICON_WAIT_SECONDS = 20.0
ICON_POLL_SECONDS = 0.25


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


def _set_windows_app_identity() -> None:
    """Give this process its own taskbar identity on Windows, so the taskbar and
    its overflow menu show the Audtheia icon instead of the Python interpreter's.

    This has to happen before the first window is created. On any other platform,
    or if Windows declines, it does nothing, so the window opens exactly as it did
    before.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:  # noqa: BLE001 - a missing identity only costs the taskbar icon
        logger.debug("could not set the Windows app identity", exc_info=True)


def _find_own_window(user32, wintypes, ctypes):
    """The handle of this process's visible, top-level, titled window, or None.

    The web view draws inside child windows, so only visible windows that have no
    owner and belong to this process are considered, and one titled exactly as the
    app window is preferred.
    """
    found = []
    pid = os.getpid()
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _visit(hwnd, _lparam):
        owner_pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
        if owner_pid.value != pid or not user32.IsWindowVisible(hwnd):
            return True
        if user32.GetWindow(hwnd, 4):  # GW_OWNER: skip owned windows such as popups
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        found.append((hwnd, buffer.value))
        return True

    user32.EnumWindows(enum_proc(_visit), 0)
    for hwnd, title in found:
        if title == WINDOW_TITLE:
            return hwnd
    return found[0][0] if found else None


def _apply_windows_window_icon(icon_file: Path) -> None:
    """Set the app window's icon by hand on Windows, once the window is on screen.

    The window frame is asked to carry the icon when it is created, but the
    taskbar button and the switcher read the icon from the window itself, so the
    small and large icons are set directly through the operating system as well.
    It runs on a background thread after the window loop starts, waits for the
    window, and quietly does nothing if anything is unavailable.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [
            wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
            ctypes.c_int, ctypes.c_int, wintypes.UINT,
        ]
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetWindow.restype = wintypes.HWND
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

        deadline = time.monotonic() + ICON_WAIT_SECONDS
        hwnd = None
        while time.monotonic() < deadline:
            hwnd = _find_own_window(user32, wintypes, ctypes)
            if hwnd:
                break
            time.sleep(ICON_POLL_SECONDS)
        if not hwnd:
            logger.debug("the app window did not appear in time to set its icon")
            return

        image_icon, load_from_file = 1, 0x0010
        wm_seticon, icon_small, icon_big = 0x0080, 0, 1
        small = (user32.GetSystemMetrics(49), user32.GetSystemMetrics(50))  # SM_CXSMICON, SM_CYSMICON
        big = (user32.GetSystemMetrics(11), user32.GetSystemMetrics(12))    # SM_CXICON, SM_CYICON
        for which, (width, height) in ((icon_small, small), (icon_big, big)):
            handle = user32.LoadImageW(None, str(icon_file), image_icon, width, height, load_from_file)
            if handle:
                # The loaded icon is kept for the life of the process on purpose:
                # the window holds it, and the process ends when the window closes.
                user32.PostMessageW(hwnd, wm_seticon, which, int(handle))
    except Exception:  # noqa: BLE001 - the icon is cosmetic and must never stop the window
        logger.debug("could not set the window icon", exc_info=True)


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

    _set_windows_app_identity()

    webview.create_window(
        WINDOW_TITLE,
        url,
        width=width,
        height=height,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
    )

    # Brand the window with the Audtheia mark rather than the interpreter's own
    # icon. The accepted image format differs by platform, so we pass the file
    # that each backend can actually load, which avoids the earlier crash from
    # handing Windows a .png it cannot read as an icon:
    #   - Windows expects a .ico.
    #   - Linux (GTK/QT) takes a .png.
    #   - macOS sets its icon from an application bundle, not at runtime, so we
    #     pass nothing there and the icon is supplied when the app is bundled.
    # On Windows the process also takes its own taskbar identity and sets the
    # window's small and large icons directly (see the two helpers above), because
    # the taskbar otherwise shows the interpreter's icon.
    # We only pass the argument when a matching file exists, and older pywebview
    # builds predate start(icon=...), so a TypeError falls back to plain start().
    static_dir = Path(__file__).resolve().parent / "static"
    if sys.platform.startswith("win"):
        icon_file = static_dir / "favicon.ico"
    elif sys.platform == "darwin":
        icon_file = None
    else:
        icon_file = static_dir / "favicon.png"

    start_kwargs = {"icon": str(icon_file)} if icon_file and icon_file.exists() else {}
    if sys.platform.startswith("win") and icon_file and icon_file.exists():
        # Runs on its own thread once the window loop is up, so the taskbar icon
        # is set even where the web view backend ignores the icon argument.
        start_kwargs["func"] = _apply_windows_window_icon
        start_kwargs["args"] = (icon_file,)
    try:
        webview.start(**start_kwargs)
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
