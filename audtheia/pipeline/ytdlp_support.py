"""Reading web-page video and audio addresses through yt-dlp.

Path: audtheia/pipeline/ytdlp_support.py

A YouTube or live-camera page is not a media stream that OpenCV can open, so a
desktop capture resolves such a page to its direct stream address with yt-dlp
first. That step depends on three things being present in the application's own
Python environment: the yt-dlp package, its YouTube challenge solver
(yt-dlp-ejs), and a JavaScript runtime (Deno) that yt-dlp runs the solver in.
The `yt-dlp[default,deno]` requirement installs all three. Sites also change
often, so an older yt-dlp can stop working on a page that worked last month.

This module holds the shared, dependency-light part: it inspects the
environment and turns a yt-dlp failure into one message that names the real
cause and the exact command that fixes it, instead of a generic "could not
open" that hides the reason. It imports nothing heavy, so both the video and the
audio drivers can use it, and it loads without yt-dlp installed.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
from typing import Optional

__all__ = [
    "INSTALL_SPEC",
    "update_command",
    "environment_report",
    "describe_failure",
    "missing_package_message",
]

# The requirement that gives yt-dlp everything it needs for YouTube: the package,
# its default extras (including the challenge solver), and a pip-installable
# Deno runtime for Windows, macOS, and Linux on x86-64 and ARM.
INSTALL_SPEC = "yt-dlp[default,deno]"

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MAX_REASON = 400


def _python_command() -> str:
    """The interpreter to run for a fix, relative to the working folder when it
    lives there (the usual `.venv`), so the command is short and copyable."""
    exe = Path(sys.executable)
    try:
        rel = exe.relative_to(Path.cwd())
        prefix = ".\\" if os.name == "nt" else "./"
        return prefix + str(rel)
    except ValueError:
        text = str(exe)
        return f'"{text}"' if " " in text else text


def update_command() -> str:
    """The one command that installs or updates yt-dlp with its YouTube parts,
    run from the Audtheia folder so it targets the application's environment."""
    return f'{_python_command()} -m pip install -U "{INSTALL_SPEC}"'


def _deno_available() -> bool:
    """Whether yt-dlp can find a Deno runtime, checked the way yt-dlp itself
    looks: its own runtime probe when present, then the environment's scripts
    folder and PATH."""
    try:  # noqa: SIM105 - a private yt-dlp helper, so any failure just falls through
        from yt_dlp.utils._jsruntime import DenoJsRuntime  # type: ignore

        info = DenoJsRuntime().info
        if info is not None:
            return bool(getattr(info, "supported", True))
        return False
    except Exception:  # noqa: BLE001
        pass
    scripts = sysconfig.get_path("scripts")
    suffix = sysconfig.get_config_var("EXE") or ""
    if scripts and Path(scripts, "deno" + suffix).exists():
        return True
    return shutil.which("deno") is not None


def environment_report() -> dict:
    """What the application's environment has for reading web-page streams."""
    version: Optional[str]
    try:
        version = metadata.version("yt-dlp")
    except metadata.PackageNotFoundError:
        version = None
    return {
        "version": version,
        "solver": importlib.util.find_spec("yt_dlp_ejs") is not None,
        "runtime": _deno_available() if version else False,
    }


def missing_package_message(what: str = "a web-page address") -> str:
    """The message for yt-dlp not being installed at all."""
    return (
        f"reading {what} needs yt-dlp, which is not installed in Audtheia's "
        f"environment. From the Audtheia folder run: {update_command()}"
    )


def _clean_reason(exc: BaseException) -> str:
    text = _ANSI.sub("", str(exc)).strip()
    text = re.sub(r"^ERROR:\s*", "", text)
    text = " ".join(text.split())
    if len(text) > _MAX_REASON:
        text = text[: _MAX_REASON - 3].rstrip() + "..."
    return text or exc.__class__.__name__


def describe_failure(exc: BaseException, address: str, *, what: str = "video") -> str:
    """One message naming why yt-dlp could not read `address`, and the fix.

    The wording depends on what is actually wrong: an address yt-dlp does not
    recognise or cannot reach, a missing package, a missing YouTube challenge
    solver or JavaScript runtime, or, when all of those are present, a yt-dlp
    that has fallen behind the site.
    """
    reason = _clean_reason(exc)
    low = reason.lower()
    head = f"yt-dlp could not read the {what} address {address!r}: {reason}."

    if "unsupported url" in low or "[generic]" in low:
        # yt-dlp fell back to its generic reader, meaning the address is not a
        # page from a site it knows, so updating it would not help.
        return (
            f"{head} It is not a video page yt-dlp knows how to read, or it could "
            f"not be reached. Check that it is spelled correctly and reachable. If "
            f"it is a direct camera or stream address, set it as url:{address} instead."
        )

    report = environment_report()
    if report["version"] is None:
        return f"{head} {missing_package_message()}"

    gaps = []
    if not report["solver"]:
        gaps.append("its YouTube challenge solver (yt-dlp-ejs) is missing")
    if not report["runtime"]:
        gaps.append("no JavaScript runtime (Deno) is available to it")
    if gaps:
        return (
            f"{head} yt-dlp {report['version']} is installed, but "
            f"{' and '.join(gaps)}, and YouTube cannot be read without them. "
            f"From the Audtheia folder run: {update_command()}"
        )

    return (
        f"{head} yt-dlp {report['version']} and its YouTube components are "
        f"installed, so the site has most likely changed since this version. "
        f"Update it from the Audtheia folder with: {update_command()}. If it still "
        f"fails, the video may be private, age-restricted, region-locked, or "
        f"require a signed-in account, which yt-dlp cannot read here."
    )
