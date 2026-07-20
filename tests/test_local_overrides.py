"""Machine paths stay on the machine: the committed configuration carries none.

These checks exist because an absolute filesystem path is a description of one
computer. It carries the account name of whoever is logged in, their home
directory, and often a drive that only they have plugged in. config/settings.json
is committed, so an absolute path written into it is published to everyone who
reads the repository and stays in the history after the file is cleaned.

The application used to write exactly that. Pointing a model field at a folder
outside the repository, through the Settings tab or through the desktop language
model endpoint, put the operator's account name into a tracked file with no
warning, and normal use reintroduced it after every cleanup.

Absolute paths are still legitimate: a store may live on an external drive and a
model folder may sit outside the repository. So they are not refused, they are
relocated to config/settings.local.json, which version control excludes, and
merged back at load time. The committed file describes the deployment; the local
file describes the machine.

What is checked here:

  - The split is decided by shape, so a path field added later is covered by it.
  - Both operating systems' notions of absolute are recognised, because one
    configuration travels between a Windows desktop and a Raspberry Pi.
  - A relocated path survives a save and reload unchanged.
  - The committed file keeps the value it already published rather than losing it.
  - An override follows its station rather than a position in the list.
  - A save that would still write an absolute path is refused outright.
  - A throwaway scenario configuration inherits nothing from this machine, which
    is what keeps a sandboxed test from being redirected at real captured data.

Run from the repository root:  python tests/test_local_overrides.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PASSED = 0
FAILED = 0

# A path that looks like the leak this work exists to prevent, and its POSIX
# counterpart. Neither names a real file; nothing here touches the filesystem
# outside a temporary directory.
WINDOWS_LEAK = r"C:\Users\someone\Models\example-3b-instruct-q4.gguf"
POSIX_LEAK = "/mnt/external/example_rfdetr.onnx"


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


def _sandbox_repo(tmp: Path) -> Path:
    """A throwaway repository root holding only a copy of the committed settings.

    Nothing is read from or written to the real configuration, the real database,
    or the real data directories at any point in this suite.
    """
    (tmp / "config").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "config" / "settings.json", tmp / "config" / "settings.json")
    return tmp


def main() -> int:
    print("Local overrides: machine paths never reach the committed configuration")
    print("=" * 72)

    from audtheia.config import (
        CANONICAL_SETTINGS_FILENAME,
        LOCAL_OVERRIDES_DEFAULT_PATH,
        _contains_machine_path,
        _is_absolute_path_value,
        _is_path_key,
        collect_absolute_paths,
        load_settings,
        local_overrides_path_for,
        pointer_value,
    )

    # -- what counts as a path, and as absolute --------------------------
    check("a key named path is a path key", _is_path_key("path"))
    check("a key ending in _path is a path key", _is_path_key("labels_path"))
    check("a key ending in _dir is a path key", _is_path_key("detections_visual_dir"))
    check("an unrelated key is not a path key", not _is_path_key("station_name"))
    check("a windows drive path is absolute", _is_absolute_path_value(WINDOWS_LEAK))
    check("a posix root path is absolute", _is_absolute_path_value(POSIX_LEAK))
    check("a windows UNC path is absolute", _is_absolute_path_value(r"\\server\share\model.onnx"))
    check("a repository relative path is not absolute",
          not _is_absolute_path_value("models/visual/porifera_rfdetr.onnx"))
    check("null is not an absolute path", not _is_absolute_path_value(None))
    check("an empty string is not an absolute path", not _is_absolute_path_value(""))

    # The capture source is stored as a source expression rather than as a path,
    # so its key is not a path key and its value is not a bare path. A machine
    # path was published through exactly this field, so it is checked directly.
    check("a machine path buried in a capture source is caught",
          _contains_machine_path('file:"C:\\Users\\someone\\Downloads\\clip.mp3"'))
    check("a UNC path buried in a string is caught",
          _contains_machine_path('file:"\\\\nas\\field\\clip.wav"'))
    check("a source with no machine path is left alone",
          not _contains_machine_path("device:0"))
    check("a relative source is left alone",
          not _contains_machine_path('file:"data/detections/audio/clip.wav"'))
    check("an ordinary sentence is not mistaken for a path",
          not _contains_machine_path("Runs after an observation has been captured"))

    # A URL is not a machine path. The first version of this check matched the
    # "s://" inside "https://", which flagged every citation and every video
    # capture source in the configuration and refused an otherwise valid save.
    check("an https URL is not mistaken for a machine path",
          not _contains_machine_path("https://youtu.be/example"))
    check("an http URL is not mistaken for a machine path",
          not _contains_machine_path("http://example.org/clip.mp4"))
    check("a URL inside a citation is not mistaken for a machine path",
          not _contains_machine_path(r"howpublished = { \url{ https://universe.roboflow.com/x } }"))
    check("an rtsp camera source is not mistaken for a machine path",
          not _contains_machine_path("rtsp://192.168.1.50:554/stream"))
    check("a drive letter is still caught when a URL is in the same string",
          _contains_machine_path('see https://example.org and file:"D:\\clips\\a.wav"'))

    # The committed configuration must be clean as it stands. This is the check
    # that fails loudest if a machine path is ever committed again.
    committed = json.loads((REPO_ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    leaked = collect_absolute_paths(committed)
    check("the committed configuration contains no absolute path", not leaked, repr(leaked))

    check("the local override file is the one version control excludes",
          LOCAL_OVERRIDES_DEFAULT_PATH == "config/settings.local.json")
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    check("the local override file is listed in .gitignore",
          LOCAL_OVERRIDES_DEFAULT_PATH in ignored)

    try:
        from fastapi.testclient import TestClient  # noqa: F401
    except Exception:
        print("  SKIP  the web framework is not installed; the save path is not exercised")
        print("=" * 72)
        print(f"RESULT: {PASSED} passed, {FAILED} failed")
        print("=" * 72)
        return 1 if FAILED else 0

    import audtheia.config as config_module
    from audtheia.app.server import BackendError, _persist_settings

    with tempfile.TemporaryDirectory() as tmp_name:
        root = _sandbox_repo(Path(tmp_name))
        cfg = root / "config" / CANONICAL_SETTINGS_FILENAME
        local = root / "config" / "settings.local.json"

        settings = load_settings(str(cfg))
        published_llm_path = settings.raw["desktop_models"]["llm"]["path"]
        second_station = settings.raw["stations"][1]["station_id"]

        check("the local override path is derived from the repository root",
              local_overrides_path_for(settings.raw, root) == local)
        check("a configuration with no local file loads unchanged",
              not collect_absolute_paths(settings.raw))

        # Reproduce the leak: a model folder outside the repository, set exactly
        # as the language model endpoint and the Settings tab set it.
        settings.raw["desktop_models"]["llm"]["path"] = WINDOWS_LEAK
        settings.raw["stations"][1]["models"]["visual_desktop"]["path"] = POSIX_LEAK
        _persist_settings(settings)

        tracked = json.loads(cfg.read_text(encoding="utf-8"))
        tracked_text = cfg.read_text(encoding="utf-8")
        check("no absolute path reaches the committed file",
              not collect_absolute_paths(tracked), repr(collect_absolute_paths(tracked)))
        check("no account name reaches the committed file", "someone" not in tracked_text)
        check("the committed file keeps the value it already published",
              tracked["desktop_models"]["llm"]["path"] == published_llm_path,
              repr(tracked["desktop_models"]["llm"]["path"]))

        check("the local override file was written", local.exists())
        overrides = json.loads(local.read_text(encoding="utf-8"))["overrides"]
        check("the windows path was relocated, not discarded",
              overrides.get("desktop_models.llm.path") == WINDOWS_LEAK, repr(overrides))
        station_pointer = f"stations[station_id={second_station}].models.visual_desktop.path"
        check("a path inside a station was relocated too",
              overrides.get(station_pointer) == POSIX_LEAK, repr(overrides))
        check("the local file explains itself to whoever opens it",
              "version control" in json.loads(local.read_text(encoding="utf-8")).get("_comment", ""))

        # -- the round trip ----------------------------------------------
        reloaded = load_settings(str(cfg))
        check("the machine path is restored at load time",
              reloaded.raw["desktop_models"]["llm"]["path"] == WINDOWS_LEAK,
              repr(reloaded.raw["desktop_models"]["llm"]["path"]))
        check("a station's machine path is restored at load time",
              reloaded.raw["stations"][1]["models"]["visual_desktop"]["path"] == POSIX_LEAK)
        check("nothing is reported stale on a fresh round trip",
              reloaded.stale_local_overrides == [], repr(reloaded.stale_local_overrides))

        # -- an override belongs to its station, not to a position -------
        reordered = load_settings(str(cfg))
        reordered.raw["stations"].reverse()
        check("an override follows its station when the list is reordered",
              reordered.raw["stations"][0]["models"]["visual_desktop"]["path"] == POSIX_LEAK,
              repr(reordered.raw["stations"][0]["models"]["visual_desktop"]["path"]))

        # -- a pointer naming something gone is inert, not fatal ---------
        stale_doc = json.loads(local.read_text(encoding="utf-8"))
        stale_doc["overrides"]["stations[station_id=removed-station].models.visual_pi.path"] = POSIX_LEAK
        local.write_text(json.dumps(stale_doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        after_stale = load_settings(str(cfg))
        check("a pointer naming a removed station does not stop the load", True)
        check("a stale pointer is reported rather than silently ignored",
              any("removed-station" in p for p in after_stale.stale_local_overrides),
              repr(after_stale.stale_local_overrides))

        # -- a scenario configuration inherits nothing -------------------
        # This is the check that matters most for data safety. The suites that
        # call shutil.rmtree redirect their data directories into a temporary
        # sandbox. If a machine override could reach one of those, a sandboxed
        # directory could be replaced by the real one and captured field data
        # would be deleted.
        scenario = root / "config" / "settings.pi.test.json"
        scenario.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        scenario_settings = load_settings(str(scenario))
        check("a scenario configuration does not inherit this machine's paths",
              scenario_settings.raw["desktop_models"]["llm"]["path"] == published_llm_path,
              repr(scenario_settings.raw["desktop_models"]["llm"]["path"]))
        check("a scenario configuration can opt in explicitly when it means to",
              load_settings(str(scenario), apply_local=True)
              .raw["desktop_models"]["llm"]["path"] == WINDOWS_LEAK)
        check("the canonical configuration can opt out explicitly",
              load_settings(str(cfg), apply_local=False)
              .raw["desktop_models"]["llm"]["path"] == published_llm_path)

        # -- the guard, with the relocation step disabled ----------------
        # A future code path that bypasses the splitter must not be able to
        # write a machine path. Detection is left intact and only the
        # relocation is broken, which is what such a bypass would look like.
        original_apply = config_module.apply_local_overrides
        config_module.apply_local_overrides = lambda raw, overrides: []
        try:
            victim = load_settings(str(cfg))
            victim.raw["desktop_models"]["llm"]["path"] = WINDOWS_LEAK
            try:
                _persist_settings(victim)
                check("a save that would write a machine path is refused", False,
                      "the save was allowed")
            except BackendError as exc:
                check("a save that would write a machine path is refused",
                      "absolute path" in str(exc), str(exc))
                check("the refusal names the offending field",
                      "desktop_models.llm.path" in str(exc), str(exc))
        finally:
            config_module.apply_local_overrides = original_apply

        check("the committed file is unchanged after a refused save",
              not collect_absolute_paths(json.loads(cfg.read_text(encoding="utf-8"))))

        # -- pointer lookup ----------------------------------------------
        check("a pointer resolves to the value it names",
              pointer_value(json.loads(cfg.read_text(encoding="utf-8")),
                            "desktop_models.llm.path") == published_llm_path)
        check("a pointer naming nothing resolves to the default",
              pointer_value({}, "desktop_models.llm.path", "fallback") == "fallback")

    print("=" * 72)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
