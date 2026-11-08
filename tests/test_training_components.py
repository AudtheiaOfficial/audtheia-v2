"""Verification for the optional training components: plan, install, check and remove.

Path: tests/test_training_components.py

pip and the package servers are replaced by stand-ins that answer like the real
ones, so these checks never download PyTorch. They prove that planning changes
nothing on disk, that the PyTorch location comes from the settings through the
hardware probe and is never guessed, that the download size is the sum of the
sizes the servers report, that installing needs an explicit confirmation, that
a failed install is reported and can be removed in one step, and that removal
deletes the environment folder and nothing else, and that a command run in the
training environment finds Audtheia's own code. For sound, checks that the sound
environment installs TensorFlow and the TFLite runtime from the standard index
with no PyTorch choice, and that on Windows a folder too long for the deepest
library files is refused before downloading. One check asks the real pip for
a real dry-run of a tiny package, and is skipped, not failed, without internet.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World  # noqa: E402
from audtheia.training import components as C  # noqa: E402

CHECKS = {"passed": 0, "failed": 0, "skipped": 0}

INDEXES = {
    "nvidia": "https://download.pytorch.org/whl/cu130",
    "nvidia_cuda12": "https://download.pytorch.org/whl/cu126",
    "intel": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
    "apple": "https://pypi.org/simple",
    "amd_linux": "https://download.pytorch.org/whl/rocm7.2",
}

LAPTOP = {"platform": "windows", "machine": "AMD64", "cpu": {"logical_cores": 12},
          "gpus": [{"vendor": "intel", "name": "Intel(R) Iris(R) Xe Graphics"}]}
NVIDIA = {"platform": "windows", "machine": "AMD64", "cpu": {"logical_cores": 16},
          "gpus": [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 4070", "cuda": "13.0", "compute_capability": "8.9"}]}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def skip(label: str, why: str) -> None:
    CHECKS["skipped"] += 1
    print(f"  SKIP  {label} ({why})")


def raises(fn, exc=Exception, status=None) -> bool:
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return status is None or getattr(e, "status", None) == status
    return False


class FakePip:
    """Answers pip dry-runs and the self-test like the real tools."""

    def __init__(self, fail_dry_run=False, selftest_ok=True):
        self.calls = []
        self.fail_dry_run = fail_dry_run
        self.selftest_ok = selftest_ok

    def __call__(self, args, timeout):
        self.calls.append(list(args))
        if "--dry-run" in args:
            if self.fail_dry_run:
                return 1, "", "ERROR: Could not find a version that satisfies the requirement torch"
            report = Path(args[args.index("--report") + 1])
            if "--index-url" not in args and "--extra-index-url" not in args:
                items = [
                    {"metadata": {"name": "tensorflow", "version": "2.21.0"},
                     "download_info": {"url": "https://files.pythonhosted.org/tensorflow-2.21.0-cp313-cp313-win_amd64.whl"}},
                    {"metadata": {"name": "ai-edge-litert", "version": "2.3.0"},
                     "download_info": {"url": "https://files.pythonhosted.org/ai_edge_litert-2.3.0-cp313-cp313-win_amd64.whl"}},
                ]
                report.write_text(json.dumps({"version": "1", "install": items}), encoding="utf-8")
                return 0, "", ""
            index = args[args.index("--index-url") + 1] if "--index-url" in args else \
                args[args.index("--extra-index-url") + 1]
            tag = index.rstrip("/").rsplit("/", 1)[-1]
            items = [
                {"metadata": {"name": "torch", "version": f"2.14.1+{tag}"},
                 "download_info": {"url": f"{index}/torch-2.14.1+{tag}-cp311-cp311-win_amd64.whl"}},
                {"metadata": {"name": "torchvision", "version": f"0.29.1+{tag}"},
                 "download_info": {"url": f"{index}/torchvision-0.29.1+{tag}-cp311-cp311-win_amd64.whl"}},
            ]
            if "--extra-index-url" in args:
                items += [
                    {"metadata": {"name": "nosize", "version": "1.0"},
                     "download_info": {"url": "https://files.pythonhosted.org/nosize-1.0-py3-none-any.whl"}},
                    {"metadata": {"name": "rfdetr", "version": "1.11.2"},
                     "download_info": {"url": "https://files.pythonhosted.org/rfdetr-1.11.2-py3-none-any.whl"}},
                ]
            report.write_text(json.dumps({"version": "1", "install": items}), encoding="utf-8")
            return 0, "", ""
        if "-c" in args:
            if not self.selftest_ok:
                return 1, "", "ImportError: DLL load failed while importing _C"
            if "import tensorflow" in args[args.index("-c") + 1]:
                return 0, 'AUDTHEIA_SELFTEST {"tensorflow": "2.21.0", "device": "cpu", "probe_seconds": 0.03}\n', ""
            return 0, 'AUDTHEIA_SELFTEST {"torch": "2.14.1+cpu", "device": "cpu", "probe_seconds": 0.02}\n', ""
        return 1, "", "unexpected"


SIZES = {"torch-2.14.1": 130_000_000, "torchvision-0.29.1": 2_000_000, "rfdetr-1.11.2": 900_000,
         "tensorflow-2.21.0": 351_205_371, "ai_edge_litert-2.3.0": 15_000_000}


def sizer(url):
    for key, size in SIZES.items():
        if key in url:
            return size
    return None


class FakeProc:
    def __init__(self, lines, code):
        self.stdout = iter(lines)
        self._code = code

    def wait(self):
        return self._code


class FakePopen:
    """Imitates venv creation and pip install. fail_on names a step that fails."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, args, **kw):
        self.calls.append(list(args))
        if "venv" in args:
            env = Path(args[-1])
            py = C.env_python(env)
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("", encoding="utf-8")
            (env / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
            return FakeProc(["created\n"], 1 if self.fail_on == "venv" else 0)
        if "--index-url" in args:
            return FakeProc(["Collecting torch\n", "Installing collected packages: torch\n"],
                            1 if self.fail_on == "torch" else 0)
        return FakeProc(["Successfully installed rfdetr\n"], 1 if self.fail_on == "rest" else 0)


class World5(World):
    def __init__(self, root, indexes=None):
        super().__init__(root)
        base = json.loads(self.settings_path.read_text(encoding="utf-8"))
        base.setdefault("training", {}).setdefault("install", {})["torch_index_by_vendor"] = dict(indexes or INDEXES)
        self.settings_path.write_text(json.dumps(base), encoding="utf-8")
        from audtheia.config import load_settings  # noqa: PLC0415
        self.settings = load_settings(self.settings_path)

    def components(self, pip=None, popen=None, windows=False, long_paths=lambda: False, cluster=None):
        extra = {"cluster": cluster} if cluster is not None else {}
        return C.Components(self.settings, base_python="/usr/bin/python-base", runner=pip or FakePip(),
                            sizer=sizer, popen=popen or FakePopen(), windows=windows, long_paths=long_paths, **extra)


def snapshot(root: Path) -> list:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def test_plan():
    print("planning")
    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        comp = w.components()
        envs = Path(w.settings.training_paths()["envs_dir"])
        before = snapshot(w.repo)
        plan = comp.plan("vision_detector", LAPTOP)
        check("planning writes nothing", snapshot(w.repo) == before and not envs.exists())
        check("the laptop gets the processor build from the settings",
              plan["vendor_key"] == "cpu" and plan["index_url"] == INDEXES["cpu"] and plan["device"] == "cpu")
        check("PyTorch is pinned to exactly what the location offers",
              plan["torch_pins"] == ["torch==2.14.1+cpu", "torchvision==0.29.1+cpu"])
        check("the training libraries come from the settings", plan["packages"] == ["rfdetr[train,onnx]==1.12.0"])
        check("the download size is the sum of the sizes the servers report",
              plan["download_bytes"] == 130_000_000 + 2_000_000 + 900_000)
        check("a file whose size the server did not report is counted and said",
              plan["files_without_size"] == 1 and len(plan["files"]) == 4)
        check("the plan states that the internet is needed once", "Once" in plan["internet_needed"])
        pip_calls = comp.runner.calls
        check("pip was only ever asked for a dry run", all("--dry-run" in c for c in pip_calls) and len(pip_calls) == 2)
        check("the first dry run reads PyTorch only from the chosen location",
              pip_calls[0][pip_calls[0].index("--index-url") + 1] == INDEXES["cpu"])
        check("the second keeps the standard index and adds the PyTorch location",
              "--extra-index-url" in pip_calls[1] and "torch==2.14.1+cpu" in pip_calls[1]
              and "rfdetr[train,onnx]==1.12.0" in pip_calls[1])
        plan_gpu = w.components().plan("vision_detector", NVIDIA)
        check("an NVIDIA card gets the CUDA 13 build", plan_gpu["index_url"].endswith("cu130")
              and plan_gpu["torch_pins"][0] == "torch==2.14.1+cu130")
        check("an engine without components is refused", raises(lambda: comp.plan("selftest", LAPTOP), C.ComponentError, 404))
        failing = w.components(pip=FakePip(fail_dry_run=True))
        check("pip failing to resolve gives a plain message naming the internet",
              raises(lambda: failing.plan("vision_detector", LAPTOP), C.ComponentError, 502))
    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td), indexes=dict(INDEXES, cpu=None))
        check("a download location left empty stops the plan and is never guessed",
              raises(lambda: w.components().plan("vision_detector", LAPTOP), C.ComponentError, 422))


def test_cluster_size():
    print("drives with large allocation units")
    check("the allocation unit of a real folder is read", (C.cluster_bytes(Path(__file__).parent) or 0) >= 512)
    check("a folder not created yet is answered for the drive it would be on",
          C.cluster_bytes(Path(__file__).parent / "not" / "yet" / "made") == C.cluster_bytes(Path(__file__).parent))
    check("a usual 4 KB unit gives no warning", C.cluster_note(4096, "vision") is None)
    check("the threshold itself gives no warning", C.cluster_note(C.LARGE_CLUSTER_BYTES, "vision") is None)
    check("an unreadable unit gives no warning", C.cluster_note(None, "vision") is None)
    note = C.cluster_note(1024 * 1024, "vision")
    check("a 1 MB unit is named in plain words with its cost and the fix",
          note is not None and "blocks of 1 MB" in note and "about 41 GB" in note and "envs_dir" in note)
    check("a 128 KB unit is named in kilobytes", "blocks of 128 KB" in C.cluster_note(128 * 1024, "acoustic"))
    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        seen = []
        plan = w.components(cluster=lambda p: seen.append(Path(p)) or 1024 * 1024).plan("vision_detector", LAPTOP)
        check("the plan carries the warning before anything is downloaded",
              plan["cluster_bytes"] == 1024 * 1024 and plan["cluster_warning"] == C.cluster_note(1024 * 1024, "vision"))
        check("the drive asked is the one the environments go to",
              seen == [Path(w.settings.training_paths()["envs_dir"])])
        plan = w.components(cluster=lambda p: 4096).plan("acoustic_head", LAPTOP)
        check("a usual drive gives a plan with no warning", plan["cluster_warning"] is None and plan["cluster_bytes"] == 4096)
        comp = w.components(cluster=lambda p: 256 * 1024 * 1024)
        env = comp.env_dir("acoustic_head")
        env.mkdir(parents=True)
        (env / "a.txt").write_bytes(b"x" * 10)
        (env / "b.txt").write_bytes(b"x" * 1500)
        check("the real size counts the bytes", comp.folder_size_gb("acoustic_head") == 0.0)
        check("the size on disk counts one whole block for each small file",
              comp.folder_size_gb("acoustic_head", on_disk=True) == 0.5)


def test_install_and_remove():
    print("installing and removing")
    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        popen = FakePopen()
        comp = w.components(popen=popen)
        plan = comp.plan("vision_detector", LAPTOP)
        check("nothing is installed without confirmation",
              raises(lambda: comp.start_install("vision_detector", plan, confirmed=False), C.ComponentError, 400))
        check("a plan for another engine is refused",
              raises(lambda: comp.start_install("vision_detector", dict(plan, engine="x"), confirmed=True), C.ComponentError))
        check("before installing the status is not installed", comp.status("vision_detector")["state"] == "not_installed")
        comp.start_install("vision_detector", plan, confirmed=True)
        job = C.wait_for_job(comp, "vision_detector", 20)
        check("the install completes", job["state"] == "completed" and job["message"] == "Installed and checked.")
        st = comp.status("vision_detector")
        check("the components are then ready with what was installed and the self-test",
              st["state"] == "ready" and st["component"]["torch_pins"] == plan["torch_pins"]
              and st["component"]["self_test"]["device"] == "cpu")
        env = comp.env_dir("vision_detector")
        check("the environment is under the training environments folder",
              env.parent == Path(w.settings.training_paths()["envs_dir"]) and env.name == "vision")
        check("the in-progress marker is gone", not (env / C.MARKER_INSTALLING).exists())
        steps = popen.calls
        check("the environment is created from this Python with venv", steps[0][1:3] == ["-m", "venv"])
        check("PyTorch is installed from the chosen location with the exact pins",
              steps[1][steps[1].index("--index-url") + 1] == INDEXES["cpu"] and steps[1][-2:] == plan["torch_pins"])
        check("then the libraries, keeping PyTorch pinned",
              "--extra-index-url" in steps[2] and plan["torch_pins"][0] in steps[2] and steps[2][-1] == "rfdetr[train,onnx]==1.12.0")
        check("every step runs inside the new environment, never the application's",
              all(s[0] == str(C.env_python(env)) for s in steps[1:]))
        log = comp.log_path("vision_detector").read_text(encoding="utf-8")
        check("the install log keeps pip's output", "Successfully installed rfdetr" in log and "Installed and checked." in log)
        check("installing twice is refused", raises(lambda: comp.start_install("vision_detector", plan, confirmed=True),
                                                    C.ComponentError, 409))
        check("removal is refused while a run uses the components",
              raises(lambda: comp.remove("vision_detector", run_active=True), C.ComponentError, 409))
        sibling = env.parent / "keep-me.txt"
        sibling.write_text("x", encoding="utf-8")
        res = comp.remove("vision_detector")
        check("removal deletes the environment folder", res["removed"] and not env.exists())
        check("and nothing beside it", sibling.read_text(encoding="utf-8") == "x")
        check("afterwards the status is not installed again", comp.status("vision_detector")["state"] == "not_installed")
        check("removing what is not there is harmless", comp.remove("vision_detector")["removed"] is False)

    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        comp = w.components(popen=FakePopen(fail_on="torch"))
        comp.start_install("vision_detector", comp.plan("vision_detector", LAPTOP), confirmed=True)
        job = C.wait_for_job(comp, "vision_detector", 20)
        check("a failed PyTorch install is reported in plain words", job["state"] == "failed"
              and "PyTorch did not install" in job["message"])
        check("and leaves an incomplete environment", comp.status("vision_detector")["state"] == "incomplete")
        check("which must be removed before trying again",
              raises(lambda: comp.start_install("vision_detector", comp.plan("vision_detector", LAPTOP), confirmed=True),
                     C.ComponentError, 409))
        check("one removal clears it", comp.remove("vision_detector")["removed"]
              and comp.status("vision_detector")["state"] == "not_installed")

    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        comp = w.components(pip=FakePip(selftest_ok=False))
        comp.start_install("vision_detector", comp.plan("vision_detector", LAPTOP), confirmed=True)
        job = C.wait_for_job(comp, "vision_detector", 20)
        check("libraries that install but do not load are not marked ready",
              job["state"] == "failed" and "did not pass the check" in job["message"]
              and "DLL load failed" in job["message"] and comp.status("vision_detector")["state"] == "incomplete")

    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        comp = w.components()
        foreign = comp.env_dir("vision_detector")
        foreign.mkdir(parents=True)
        (foreign / "thesis.docx").write_text("precious", encoding="utf-8")
        check("a folder that is not a training environment is never deleted",
              raises(lambda: comp.remove("vision_detector"), C.ComponentError, 409)
              and (foreign / "thesis.docx").read_text(encoding="utf-8") == "precious")


def test_sound_components():
    print("the sound training components")
    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        popen = FakePopen()
        comp = w.components(popen=popen)
        plan = comp.plan("acoustic_head", LAPTOP)
        check("the sound engine installs into its own environment", comp.env_dir("acoustic_head").name == "acoustic"
              and comp.environment("acoustic_head") == "acoustic")
        check("its packages come from the settings and no PyTorch build is chosen",
              plan["packages"] == ["tensorflow", "ai-edge-litert"] and plan["torch_pins"] == []
              and plan["index_url"] is None and plan["device"] == "cpu")
        calls = comp.runner.calls
        check("pip is asked once, against the standard index only", len(calls) == 1
              and "--index-url" not in calls[0] and "--extra-index-url" not in calls[0] and calls[0][-2:] == plan["packages"])
        check("the download size is the sum of the sizes the servers report",
              plan["download_bytes"] == 351_205_371 + 15_000_000 and plan["files_without_size"] == 0)
        comp.start_install("acoustic_head", plan, confirmed=True)
        job = C.wait_for_job(comp, "acoustic_head", 20)
        st = comp.status("acoustic_head")
        check("the install completes and the TensorFlow check is recorded", job["state"] == "completed"
              and st["state"] == "ready" and st["component"]["self_test"]["tensorflow"] == "2.21.0")
        steps = popen.calls
        check("the environment is created, then the libraries installed from the standard index",
              steps[0][1:3] == ["-m", "venv"] and len(steps) == 2 and "--index-url" not in steps[1]
              and "--extra-index-url" not in steps[1] and steps[1][-2:] == ["tensorflow", "ai-edge-litert"])
        snippet = [c for c in comp.runner.calls if "-c" in c][0]
        check("the check imports TensorFlow and the TFLite runtime by their module names",
              "import tensorflow" in snippet[snippet.index("-c") + 1] and snippet[-2:] == ["tensorflow", "ai_edge_litert"])
        check("the image components are untouched", comp.status("vision_detector")["state"] == "not_installed")
        check("a sound plan without its packages is refused",
              raises(lambda: comp.start_install("acoustic_head", dict(plan, packages=[]), confirmed=True),
                     C.ComponentError, 400))

    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        long_root = Path(td) / ("x" * 120)
        raw = json.loads(w.settings_path.read_text(encoding="utf-8"))
        raw["training"].setdefault("paths", {})["envs_dir"] = str(long_root)
        w.settings_path.write_text(json.dumps(raw), encoding="utf-8")
        from audtheia.config import load_settings  # noqa: PLC0415
        w.settings = load_settings(w.settings_path)
        try:
            w.components(windows=True).plan("acoustic_head", LAPTOP)
            reason = ""
        except C.ComponentError as exc:
            reason = str(exc)
        check("on Windows a folder too long for the deepest library files is refused before downloading, with the fix",
              "260" in reason and "long paths" in reason and "envs_dir" in reason)
        check("the same folder is accepted when Windows long paths are switched on",
              w.components(windows=True, long_paths=lambda: True).plan("acoustic_head", LAPTOP)["packages"])
        check("and off Windows the length does not matter",
              w.components(windows=False).plan("acoustic_head", LAPTOP)["packages"])
    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        short = w.components(windows=True)
        room = C.WINDOWS_PATH_LIMIT - len(str(short.env_dir("vision_detector") / "Lib" / "site-packages")) - 1
        check("a folder with enough room is accepted on Windows",
              room < C.DEEPEST_PACKAGE_PATH or short.plan("vision_detector", LAPTOP)["torch_pins"])


def test_command_search_path():
    print("commands in the training environment")
    probe = [sys.executable, "-c", "import os; print(os.environ.get('PYTHONPATH', ''))"]
    code, out, _ = C.run_capture(probe, timeout=60, pythonpath="/audtheia/code")
    check("the code folder is put first on the command's module search path",
          code == 0 and out.strip().split(os.pathsep)[0] == "/audtheia/code")
    code, out, _ = C.run_capture([sys.executable, "-c", "print(1)"], timeout=60)
    check("without it the command runs unchanged", code == 0 and out.strip() == "1")
    code, out, err = C.run_capture(["/no/such/program"], timeout=5)
    check("a command that cannot start is reported, never raised", code is None and err)


def test_real_pip():
    print("a real dry run")
    base = sys.executable
    code, _out, _err = C.run_capture([base, "-m", "pip", "--version"], timeout=60)
    if code != 0:
        skip("a real pip dry run", "pip is not available in this interpreter")
        return
    with tempfile.TemporaryDirectory() as td:
        w = World5(Path(td))
        comp = C.Components(w.settings, base_python=base)
        try:
            items = comp._dry_run(["--index-url", "https://pypi.org/simple", "six==1.17.0"], "a test package")
        except C.ComponentError as exc:
            skip("a real pip dry run", f"no internet or index: {str(exc)[:80]}")
            return
        six = [i for i in items if (i["name"] or "").lower() == "six"]
        check("the real pip report is read", six and six[0]["version"] == "1.17.0" and six[0]["url"].startswith("https://"))
        size = C.content_length(six[0]["url"])
        if size is None:
            skip("the real file size", "the package host did not answer a size request")
        else:
            check("the real file size is read from the package host", 5_000 < size < 100_000)
    check("an unreachable address gives no size", C.content_length("https://audtheia.invalid/x.whl", timeout=3) is None)


def main() -> int:
    test_plan()
    test_cluster_size()
    test_install_and_remove()
    test_sound_components()
    test_command_search_path()
    test_real_pip()
    if CHECKS["skipped"]:
        print(f"\nSkipped {CHECKS['skipped']} check(s) that need the internet.")
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
