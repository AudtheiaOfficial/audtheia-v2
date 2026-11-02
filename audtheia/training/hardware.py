"""Hardware probe, safety limits and resource plans for training.

Path: audtheia/training/hardware.py

Reads what the computer has and what state it is in, without changing anything:
processor, memory, free disk, whether it is on mains power and how charged the
battery is, any graphics card, and, where the operating system exposes one, a
sign that the processor or graphics card is slowing itself down to stay cool.
From that it decides which PyTorch build suits the computer, how many processor
threads each training mode may use, and whether a run may start or must pause.

Every probe is wrapped so a missing tool or an unexpected answer gives "not
known" rather than an error, and "not known" is always reported as such, never
as safe. Standard library only, so it runs in the application and in every
training environment.

Nothing here can harm the computer. It only reads. The decisions it returns are
carried out by the run control, which pauses a run with its progress saved.
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

GIB = 1024 ** 3

# CUDA 13 builds of PyTorch support NVIDIA GPUs of compute capability 7.5 and
# newer. Older cards use the CUDA 12 build. Source: PyTorch 2.14 release notes.
CUDA13_MIN_COMPUTE_CAPABILITY = 7.5

# The CUDA version each NVIDIA build needs from the driver, used only when the
# download location does not name it (a mirror, for example). An official
# location such as .../whl/cu130 names it and that name is used instead.
CUDA_BUILD_FALLBACK = {"nvidia": (13, 0), "nvidia_cuda12": (12, 6)}

# Fixed fields every training mode must share. Gentle may change only how the
# work is divided, never what is learned.
EQUIVALENCE_FIELDS = (
    "seed", "epochs", "effective_batch", "resolution", "base_model", "precision",
    "learning_rate", "encoder_learning_rate", "weight_decay", "use_ema",
)

LIMIT_MESSAGES = {
    "on_battery": (
        "Paused to protect your laptop: the charger is not connected. "
        "Connect the charger and press Resume."
    ),
    "battery_low": (
        "Paused to protect your laptop: the battery is below {limit:g} percent. "
        "Connect the charger and press Resume."
    ),
    "disk_low": (
        "Stopped to protect your disk: less than {limit:g} GB free. Free some "
        "space and press Resume."
    ),
    "thermal": (
        "Paused to let the computer cool down: it has been slowing itself down because of heat for "
        "{limit:g} seconds. Press Resume when it has cooled."
    ),
    "gpu_memory": (
        "Paused because the graphics card is almost out of memory. Close other "
        "programs that use the graphics card, or choose Gentle, then press Resume."
    ),
    "power_unknown": (
        "Audtheia cannot tell whether the charger is connected on this computer, so it cannot pause "
        "automatically when the charger is removed."
    ),
    "thermal_unknown": (
        "This computer does not report when it slows down because of heat, so Audtheia cannot pause "
        "automatically for heat. Gentle mode limits how hard the processor works."
    ),
    "cpu_only_long": (
        "Training on the processor alone is estimated to take about {hours:.1f} hours. Gentle with the "
        "Nano or Small size is recommended. You can pause and resume at any time."
    ),
}


class HardwareError(ValueError):
    """A plan or a configuration that cannot be used, with a plain explanation."""


# ---------------------------------------------------------------------------
# Command and file access, replaceable in tests
# ---------------------------------------------------------------------------


def run_command(args: Sequence[str], timeout: float = 8.0) -> tuple:
    """Run a read-only system command. Returns (return code, standard output).

    Never raises: a missing program, a timeout or any error gives (None, "").
    On Windows no console window is shown.
    """
    kwargs: dict = {"capture_output": True, "text": True, "timeout": timeout}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(list(args), **kwargs)  # noqa: S603 - fixed argument lists only
    except Exception:  # noqa: BLE001
        return None, ""
    return done.returncode, done.stdout or ""


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


class SystemSources:
    """Where the probe reads from. Tests replace these methods with fixed answers."""

    def platform(self) -> str:
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform == "darwin":
            return "darwin"
        return "linux"

    def machine(self) -> str:
        return platform.machine() or ""

    def python(self) -> str:
        return platform.python_version()

    def cpu_count(self) -> int | None:
        return os.cpu_count()

    def run(self, args: Sequence[str], timeout: float = 8.0) -> tuple:
        return run_command(args, timeout)

    def read(self, path: str) -> str | None:
        return _read(path)

    def glob(self, pattern: str) -> list:
        root = Path("/")
        try:
            return sorted(str(p) for p in root.glob(pattern.lstrip("/")))
        except Exception:  # noqa: BLE001
            return []

    def disk_usage(self, path: str) -> tuple | None:
        try:
            target = Path(path)
            while not target.exists() and target.parent != target:
                target = target.parent
            usage = shutil.disk_usage(str(target))
            return usage.total, usage.free
        except Exception:  # noqa: BLE001
            return None

    def windows_cpu_name(self) -> str | None:
        try:
            import winreg  # noqa: PLC0415
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except Exception:  # noqa: BLE001
            return None

    def windows_memory(self) -> tuple | None:
        try:
            import ctypes  # noqa: PLC0415

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:  # noqa: BLE001
            return None

    def windows_power(self) -> tuple | None:
        """(AC line status, battery flag, battery percent) from GetSystemPowerStatus."""
        try:
            import ctypes  # noqa: PLC0415

            class PowerStatus(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            status = PowerStatus()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                return None
            return int(status.ACLineStatus), int(status.BatteryFlag), int(status.BatteryLifePercent)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def _probe_cpu(src: SystemSources, plat: str) -> dict:
    name = None
    if plat == "windows":
        name = src.windows_cpu_name()
    elif plat == "linux":
        text = src.read("/proc/cpuinfo") or ""
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        name = m.group(1).strip() if m else None
    elif plat == "darwin":
        code, out = src.run(["sysctl", "-n", "machdep.cpu.brand_string"])
        name = out.strip() if code == 0 and out.strip() else None
    cores = src.cpu_count()
    return {"name": name, "logical_cores": int(cores) if cores else None}


def _probe_memory(src: SystemSources, plat: str) -> dict:
    total = available = None
    source = None
    if plat == "windows":
        got = src.windows_memory()
        if got:
            total, available = got
            source = "GlobalMemoryStatusEx"
    elif plat == "linux":
        text = src.read("/proc/meminfo") or ""
        t = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
        a = re.search(r"^MemAvailable:\s+(\d+)\s*kB", text, re.MULTILINE)
        if t:
            total = int(t.group(1)) * 1024
            available = int(a.group(1)) * 1024 if a else None
            source = "/proc/meminfo"
    elif plat == "darwin":
        code, out = src.run(["sysctl", "-n", "hw.memsize"])
        if code == 0 and out.strip().isdigit():
            total = int(out.strip())
            source = "sysctl"
    return {
        "total_gb": round(total / GIB, 2) if total else None,
        "available_gb": round(available / GIB, 2) if available else None,
        "source": source,
    }


def _probe_disk(src: SystemSources, path: str) -> dict:
    got = src.disk_usage(path)
    if not got:
        return {"path": str(path), "total_gb": None, "free_gb": None}
    total, free = got
    return {"path": str(path), "total_gb": round(total / GIB, 2), "free_gb": round(free / GIB, 2)}


def _probe_power(src: SystemSources, plat: str) -> dict:
    """on_ac True, False or None for not known. has_battery likewise."""
    out = {"on_ac": None, "has_battery": None, "battery_percent": None, "source": None}
    if plat == "windows":
        got = src.windows_power()
        if got is None:
            return out
        ac, flag, percent = got
        out["source"] = "GetSystemPowerStatus"
        out["has_battery"] = None if flag == 255 else not (flag & 128)
        out["on_ac"] = {0: False, 1: True}.get(ac)
        if out["has_battery"] is False and out["on_ac"] is None:
            out["on_ac"] = True
        if out["has_battery"] and percent != 255:
            out["battery_percent"] = float(percent)
        return out
    if plat == "linux":
        supplies = src.glob("/sys/class/power_supply/*")
        if not supplies:
            return out
        out["source"] = "/sys/class/power_supply"
        mains_online = []
        batteries = []
        for sup in supplies:
            kind = (src.read(f"{sup}/type") or "").strip()
            if kind == "Mains":
                online = (src.read(f"{sup}/online") or "").strip()
                if online in ("0", "1"):
                    mains_online.append(online == "1")
            elif kind == "Battery":
                if (src.read(f"{sup}/scope") or "").strip() == "Device":
                    continue                       # a mouse or keyboard battery
                cap = (src.read(f"{sup}/capacity") or "").strip()
                status = (src.read(f"{sup}/status") or "").strip()
                batteries.append((float(cap) if cap.isdigit() else None, status))
        out["has_battery"] = bool(batteries)
        if mains_online:
            out["on_ac"] = any(mains_online)
        elif batteries:
            statuses = {s for _, s in batteries}
            if statuses & {"Charging", "Full", "Not charging"}:
                out["on_ac"] = True
            elif "Discharging" in statuses:
                out["on_ac"] = False
        else:
            out["on_ac"] = True                       # no battery at all: a desktop
        caps = [c for c, _ in batteries if c is not None]
        if caps:
            out["battery_percent"] = min(caps)
        return out
    if plat == "darwin":
        code, text = src.run(["pmset", "-g", "batt"])
        if code != 0 or not text:
            return out
        out["source"] = "pmset"
        if "AC Power" in text:
            out["on_ac"] = True
        elif "Battery Power" in text:
            out["on_ac"] = False
        m = re.search(r"(\d{1,3})%", text)
        out["has_battery"] = bool(m)
        if m:
            out["battery_percent"] = float(m.group(1))
        return out
    return out


def _parse_version(text: str | None) -> tuple | None:
    if not text:
        return None
    m = re.match(r"\s*(\d+)(?:\.(\d+))?", str(text))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def _probe_nvidia(src: SystemSources) -> list:
    code, out = src.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                         "--format=csv,noheader,nounits"])
    if code != 0 or not out.strip():
        return []
    code2, header = src.run(["nvidia-smi"])
    cuda = None
    if code2 == 0:
        m = re.search(r"CUDA Version:\s*([\d.]+)", header)
        cuda = m.group(1) if m else None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            mem = float(parts[1]) / 1024.0
        except ValueError:
            mem = None
        cap = parts[3] if len(parts) > 3 and re.match(r"^\d+(\.\d+)?$", parts[3]) else None
        gpus.append({"vendor": "nvidia", "name": parts[0], "memory_gb": round(mem, 2) if mem else None,
                     "driver": parts[2], "cuda": cuda, "compute_capability": cap})
    return gpus


def _vendor_of(name: str) -> str:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low:
        return "nvidia"
    if "intel" in low or "iris" in low or "uhd graphics" in low or re.search(r"\barc\b", low):
        return "intel"
    if "amd" in low or "radeon" in low:
        return "amd"
    if "apple" in low:
        return "apple"
    return "other"


def _probe_gpus(src: SystemSources, plat: str, machine: str) -> list:
    gpus = _probe_nvidia(src)
    seen = {g["name"] for g in gpus}
    if plat == "windows":
        code, out = src.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                             "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"], timeout=15)
        if code == 0:
            for line in out.splitlines():
                name = line.strip()
                if name and name not in seen and "basic display" not in name.lower() \
                        and "remote" not in name.lower():
                    gpus.append({"vendor": _vendor_of(name), "name": name, "memory_gb": None,
                                 "driver": None, "cuda": None, "compute_capability": None})
                    seen.add(name)
    elif plat == "linux":
        ids = {"0x10de": "nvidia", "0x8086": "intel", "0x1002": "amd"}
        for card in src.glob("/sys/class/drm/card[0-9]"):
            vendor = (src.read(f"{card}/device/vendor") or "").strip().lower()
            kind = ids.get(vendor)
            if kind and not any(g["vendor"] == kind for g in gpus):
                gpus.append({"vendor": kind, "name": f"{kind} graphics", "memory_gb": None, "driver": None,
                             "cuda": None, "compute_capability": None})
    elif plat == "darwin" and machine == "arm64":
        gpus.append({"vendor": "apple", "name": "Apple silicon", "memory_gb": None, "driver": None,
                     "cuda": None, "compute_capability": None})
    return gpus


def _linux_throttle_count(src: SystemSources) -> int | None:
    files = src.glob("/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle/core_throttle_count")
    files += src.glob("/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle/package_throttle_count")
    total, found = 0, False
    for f in files:
        text = (src.read(f) or "").strip()
        if text.isdigit():
            total += int(text)
            found = True
    return total if found else None


def _nvidia_thermal(src: SystemSources) -> bool | None:
    for prefix in ("clocks_event_reasons", "clocks_throttle_reasons"):
        code, out = src.run(["nvidia-smi", f"--query-gpu={prefix}.hw_thermal_slowdown,{prefix}.sw_thermal_slowdown",
                             "--format=csv,noheader"])
        if code == 0 and out.strip():
            return "Active" in out
    return None


def thermal_sample(src: SystemSources | None = None, plat: str | None = None, has_nvidia: bool = False) -> dict:
    """One reading of the heat signals this computer exposes.

    cpu_throttle_count is a counter that rises while the processor slows down
    because of heat (Linux only). gpu_thermal_slowdown is True while an NVIDIA
    card reports a heat slowdown. Either can be None for not available.
    """
    src = src or SystemSources()
    plat = plat or src.platform()
    count = _linux_throttle_count(src) if plat == "linux" else None
    gpu = _nvidia_thermal(src) if has_nvidia else None
    return {"cpu_throttle_count": count, "gpu_thermal_slowdown": gpu}


def probe(path: str | os.PathLike = ".", *, sources: SystemSources | None = None,
          include_torch: bool = False) -> dict:
    """Everything the Train tab shows about this computer. Never raises."""
    src = sources or SystemSources()
    plat = src.platform()
    machine = src.machine()
    result: dict = {"platform": plat, "machine": machine, "python": src.python(), "notes": []}
    for key, fn in (("cpu", lambda: _probe_cpu(src, plat)), ("memory", lambda: _probe_memory(src, plat)),
                    ("disk", lambda: _probe_disk(src, str(path))), ("power", lambda: _probe_power(src, plat)),
                    ("gpus", lambda: _probe_gpus(src, plat, machine))):
        try:
            result[key] = fn()
        except Exception as exc:  # noqa: BLE001
            result[key] = [] if key == "gpus" else {}
            result["notes"].append(f"Could not read {key}: {type(exc).__name__}.")
    has_nvidia = any(g.get("vendor") == "nvidia" for g in result["gpus"])
    try:
        sample = thermal_sample(src, plat, has_nvidia)
    except Exception:  # noqa: BLE001
        sample = {"cpu_throttle_count": None, "gpu_thermal_slowdown": None}
    result["thermal"] = {
        "cpu_signal": sample["cpu_throttle_count"] is not None,
        "gpu_signal": sample["gpu_thermal_slowdown"] is not None,
        "sample": sample,
    }
    if include_torch:
        result["torch"] = torch_devices()
    return result


def live_state(path: str | os.PathLike = ".", *, sources: SystemSources | None = None,
               has_nvidia: bool = False) -> dict:
    """The fast-changing readings a running job checks every half minute: power,
    free disk and heat. Cheaper than probe, which also lists graphics cards."""
    src = sources or SystemSources()
    plat = src.platform()
    out: dict = {}
    try:
        out["power"] = _probe_power(src, plat)
    except Exception:  # noqa: BLE001
        out["power"] = {"on_ac": None, "has_battery": None, "battery_percent": None, "source": None}
    try:
        out["disk"] = _probe_disk(src, str(path))
    except Exception:  # noqa: BLE001
        out["disk"] = {"path": str(path), "total_gb": None, "free_gb": None}
    try:
        sample = thermal_sample(src, plat, has_nvidia)
    except Exception:  # noqa: BLE001
        sample = {"cpu_throttle_count": None, "gpu_thermal_slowdown": None}
    out["thermal"] = {"cpu_signal": sample["cpu_throttle_count"] is not None,
                      "gpu_signal": sample["gpu_thermal_slowdown"] is not None, "sample": sample}
    return out


def torch_devices() -> dict:
    """What an installed PyTorch can use. Only meaningful inside a training environment."""
    out = {"installed": False, "version": None, "cuda": False, "xpu": False, "mps": False, "device_names": []}
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return out
    out["installed"] = True
    out["version"] = getattr(torch, "__version__", None)
    try:
        if torch.cuda.is_available():
            out["cuda"] = True
            out["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    try:
        out["xpu"] = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:  # noqa: BLE001
        pass
    try:
        out["mps"] = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# Which PyTorch build
# ---------------------------------------------------------------------------


def _index_cuda_version(url: str | None) -> tuple | None:
    """The CUDA version an index serves, read from its name, for example cu130 is 13.0."""
    if not url:
        return None
    m = re.search(r"/cu(\d+)(\d)/?$", url.rstrip("/") + "/")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def choose_torch_index(hw: Mapping, indexes: Mapping[str, Any]) -> dict:
    """Pick the PyTorch download location for this computer.

    Returns {"vendor_key", "url", "device", "reason"}. device is what training
    will run on: cuda, xpu, rocm, mps or cpu. url is None when the matching
    location is not configured, and the reason then says so.
    """
    plat = hw.get("platform")
    gpus = hw.get("gpus") or []

    def pick(key: str, device: str, reason: str) -> dict:
        url = indexes.get(key)
        if not url:
            return {"vendor_key": key, "url": None, "device": device,
                    "reason": reason + f" The download location for {key} is not set in the settings, so "
                                       "the install will not start."}
        return {"vendor_key": key, "url": url, "device": device, "reason": reason}

    nvidia = [g for g in gpus if g.get("vendor") == "nvidia"]
    if nvidia and plat in ("windows", "linux"):
        driver_cuda = _parse_version(nvidia[0].get("cuda"))
        caps = [_parse_version(g.get("compute_capability")) for g in nvidia]
        cap = min((c for c in caps if c), default=None)
        need13 = _index_cuda_version(indexes.get("nvidia")) or CUDA_BUILD_FALLBACK["nvidia"]
        need12 = _index_cuda_version(indexes.get("nvidia_cuda12")) or CUDA_BUILD_FALLBACK["nvidia_cuda12"]
        cap_ok_13 = cap is not None and (cap[0] + cap[1] / 10.0) >= CUDA13_MIN_COMPUTE_CAPABILITY
        if driver_cuda and driver_cuda >= need13 and cap_ok_13:
            return pick("nvidia", "cuda", f"NVIDIA {nvidia[0]['name']} with a driver for CUDA "
                                          f"{driver_cuda[0]}.{driver_cuda[1]}.")
        if driver_cuda and driver_cuda >= need12:
            return pick("nvidia_cuda12", "cuda", f"NVIDIA {nvidia[0]['name']} with a driver for CUDA "
                                                 f"{driver_cuda[0]}.{driver_cuda[1]}, so the CUDA 12 build is used.")
        reason = ("An NVIDIA card was found, but its driver does not report a CUDA version this build "
                  "supports. Training will use the processor. Updating the NVIDIA driver may enable the card.")
        return pick("cpu", "cpu", reason)
    arc = [g for g in gpus if g.get("vendor") == "intel" and re.search(r"\barc\b", str(g.get("name", "")).lower())]
    if arc and plat in ("windows", "linux"):
        return pick("intel", "xpu", f"Intel {arc[0]['name']}, which PyTorch supports for training.")
    amd = [g for g in gpus if g.get("vendor") == "amd"]
    if amd and plat == "linux":
        return pick("amd_linux", "rocm", "An AMD graphics card on Linux.")
    if plat == "darwin" and hw.get("machine") == "arm64":
        return pick("apple", "mps", "Apple silicon.")
    intel = [g for g in gpus if g.get("vendor") == "intel"]
    if intel:
        reason = (f"{intel[0]['name']} is not on PyTorch's list of Intel graphics it can train on, so "
                  "training will use the processor. The graphics chip is not used.")
    elif amd:
        reason = "PyTorch's AMD builds are for Linux, so training will use the processor."
    else:
        reason = "No supported graphics card was found, so training will use the processor."
    return pick("cpu", "cpu", reason)


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------


def _auto_int(value: Any, auto: int) -> int:
    if value == "auto" or value is None:
        return auto
    return int(value)


def resource_plan(mode: str, hw: Mapping, hardware_cfg: Mapping, *, batch_size: int,
                  grad_accum_steps: int = 1) -> dict:
    """How a run divides its work in Full speed or Gentle.

    The effective batch, batch_size times grad_accum_steps, is the same in both
    modes. Gentle feeds it through smaller pieces, accumulating the learning
    signal, with fewer processor threads and loader workers, and keeps
    activations small with gradient checkpointing.
    """
    if mode not in ("full_speed", "gentle"):
        raise HardwareError("mode must be full_speed or gentle")
    batch_size, grad_accum_steps = int(batch_size), int(grad_accum_steps)
    if batch_size < 1 or grad_accum_steps < 1:
        raise HardwareError("batch size and accumulation steps must be at least 1")
    effective = batch_size * grad_accum_steps
    cores = (hw.get("cpu") or {}).get("logical_cores") or 1
    cfg = hardware_cfg.get(mode) or {}
    if mode == "full_speed":
        threads = _auto_int(cfg.get("cpu_threads"), max(1, cores - 1))
        workers = _auto_int(cfg.get("dataloader_workers"), max(0, min(4, cores // 2)))
        micro = batch_size
        checkpointing = bool(cfg.get("gradient_checkpointing", False))
        gpu_fraction = None
    else:
        fraction = float(cfg.get("cpu_threads_fraction", 0.5))
        threads = max(1, int(math.floor(cores * fraction)))
        workers = int(cfg.get("dataloader_workers", 1))
        micro = max(1, min(int(cfg.get("micro_batch", 1)), effective))
        checkpointing = bool(cfg.get("gradient_checkpointing", True))
        gpu_fraction = cfg.get("max_gpu_memory_fraction")
    while effective % micro:
        micro -= 1
    return {
        "mode": mode,
        "cpu_threads": int(threads),
        "dataloader_workers": int(workers),
        "micro_batch": int(micro),
        "grad_accum_steps": int(effective // micro),
        "effective_batch": int(effective),
        "gradient_checkpointing": checkpointing,
        "max_gpu_memory_fraction": gpu_fraction,
    }


def check_equivalent(full: Mapping, gentle: Mapping) -> list:
    """The fixed fields on which two run configurations differ. Empty means the
    two modes train with the same settings and differ only in how the work is split."""
    return [f for f in EQUIVALENCE_FIELDS if full.get(f) != gentle.get(f)]


# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------


class ThermalWatch:
    """Turns repeated heat readings into "sustained for N seconds".

    A reading counts as slowing down when the processor's throttle counter rose
    since the previous reading, or the graphics card reports a heat slowdown.
    """

    def __init__(self) -> None:
        self._last_count: int | None = None
        self._since: float | None = None

    def update(self, sample: Mapping, now: float) -> float:
        """Seconds the slowdown has lasted, 0 when it is not happening."""
        hot = False
        count = sample.get("cpu_throttle_count")
        if count is not None:
            if self._last_count is not None and count > self._last_count:
                hot = True
            self._last_count = count
        if sample.get("gpu_thermal_slowdown"):
            hot = True
        if not hot:
            self._since = None
            return 0.0
        if self._since is None:
            self._since = now
        return max(0.0, now - self._since)


def check_limits(hw: Mapping, limits: Mapping, *, starting: bool, thermal_seconds: float = 0.0,
                 gpu_memory_used_fraction: float | None = None) -> dict:
    """Decide whether a run may start or continue.

    Returns {"ok": bool, "stop": [...], "warnings": [...]}, each item
    {"code", "message"}. A value that cannot be read never counts as safe in the
    message, but it does not block on its own, because many desktops report no
    power state at all.
    """
    stop, warnings = [], []
    power = hw.get("power") or {}
    if limits.get("require_ac_power", True):
        if power.get("on_ac") is False:
            stop.append({"code": "on_battery", "message": LIMIT_MESSAGES["on_battery"]})
        elif power.get("on_ac") is None:
            warnings.append({"code": "power_unknown", "message": LIMIT_MESSAGES["power_unknown"]})
    pct = power.get("battery_percent")
    min_pct = limits.get("min_battery_percent")
    if power.get("has_battery") and pct is not None and min_pct is not None and power.get("on_ac") is not True \
            and pct < float(min_pct):
        stop.append({"code": "battery_low", "message": LIMIT_MESSAGES["battery_low"].format(limit=float(min_pct))})
    free = (hw.get("disk") or {}).get("free_gb")
    min_free = limits.get("min_free_disk_gb")
    if free is not None and min_free is not None and free < float(min_free):
        stop.append({"code": "disk_low", "message": LIMIT_MESSAGES["disk_low"].format(limit=float(min_free))})
    stop_after = limits.get("thermal_throttle_stop_seconds")
    thermal = hw.get("thermal") or {}
    if stop_after is not None and thermal_seconds >= float(stop_after) and thermal_seconds > 0:
        stop.append({"code": "thermal", "message": LIMIT_MESSAGES["thermal"].format(limit=float(stop_after))})
    if starting and not thermal.get("cpu_signal") and not thermal.get("gpu_signal"):
        warnings.append({"code": "thermal_unknown", "message": LIMIT_MESSAGES["thermal_unknown"]})
    headroom = limits.get("gpu_memory_headroom_fraction")
    if gpu_memory_used_fraction is not None and headroom is not None \
            and gpu_memory_used_fraction > 1.0 - float(headroom):
        stop.append({"code": "gpu_memory", "message": LIMIT_MESSAGES["gpu_memory"]})
    return {"ok": not stop, "stop": stop, "warnings": warnings}


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------


def estimate_run(*, train_images: int, eval_images: int, epochs: int, seconds_per_train_image: float | None,
                 seconds_per_eval_image: float | None, checkpoint_mb: float | None, checkpoints_kept: int,
                 dataset_mb: float = 0.0, export_mb: float | None = None) -> dict:
    """Time and disk for a run, from a measured speed on this computer.

    The speeds come from a short measurement in the training environment. Before
    that measurement exists the time is None, and the interface says it will be
    measured, rather than guessing.
    """
    hours = None
    if seconds_per_train_image is not None and seconds_per_eval_image is not None:
        seconds = epochs * (train_images * seconds_per_train_image + eval_images * seconds_per_eval_image)
        hours = seconds / 3600.0
    disk_mb = None
    if checkpoint_mb is not None:
        disk_mb = checkpoint_mb * max(1, checkpoints_kept) + float(dataset_mb) + float(export_mb or 0.0)
    return {"hours": hours, "disk_gb": (disk_mb / 1024.0) if disk_mb is not None else None}


def start_advice(hw: Mapping, limits: Mapping, *, device: str, estimate: Mapping) -> list:
    """Plain warnings shown beside the Start button. Never a refusal."""
    out = []
    warn_hours = limits.get("cpu_only_warn_hours")
    if device == "cpu" and estimate.get("hours") is not None and warn_hours is not None \
            and estimate["hours"] > float(warn_hours):
        out.append({"code": "cpu_only_long", "message": LIMIT_MESSAGES["cpu_only_long"].format(hours=estimate["hours"])})
    free = (hw.get("disk") or {}).get("free_gb")
    need = estimate.get("disk_gb")
    min_free = float(limits.get("min_free_disk_gb") or 0.0)
    if free is not None and need is not None and free - need < min_free:
        out.append({"code": "disk_tight", "message": (
            f"This run needs about {need:.1f} GB and would leave less than {min_free:g} GB free, so it would "
            "stop partway to protect your disk. Free some space first.")})
    return out


def now() -> float:
    """Monotonic seconds, the clock the heat watch uses."""
    return time.monotonic()


def gpu_memory_used_fraction(src: SystemSources | None = None) -> float | None:
    """Fraction of the first NVIDIA card's memory in use, or None."""
    src = src or SystemSources()
    code, out = src.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    if code != 0 or not out.strip():
        return None
    try:
        used, total = (float(x) for x in out.strip().splitlines()[0].split(",")[:2])
    except ValueError:
        return None
    return used / total if total > 0 else None


def describe(hw: Mapping, choice: Mapping) -> list:
    """Plain-language lines for the hardware card."""
    lines = []
    cpu = hw.get("cpu") or {}
    mem = hw.get("memory") or {}
    disk = hw.get("disk") or {}
    if cpu.get("name") or cpu.get("logical_cores"):
        lines.append(f"Processor: {cpu.get('name') or 'not reported'}, {cpu.get('logical_cores') or '?'} threads.")
    if mem.get("total_gb") is not None:
        avail = f", {mem['available_gb']:.1f} GB free now" if mem.get("available_gb") is not None else ""
        lines.append(f"Memory: {mem['total_gb']:.1f} GB{avail}.")
    if disk.get("free_gb") is not None:
        lines.append(f"Free disk where training writes: {disk['free_gb']:.1f} GB.")
    for g in hw.get("gpus") or []:
        mem_text = f", {g['memory_gb']:.1f} GB" if g.get("memory_gb") else ""
        lines.append(f"Graphics: {g['name']}{mem_text}.")
    lines.append(f"Training will run on: {choice.get('device', 'cpu')}. {choice.get('reason', '')}".strip())
    return lines
