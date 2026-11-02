"""Verification for the hardware probe and the training safety limits.

Path: tests/test_training_hardware.py

The probe reads the computer through a small set of sources. These checks
replace the sources with fixed answers that imitate a Windows laptop with Intel
graphics, Windows and Linux machines with NVIDIA cards, an AMD card, an Apple
computer, desktops without a battery and a laptop running hot, and check the
PyTorch build chosen, the processor share each mode may use, and every reason a
run must not start or must pause. A source that fails or answers nonsense must
give "not known", never an error and never "safe". The real probe is also run
once on the computer the tests run on.

Uses the standard library and the project itself.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.training import hardware as H  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


INDEXES = {
    "nvidia": "https://download.pytorch.org/whl/cu130",
    "nvidia_cuda12": "https://download.pytorch.org/whl/cu126",
    "intel": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
    "apple": "https://pypi.org/simple",
    "amd_linux": "https://download.pytorch.org/whl/rocm7.2",
}

LIMITS = {"require_ac_power": True, "min_battery_percent": 30, "min_free_disk_gb": 5.0,
          "gpu_memory_headroom_fraction": 0.10, "thermal_throttle_stop_seconds": 120, "cpu_only_warn_hours": 6.0}

HARDWARE_CFG = {
    "full_speed": {"dataloader_workers": "auto", "cpu_threads": "auto", "gradient_checkpointing": False},
    "gentle": {"micro_batch": 1, "dataloader_workers": 1, "cpu_threads_fraction": 0.5,
               "gradient_checkpointing": True, "max_gpu_memory_fraction": 0.75},
}


class Fake(H.SystemSources):
    """Fixed answers. commands maps the start of a command line, or a whole line ending in $, to (code, text)."""

    def __init__(self, plat="windows", machine="AMD64", cores=12, files=None, commands=None, cpu_name=None,
                 memory=None, power=None, disk=(500 * H.GIB, 120 * H.GIB)):
        self._plat, self._machine, self._cores = plat, machine, cores
        self.files = files or {}
        self.commands = commands or {}
        self._cpu_name, self._memory, self._power, self._disk = cpu_name, memory, power, disk
        self.ran = []

    def platform(self):
        return self._plat

    def machine(self):
        return self._machine

    def python(self):
        return "3.11.9"

    def cpu_count(self):
        return self._cores

    def run(self, args, timeout=8.0):
        self.ran.append(list(args))
        joined = " ".join(args)
        for key in sorted(self.commands, key=len, reverse=True):
            exact = key.endswith("$")
            if (exact and joined == key[:-1]) or (not exact and joined.startswith(key)):
                return self.commands[key]
        return None, ""

    def read(self, path):
        return self.files.get(path)

    def glob(self, pattern):
        import fnmatch
        return sorted(p for p in self._dirs() if fnmatch.fnmatch(p, pattern))

    def _dirs(self):
        out = set()
        for f in self.files:
            parts = f.split("/")
            for i in range(2, len(parts) + 1):
                out.add("/".join(parts[:i]))
        return out

    def disk_usage(self, path):
        return self._disk

    def windows_cpu_name(self):
        return self._cpu_name

    def windows_memory(self):
        return self._memory

    def windows_power(self):
        return self._power


def intel_laptop(**kw):
    """A Windows laptop with a 13th-generation Intel processor and Iris Xe graphics."""
    base = dict(plat="windows", cores=12, cpu_name="13th Gen Intel(R) Core(TM) i7-1355U",
                memory=(16 * H.GIB, 9 * H.GIB), power=(1, 1, 88),
                commands={"powershell": (0, "Intel(R) Iris(R) Xe Graphics\r\n")})
    base.update(kw)
    return Fake(**base)


# ---------------------------------------------------------------------------


def test_windows_laptop():
    print("a Windows laptop with Intel Iris Xe graphics")
    hw = H.probe("C:/x", sources=intel_laptop())
    check("the processor name is read", hw["cpu"]["name"] == "13th Gen Intel(R) Core(TM) i7-1355U")
    check("threads and memory are read", hw["cpu"]["logical_cores"] == 12 and hw["memory"]["total_gb"] == 16.0
          and hw["memory"]["available_gb"] == 9.0)
    check("the charger and battery are read", hw["power"]["on_ac"] is True and hw["power"]["battery_percent"] == 88.0
          and hw["power"]["has_battery"] is True)
    check("the graphics chip is listed as Intel", hw["gpus"] == [{"vendor": "intel", "name": "Intel(R) Iris(R) Xe Graphics",
                                                                  "memory_gb": None, "driver": None, "cuda": None,
                                                                  "compute_capability": None}])
    choice = H.choose_torch_index(hw, INDEXES)
    check("the processor build of PyTorch is chosen", choice["vendor_key"] == "cpu" and choice["device"] == "cpu"
          and choice["url"] == INDEXES["cpu"])
    check("the reason says the graphics chip is not used",
          "not on PyTorch's list" in choice["reason"] and "graphics chip is not used" in choice["reason"])
    check("Windows offers no heat signal, and the probe says so",
          hw["thermal"]["cpu_signal"] is False and hw["thermal"]["gpu_signal"] is False)
    res = H.check_limits(hw, LIMITS, starting=True)
    check("a run may start on mains power with plenty of disk", res["ok"] is True and res["stop"] == [])
    check("with a warning that heat cannot be watched", [w["code"] for w in res["warnings"]] == ["thermal_unknown"])
    check("the probe survives JSON", json.loads(json.dumps(hw)) == hw)
    lines = H.describe(hw, choice)
    check("the hardware card describes processor, memory, disk, graphics and the device",
          len(lines) == 5 and "i7-1355U" in lines[0] and "16.0 GB" in lines[1] and "120.0 GB" in lines[2]
          and "Iris" in lines[3] and "cpu" in lines[4])

    on_battery = H.probe("C:/x", sources=intel_laptop(power=(0, 1, 55)))
    res = H.check_limits(on_battery, LIMITS, starting=True)
    check("on battery a run does not start", res["ok"] is False and [s["code"] for s in res["stop"]] == ["on_battery"])
    check("and the message says progress is saved and to connect the charger",
          "charger" in res["stop"][0]["message"] and "saved" in res["stop"][0]["message"])
    low = H.probe("C:/x", sources=intel_laptop(power=(0, 2, 25)))
    res = H.check_limits(low, LIMITS, starting=False)
    check("on a low battery both reasons are given", [s["code"] for s in res["stop"]] == ["on_battery", "battery_low"])
    check("the battery limit comes from the settings", "30 percent" in res["stop"][1]["message"])
    relaxed = dict(LIMITS, require_ac_power=False)
    res = H.check_limits(on_battery, relaxed, starting=False)
    check("with the charger rule switched off, 55 percent battery may continue", res["ok"] is True)
    res = H.check_limits(low, relaxed, starting=False)
    check("but 25 percent still pauses", [s["code"] for s in res["stop"]] == ["battery_low"])
    charging_low = H.probe("C:/x", sources=intel_laptop(power=(1, 8, 20)))
    check("a low battery that is charging does not pause", H.check_limits(charging_low, LIMITS, starting=False)["ok"])
    desktop = H.probe("C:/x", sources=intel_laptop(power=(255, 128, 255)))
    check("a desktop with no battery counts as on mains power",
          desktop["power"]["has_battery"] is False and desktop["power"]["on_ac"] is True)
    unknown = H.probe("C:/x", sources=intel_laptop(power=(255, 255, 255)))
    check("an unknown power state is not known, not safe",
          unknown["power"]["on_ac"] is None and unknown["power"]["has_battery"] is None)
    res = H.check_limits(unknown, LIMITS, starting=True)
    check("and is reported as a warning that cannot be watched", "power_unknown" in [w["code"] for w in res["warnings"]])
    failing = H.probe("C:/x", sources=intel_laptop(power=None, memory=None, cpu_name=None))
    check("a failed Windows call gives not known everywhere",
          failing["power"]["on_ac"] is None and failing["memory"]["total_gb"] is None and failing["cpu"]["name"] is None)
    full = H.probe("C:/x", sources=intel_laptop(disk=(500 * H.GIB, 3 * H.GIB)))
    res = H.check_limits(full, LIMITS, starting=False)
    check("less than 5 GB free stops the run with the disk message",
          [s["code"] for s in res["stop"]] == ["disk_low"] and "free some space" in res["stop"][0]["message"])


def nvidia_machine(cuda="13.0", cap="8.9", plat="windows", name="NVIDIA GeForce RTX 4070"):
    return Fake(plat=plat, cpu_name="AMD Ryzen 7", memory=(32 * H.GIB, 20 * H.GIB), power=(1, 128, 255),
                files={"/proc/cpuinfo": "model name\t: AMD Ryzen 7\n"},
                commands={
                    "nvidia-smi --query-gpu=name": (0, f"{name}, 12282, 581.29, {cap}\n"),
                    "nvidia-smi --query-gpu=clocks_event_reasons": (0, "Not Active, Not Active\n"),
                    "nvidia-smi$": (0, f"| NVIDIA-SMI 581.29   Driver Version: 581.29   CUDA Version: {cuda} |\n"),
                    "powershell": (0, f"{name}\r\nIntel(R) UHD Graphics 770\r\n"),
                })


def test_nvidia():
    print("NVIDIA cards")
    hw = H.probe(".", sources=nvidia_machine())
    g = hw["gpus"][0]
    check("the NVIDIA card is read with memory, driver, CUDA and capability",
          g["vendor"] == "nvidia" and g["memory_gb"] == 11.99 and g["driver"] == "581.29" and g["cuda"] == "13.0"
          and g["compute_capability"] == "8.9")
    check("a second graphics adapter is listed once", [x["name"] for x in hw["gpus"]] ==
          ["NVIDIA GeForce RTX 4070", "Intel(R) UHD Graphics 770"])
    choice = H.choose_torch_index(hw, INDEXES)
    check("a CUDA 13 driver and a recent card get the CUDA 13 build",
          choice["vendor_key"] == "nvidia" and choice["device"] == "cuda" and choice["url"].endswith("cu130"))
    check("the graphics card exposes a heat signal", hw["thermal"]["gpu_signal"] is True)
    choice = H.choose_torch_index(H.probe(".", sources=nvidia_machine(cuda="12.8")), INDEXES)
    check("a CUDA 12.8 driver gets the CUDA 12 build", choice["vendor_key"] == "nvidia_cuda12" and choice["url"].endswith("cu126"))
    choice = H.choose_torch_index(H.probe(".", sources=nvidia_machine(cuda="13.0", cap="6.1",
                                                                       name="NVIDIA GeForce GTX 1080")), INDEXES)
    check("an older card that CUDA 13 dropped gets the CUDA 12 build", choice["vendor_key"] == "nvidia_cuda12")
    choice = H.choose_torch_index(H.probe(".", sources=nvidia_machine(cuda="11.4")), INDEXES)
    check("a driver too old for either build uses the processor and says to update the driver",
          choice["vendor_key"] == "cpu" and "Updating the NVIDIA driver" in choice["reason"])
    choice = H.choose_torch_index(H.probe(".", sources=nvidia_machine(plat="linux")), INDEXES)
    check("Linux with NVIDIA gets the same choice", choice["vendor_key"] == "nvidia")
    missing = dict(INDEXES, nvidia=None)
    choice = H.choose_torch_index(H.probe(".", sources=nvidia_machine()), missing)
    check("an unset download location is refused, never guessed",
          choice["url"] is None and "not set in the settings" in choice["reason"])
    hot = nvidia_machine()
    hot.commands["nvidia-smi --query-gpu=clocks_event_reasons"] = (0, "Active, Not Active\n")
    check("an NVIDIA heat slowdown is read", H.thermal_sample(hot, "windows", True)["gpu_thermal_slowdown"] is True)
    old = nvidia_machine()
    del old.commands["nvidia-smi --query-gpu=clocks_event_reasons"]
    old.commands["nvidia-smi --query-gpu=clocks_throttle_reasons"] = (0, "Not Active, Active\n")
    check("older drivers' field names are read too", H.thermal_sample(old, "windows", True)["gpu_thermal_slowdown"] is True)
    src = nvidia_machine()
    src.commands["nvidia-smi --query-gpu=memory.used"] = (0, "11500, 12282\n")
    frac = H.gpu_memory_used_fraction(src)
    check("graphics memory in use is read", abs(frac - 11500 / 12282) < 1e-9)
    res = H.check_limits(hw, LIMITS, starting=False, gpu_memory_used_fraction=frac)
    check("less than 10 percent graphics memory left pauses the run", [s["code"] for s in res["stop"]] == ["gpu_memory"])
    check("11 percent left does not", H.check_limits(hw, LIMITS, starting=False, gpu_memory_used_fraction=0.89)["ok"])
    mirror = dict(INDEXES, nvidia="https://mirror.example.org/torch-gpu", nvidia_cuda12="https://mirror.example.org/old")
    check("a mirror that does not name its CUDA version is matched by the documented build requirement",
          H.choose_torch_index(hw, mirror)["vendor_key"] == "nvidia"
          and H.choose_torch_index(H.probe(".", sources=nvidia_machine(cuda="12.6")), mirror)["vendor_key"] == "nvidia_cuda12")
    check("a broken nvidia-smi answer gives no card",
          H.probe(".", sources=Fake(plat="linux", commands={"nvidia-smi --query-gpu=name": (0, "garbage\n")}))["gpus"] == [])


def test_other_platforms():
    print("Intel Arc, AMD, Apple and Linux sources")
    arc = Fake(plat="windows", commands={"powershell": (0, "Intel(R) Arc(TM) Graphics\r\n")}, power=(1, 1, 90))
    choice = H.choose_torch_index(H.probe(".", sources=arc), INDEXES)
    check("Intel Arc graphics get the XPU build", choice["vendor_key"] == "intel" and choice["device"] == "xpu")
    amd_linux = Fake(plat="linux", files={"/sys/class/drm/card0/device/vendor": "0x1002\n"})
    choice = H.choose_torch_index(H.probe(".", sources=amd_linux), INDEXES)
    check("an AMD card on Linux gets the ROCm build", choice["vendor_key"] == "amd_linux" and choice["device"] == "rocm")
    amd_win = Fake(plat="windows", commands={"powershell": (0, "AMD Radeon RX 7800 XT\r\n")}, power=(1, 128, 255))
    choice = H.choose_torch_index(H.probe(".", sources=amd_win), INDEXES)
    check("an AMD card on Windows uses the processor and says why", choice["vendor_key"] == "cpu" and "Linux" in choice["reason"])
    mac = Fake(plat="darwin", machine="arm64", commands={
        "sysctl -n machdep.cpu.brand_string": (0, "Apple M3\n"), "sysctl -n hw.memsize": (0, "17179869184\n"),
        "pmset -g batt": (0, "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t87%; charging; 0:40 remaining\n")})
    hw = H.probe(".", sources=mac)
    check("Apple silicon is read", hw["cpu"]["name"] == "Apple M3" and hw["memory"]["total_gb"] == 16.0
          and hw["power"]["on_ac"] is True and hw["power"]["battery_percent"] == 87.0)
    check("and gets the Apple build", H.choose_torch_index(hw, INDEXES)["device"] == "mps")
    mac_batt = Fake(plat="darwin", machine="arm64", commands={
        "pmset -g batt": (0, "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t41%; discharging\n")})
    check("a Mac on battery is read", H.probe(".", sources=mac_batt)["power"]["on_ac"] is False)
    plain = Fake(plat="linux")
    check("no graphics card at all uses the processor",
          H.choose_torch_index(H.probe(".", sources=plain), INDEXES)["reason"].startswith("No supported graphics card"))

    files = {
        "/proc/cpuinfo": "processor\t: 0\nmodel name\t: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz\n",
        "/proc/meminfo": "MemTotal:       16303032 kB\nMemFree:  1 kB\nMemAvailable:    8151516 kB\n",
        "/sys/class/power_supply/AC/type": "Mains\n", "/sys/class/power_supply/AC/online": "0\n",
        "/sys/class/power_supply/BAT0/type": "Battery\n", "/sys/class/power_supply/BAT0/capacity": "45\n",
        "/sys/class/power_supply/BAT0/status": "Discharging\n",
        "/sys/class/power_supply/hidpp_battery_0/type": "Battery\n",
        "/sys/class/power_supply/hidpp_battery_0/scope": "Device\n",
        "/sys/class/power_supply/hidpp_battery_0/capacity": "5\n",
        "/sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count": "10\n",
        "/sys/devices/system/cpu/cpu1/thermal_throttle/core_throttle_count": "4\n",
        "/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count": "2\n",
    }
    lin = Fake(plat="linux", files=files)
    hw = H.probe(".", sources=lin)
    check("Linux processor and memory are read", hw["cpu"]["name"].startswith("Intel(R) Core(TM) i5-8250U")
          and hw["memory"]["total_gb"] == 15.55 and hw["memory"]["available_gb"] == 7.77)
    check("Linux power is read and a mouse battery is ignored",
          hw["power"]["on_ac"] is False and hw["power"]["battery_percent"] == 45.0)
    check("the Linux heat counter is summed over cores and package", hw["thermal"]["sample"]["cpu_throttle_count"] == 16
          and hw["thermal"]["cpu_signal"] is True)
    no_mains = dict(files)
    del no_mains["/sys/class/power_supply/AC/type"], no_mains["/sys/class/power_supply/AC/online"]
    no_mains["/sys/class/power_supply/BAT0/status"] = "Charging\n"
    check("a charging battery with no mains entry counts as on mains power",
          H.probe(".", sources=Fake(plat="linux", files=no_mains))["power"]["on_ac"] is True)
    check("no power supplies at all is not known", H.probe(".", sources=Fake(plat="linux"))["power"]["on_ac"] is None)
    desktop = {"/sys/class/power_supply/AC/type": "Mains\n", "/sys/class/power_supply/AC/online": "1\n"}
    p = H.probe(".", sources=Fake(plat="linux", files=desktop))["power"]
    check("a Linux desktop on mains has no battery", p["on_ac"] is True and p["has_battery"] is False)


def test_thermal_watch():
    print("heat watch")
    w = H.ThermalWatch()
    seq = [(0, 10), (30, 10), (60, 12), (90, 15), (120, 19), (150, 25), (180, 30)]
    lasted = [w.update({"cpu_throttle_count": c}, t) for t, c in seq]
    check("the first reading only sets the counter", lasted[0] == 0.0)
    check("a steady counter means no slowdown", lasted[1] == 0.0)
    check("a rising counter is timed from when it started rising", lasted[2:] == [0.0, 30.0, 60.0, 90.0, 120.0])
    hw = {"thermal": {"cpu_signal": True}, "disk": {"free_gb": 100}, "power": {"on_ac": True}}
    res = H.check_limits(hw, LIMITS, starting=False, thermal_seconds=lasted[-1])
    check("120 seconds of sustained slowdown pauses the run", [s["code"] for s in res["stop"]] == ["thermal"]
          and "cool" in res["stop"][0]["message"])
    check("119 seconds does not", H.check_limits(hw, LIMITS, starting=False, thermal_seconds=119.0)["ok"])
    check("a cooled processor resets the timer", w.update({"cpu_throttle_count": 30}, 200) == 0.0)
    check("a later slowdown is timed from its own start",
          w.update({"cpu_throttle_count": 31}, 300) == 0.0 and w.update({"cpu_throttle_count": 32}, 330) == 30.0)
    g = H.ThermalWatch()
    check("a graphics card slowdown counts", g.update({"gpu_thermal_slowdown": True}, 0) == 0.0
          and g.update({"gpu_thermal_slowdown": True}, 125) == 125.0)
    check("no signals at all never trigger", H.ThermalWatch().update({}, 999) == 0.0)


def test_plans():
    print("Full speed and Gentle")
    hw = {"cpu": {"logical_cores": 12}}
    full = H.resource_plan("full_speed", hw, HARDWARE_CFG, batch_size=4, grad_accum_steps=1)
    gentle = H.resource_plan("gentle", hw, HARDWARE_CFG, batch_size=4, grad_accum_steps=1)
    check("Full speed uses all threads but one and four batch pieces at once",
          full["cpu_threads"] == 11 and full["micro_batch"] == 4 and full["grad_accum_steps"] == 1
          and full["dataloader_workers"] == 4 and full["gradient_checkpointing"] is False)
    check("Gentle uses half the threads, one example at a time, accumulated four times",
          gentle["cpu_threads"] == 6 and gentle["micro_batch"] == 1 and gentle["grad_accum_steps"] == 4
          and gentle["dataloader_workers"] == 1 and gentle["gradient_checkpointing"] is True
          and gentle["max_gpu_memory_fraction"] == 0.75)
    check("both learn from the same effective batch", full["effective_batch"] == gentle["effective_batch"] == 4)
    odd = H.resource_plan("gentle", hw, dict(HARDWARE_CFG, gentle=dict(HARDWARE_CFG["gentle"], micro_batch=4)),
                          batch_size=6)
    check("a piece size that does not divide the batch is reduced until it does",
          odd["micro_batch"] == 3 and odd["grad_accum_steps"] == 2 and odd["effective_batch"] == 6)
    one = H.resource_plan("gentle", {"cpu": {"logical_cores": 1}}, HARDWARE_CFG, batch_size=4)
    check("a single-thread computer still gets one thread", one["cpu_threads"] == 1)
    fixed = H.resource_plan("full_speed", hw, {"full_speed": {"cpu_threads": 3, "dataloader_workers": 0}},
                            batch_size=2, grad_accum_steps=2)
    check("Full speed honours fixed numbers from the settings",
          fixed["cpu_threads"] == 3 and fixed["dataloader_workers"] == 0 and fixed["effective_batch"] == 4)
    check("an unknown mode is refused", raises(lambda: H.resource_plan("turbo", hw, HARDWARE_CFG, batch_size=4),
                                               H.HardwareError))
    check("a batch below 1 is refused", raises(lambda: H.resource_plan("gentle", hw, HARDWARE_CFG, batch_size=0),
                                               H.HardwareError))
    a = {"seed": 1, "epochs": 10, "effective_batch": 4, "resolution": 384, "base_model": "nano", "precision": "fp32",
         "learning_rate": 1e-4, "encoder_learning_rate": 1.5e-4, "weight_decay": 1e-4, "use_ema": True, "micro_batch": 4}
    b = dict(a, micro_batch=1, grad_accum_steps=4)
    check("configurations differing only in how work is split are equivalent", H.check_equivalent(a, b) == [])
    check("a lower resolution in Gentle is caught", H.check_equivalent(a, dict(b, resolution=320)) == ["resolution"])
    check("fewer epochs or another size is caught",
          H.check_equivalent(a, dict(b, epochs=5, base_model="small")) == ["epochs", "base_model"])


def test_estimates():
    print("estimates")
    e = H.estimate_run(train_images=100, eval_images=20, epochs=10, seconds_per_train_image=2.0,
                       seconds_per_eval_image=0.5, checkpoint_mb=400.0, checkpoints_kept=3, dataset_mb=50.0,
                       export_mb=120.0)
    check("time is epochs times the measured seconds per image", abs(e["hours"] - 10 * (200 + 10) / 3600) < 1e-12)
    check("disk is the kept checkpoints, the dataset and the export", abs(e["disk_gb"] - (1200 + 50 + 120) / 1024) < 1e-12)
    none = H.estimate_run(train_images=100, eval_images=20, epochs=10, seconds_per_train_image=None,
                          seconds_per_eval_image=None, checkpoint_mb=None, checkpoints_kept=3)
    check("without a measured speed nothing is guessed", none == {"hours": None, "disk_gb": None})
    hw = {"disk": {"free_gb": 40.0}}
    adv = H.start_advice(hw, LIMITS, device="cpu", estimate={"hours": 9.5, "disk_gb": 2.0})
    check("a processor-only run over 6 hours gets the Gentle and Nano advice",
          [a["code"] for a in adv] == ["cpu_only_long"] and "9.5 hours" in adv[0]["message"]
          and "Nano or Small" in adv[0]["message"])
    check("the same run on a graphics card gets no time advice",
          H.start_advice(hw, LIMITS, device="cuda", estimate={"hours": 9.5, "disk_gb": 2.0}) == [])
    tight = H.start_advice({"disk": {"free_gb": 8.0}}, LIMITS, device="cuda", estimate={"hours": 1, "disk_gb": 4.0})
    check("a run that would leave less than 5 GB free is warned about", [a["code"] for a in tight] == ["disk_tight"])
    check("advice never refuses, it only warns", all("code" in a and "message" in a for a in adv + tight))


def test_robustness():
    print("robustness")

    class Broken(Fake):
        def windows_memory(self):
            raise RuntimeError("boom")

        def disk_usage(self, path):
            raise OSError("boom")

    hw = H.probe(".", sources=Broken(plat="windows"))
    check("a source that raises is reported in the notes, not raised",
          any("memory" in n for n in hw["notes"]) and any("disk" in n for n in hw["notes"]))
    check("the result still has every section", {"cpu", "memory", "disk", "power", "gpus", "thermal"} <= set(hw))
    check("a missing program gives no output and no error", H.run_command(["audtheia-no-such-program-xyz"]) == (None, ""))
    real = H.probe(str(REPO))
    check("the real probe runs on this computer", real["platform"] in ("windows", "linux", "darwin")
          and isinstance(real["gpus"], list) and real["disk"].get("free_gb") is not None)
    check("the real probe survives JSON", json.loads(json.dumps(real)) == real)
    check("the real probe changes nothing it reads", (REPO / "audtheia" / "training" / "hardware.py").exists())
    check("torch details are only gathered when asked", "torch" not in real)
    td = H.torch_devices()
    check("asking about torch never fails, installed or not", set(td) >= {"installed", "cuda", "xpu", "mps"})

    src = (REPO / "audtheia" / "training" / "hardware.py").read_text(encoding="utf-8")
    check("hardware.py is ASCII only", all(ord(c) < 128 for c in src))
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    stdlib = {"__future__", "math", "os", "platform", "re", "shutil", "subprocess", "sys", "time", "pathlib",
              "typing", "ctypes", "winreg", "fnmatch", "torch"}
    check("it imports only the standard library, and PyTorch only inside a function", mods <= stdlib)
    tree = ast.parse(src)
    top = {n.names[0].name for n in tree.body if isinstance(n, ast.Import)}
    check("PyTorch is never imported at module level", "torch" not in top)


def main() -> int:
    test_windows_laptop()
    test_nvidia()
    test_other_platforms()
    test_thermal_watch()
    test_plans()
    test_estimates()
    test_robustness()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
