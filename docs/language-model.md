# The desktop language model

The desktop is the only tier that runs a generative language model. It runs a
local GGUF model through llama.cpp in two places, both optional: the verification
step asks it for qualitative interpretation of an event, and the longitudinal
pass asks it to phrase a candidate pattern in plainer words. The whole pipeline
runs end to end with no language model at all; a model only adds interpretation
and narration once one is present and loadable.

Because it is optional, a missing model, a missing runtime, or a model that
fails to load never stops the application. The desktop logs the reason and
continues without interpretation and narration. The Brain tab, under Models and
Memory, reports the current status and, when something is wrong, the exact fix.

## Requirements

1. The runtime: the `llama-cpp-python` package.
2. A model file: a single `.gguf` file in the language model folder, or a
   folder of them with one selected in the interface.

## The common failure: Windows error 0xc000001d

The most common problem is that the runtime installs, and a model is present,
but loading the model fails with `WinError 0xc000001d` (an illegal instruction),
often surfaced as an `OSError`. This is not a corrupt model or a bad path. It
means the installed `llama-cpp-python` was built for CPU instructions that this
computer's processor does not have. The default prebuilt wheel assumes AVX2 and
FMA; an older or low-power CPU without them executes an unsupported instruction
and the load aborts.

The fix is to install a build that matches this CPU. Two ways:

Install a prebuilt wheel for the right instruction set. Wheels are published for
`basic` (no AVX, FMA, or F16C, for the oldest CPUs), `AVX`, and `AVX2`. Choose
the highest set this CPU actually supports. For example, for a CPU with AVX but
not AVX2:

```
python -m pip install llama-cpp-python --prefer-binary --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

If a matching prebuilt wheel is not available, rebuild from source with the
unsupported instructions turned off:

```
set CMAKE_ARGS=-DGGML_NATIVE=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF
python -m pip install llama-cpp-python --force-reinstall --no-cache-dir
```

`-DGGML_NATIVE=OFF` stops the build from targeting whatever CPU compiled it, and
the two `OFF` flags drop the instruction families that most often cause the
crash. On the very oldest CPUs, also add `-DGGML_AVX=OFF -DGGML_F16C=OFF`.

To confirm what this CPU supports before choosing, check its feature flags (for
example CPU-Z on Windows, or `lscpu` on Linux) and look for `avx`, `avx2`, and
`fma`.

After reinstalling, restart the desktop application. The Brain tab status should
move from a load failure to the model being present, and the next verification
run or longitudinal pass will use it.

## Choosing a model

Any GGUF chat or instruct model works. A model around 3B parameters is a good
default on a desktop CPU; a larger model is slower per event. The interface lists
every `.gguf` file in the folder and lets one be selected; the selection applies
the next time the desktop starts, since the model is loaded once at startup.
