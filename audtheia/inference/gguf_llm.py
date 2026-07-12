"""Desktop language-model adapter (llama.cpp + GGUF).

Path: audtheia/inference/gguf_llm.py

The desktop is the only tier that runs a generative language model, and it runs
it in two places, both behind an injection seam so the engines that use it never
import a model runtime themselves. The verification engine
(audtheia/analysis/verify.py) asks an injected interpreter for the desktop's
interpretive points about one event; the longitudinal dream pass
(audtheia/analysis/dream.py) asks an injected narrator to phrase a candidate
pattern in plain words. This module supplies both, running a local GGUF model
through llama.cpp.

Two rules shape everything here.

  Interpretation is always inference, never measurement. The interpreter returns
  only interpretive points, each one a labelled inference the verification engine
  records with an inferred provenance. It never returns a measured value and
  never invents a number that stands in for one. In particular it does not
  produce a numeric rarity or an anomaly figure: those are quantities the system
  derives from real counts and baselines, not something a language model should
  guess. The interpreter is limited to genuinely interpretive, qualitative
  points (the ecological role a taxon plays, its likely behavioral context, a
  seasonal reading, a habitat-quality note, an interaction pattern), so the model
  can only ever add interpretation, never fabricate a measurement.

  A model fault is never fatal. Every call is defended: a model that returns
  malformed text, or is missing, or whose runtime is not installed, degrades to
  no interpretation and to the built-in pattern description. The pipeline runs
  end to end with no language model at all and simply gains richer interpretation
  and narration the moment one is present. This mirrors how the engines already
  treat these collaborators as optional.

The model runtime is imported lazily inside the loader, so importing this module
never requires it, and the loaded model is wrapped in a small completer whose
only method is to turn a prompt into text. The interpreter and narrator take that
completer as an argument, so their prompt-building and their defensive parsing
run end to end in a test against a scripted completer with no model file and no
llama.cpp present. The real model drops in unchanged behind the same seam.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("audtheia.inference.gguf_llm")

__all__ = [
    "GGUFInterpreter",
    "GGUFNarrator",
    "LlamaCompleter",
    "load_completer",
    "build_interpreter",
    "build_narrator",
    "LLMError",
    "LLMDependencyError",
    "INTERPRETER_POINT_TYPES",
    "DEFAULT_CONTEXT_TOKENS",
    "DEFAULT_INTERPRETER_MAX_TOKENS",
    "DEFAULT_NARRATOR_MAX_TOKENS",
]


# The interpretive point types this adapter is allowed to produce. Every one of
# these is a qualitative reading a language model can reasonably offer about a
# resolved taxon and its surroundings, and every one is a subset of the point
# types the verification storage contract accepts. The data-derived quantities a
# report needs to be exact (a numeric rarity, an anomaly magnitude, a cross-modal
# source attribution) are deliberately excluded, because the system computes
# those from measured counts and baselines and must never let a generated number
# stand in for a measured one.
INTERPRETER_POINT_TYPES = frozenset(
    {
        "ecological_role",
        "behavioral_context",
        "seasonal_assessment",
        "habitat_quality_flag",
        "interaction_pattern",
    }
)

# The provenance the verification engine records for a point the desktop model
# produced. It marks the point as the desktop's own analysis rather than a
# skill's output, and the engine fixes its data source to the inferred vocabulary
# regardless, so a point from here can never be recorded as a measurement.
_PRODUCED_BY_VERIFY = "verify"

# Model-family and runtime defaults. These are starting values for the local
# runtime, not per-deployment policy: a modest context window and a short output
# budget keep a per-event interpretation and a one-line narration fast on a
# desktop CPU. A low temperature keeps the interpreter close to the facts it is
# given and the narrator close to the description it is asked to rephrase.
DEFAULT_CONTEXT_TOKENS = 4096
DEFAULT_INTERPRETER_MAX_TOKENS = 512
DEFAULT_NARRATOR_MAX_TOKENS = 160
DEFAULT_TEMPERATURE = 0.2

# The most interpretive points the adapter will accept from one response, so a
# runaway generation cannot flood a record. The verification engine validates
# each surviving point again before it is stored.
_MAX_POINTS_PER_EVENT = 6


class LLMError(RuntimeError):
    """The desktop language model could not be loaded for an operational reason."""


class LLMDependencyError(LLMError):
    """The desktop language model needs a library that is not installed."""


# ===========================================================================
# The completer: a loaded model wrapped down to one prompt-to-text method
# ===========================================================================


class LlamaCompleter:
    """A loaded GGUF model exposed as a single prompt-to-text call.

    The interpreter and the narrator both reason through this one small surface,
    so neither depends on the exact shape of the llama.cpp API and both can be
    driven in a test by any object exposing the same complete method and a
    version. The version travels with the completer so the engines can stamp
    which model produced a verdict's interpretation or a pattern's narration.
    """

    def __init__(self, llm, *, version: Optional[str] = None) -> None:
        self._llm = llm
        self.version = version

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_INTERPRETER_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Run the model on one prompt and return its text, or an empty string.

        The llama.cpp completion returns a choices list whose first entry carries
        the generated text. Any deviation from that shape yields an empty string
        rather than an error, because a caller here treats empty as no result and
        falls back to its own safe default.
        """
        result = self._llm.create_completion(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )
        try:
            return str(result["choices"][0]["text"])
        except (KeyError, IndexError, TypeError):
            logger.warning("the language model returned an unexpected completion shape")
            return ""

    def close(self) -> None:
        self._llm = None


# ===========================================================================
# The interpreter the verification engine injects
# ===========================================================================


class GGUFInterpreter:
    """Turns one event's context into the desktop's interpretive points.

    The verification engine hands this a context carrying the observation, its
    per-taxon detections, its environmental readings, and the re-score verdict,
    and expects a list of interpretive points back. This interpreter builds a
    prompt from the resolved taxon and the measured surroundings, asks the model
    for qualitative interpretation as a small JSON array, and parses that array
    defensively into points the engine will accept. It adds no measured value and
    invents no number, and any fault along the way yields no points rather than a
    broken record.
    """

    def __init__(self, completer) -> None:
        self._completer = completer
        self.version = getattr(completer, "version", None)

    def interpret(self, context) -> list:
        """Return the model's interpretive points for one event, or an empty list."""
        try:
            taxon = self._resolved_taxon(context)
            if not taxon:
                # With no resolved taxon there is nothing to interpret; an event
                # with no visual subject (for example a pure acoustic event) is
                # left for a later tier rather than described from nothing.
                return []
            prompt = self._build_prompt(taxon, context)
            text = self._completer.complete(
                prompt, max_tokens=DEFAULT_INTERPRETER_MAX_TOKENS
            )
            return self._parse_points(text)
        except Exception:  # noqa: BLE001 - interpretation is enrichment, never load-bearing
            logger.exception("the desktop interpreter failed; recording no interpretation for this event")
            return []

    # -- prompt ----------------------------------------------------------

    @staticmethod
    def _resolved_taxon(context) -> Optional[str]:
        """The taxon to interpret: the verdict's resolution, else the field call."""
        verdict = getattr(context, "verdict", None)
        name = getattr(verdict, "resolved_scientific_name", None)
        if name:
            return str(name)
        field_name = getattr(context, "field_scientific_name", None)
        if field_name:
            return str(field_name)
        return None

    def _build_prompt(self, taxon: str, context) -> str:
        """Compose the instruction, the allowed point types, and the measured facts.

        The prompt states the taxon and the measured surroundings as given facts,
        asks only for interpretation drawn from a fixed list of point types, and
        requires a strict JSON array so the response parses without guesswork. It
        tells the model plainly not to restate a measurement and not to invent a
        number, which keeps the response on the interpretive side of the line the
        engine enforces anyway.
        """
        readings = self._describe_readings(getattr(context, "environmental_readings", []))
        companions = self._describe_companions(taxon, getattr(context, "child_detections", []))
        allowed = ", ".join(sorted(INTERPRETER_POINT_TYPES))

        lines = [
            "You are an ecological interpretation assistant for an environmental",
            "monitoring platform. You are given a confirmed taxon and the measured",
            "conditions recorded with it. Offer only qualitative ecological",
            "interpretation. Do not restate any measurement as your own finding and",
            "do not invent any number, count, or probability.",
            "",
            f"Confirmed taxon: {taxon}",
        ]
        if companions:
            lines.append(f"Other taxa recorded in the same event: {companions}")
        if readings:
            lines.append(f"Measured conditions: {readings}")
        else:
            lines.append("Measured conditions: none were recorded for this event.")
        lines.extend(
            [
                "",
                "Respond with a JSON array of at most five objects. Each object has",
                '"point_type" and "value". "value" is one concise sentence of',
                "interpretation with no numbers. Choose each \"point_type\" from",
                f"exactly this list: {allowed}.",
                "Return only the JSON array and nothing else.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _describe_readings(readings) -> str:
        """A short, readable list of the event's qualifying measured channels.

        Only channels that were actually measured are named, so the prompt never
        presents a missing or failed reading as a fact. The value and unit are
        given as context the interpreter may reason about, not as something to
        repeat back.
        """
        parts: list[str] = []
        for r in readings or []:
            if not isinstance(r, dict):
                continue
            if r.get("status") not in (None, "measured"):
                continue
            value = r.get("value")
            if value is None:
                continue
            channel = r.get("channel") or r.get("id") or "channel"
            unit = r.get("unit")
            parts.append(f"{channel} {value}{(' ' + str(unit)) if unit else ''}")
        return "; ".join(parts)

    @staticmethod
    def _describe_companions(taxon: str, children) -> str:
        """The other taxa in the event, so an interaction reading has something to
        rest on. The interpreted taxon itself is left out of the list."""
        names: list[str] = []
        for c in children or []:
            if not isinstance(c, dict):
                continue
            name = c.get("scientific_name") or c.get("common_name")
            if not name or str(name) == taxon:
                continue
            if str(name) not in names:
                names.append(str(name))
        return ", ".join(names)

    # -- parsing ---------------------------------------------------------

    def _parse_points(self, text: str) -> list:
        """Turn the model's text into accepted interpretive points.

        The response is expected to be a JSON array of objects. The first
        balanced array in the text is parsed, so a model that adds a stray word
        around the array is still read. Every object is checked: its point type
        must be one of the allowed interpretive types and its value must be
        non-empty text. Anything else is dropped, and a response that does not
        parse at all yields no points. Each accepted point is a plain dictionary
        the verification engine coerces and validates again before storing.
        """
        array = _extract_json_array(text)
        if array is None:
            logger.warning("the interpreter response did not contain a JSON array; recording no interpretation")
            return []

        points: list = []
        for item in array:
            if len(points) >= _MAX_POINTS_PER_EVENT:
                break
            if not isinstance(item, dict):
                continue
            point_type = item.get("point_type")
            value = item.get("value")
            if point_type not in INTERPRETER_POINT_TYPES:
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            points.append(
                {
                    "point_type": point_type,
                    "value": value.strip(),
                    "produced_by": _PRODUCED_BY_VERIFY,
                }
            )
        return points


# ===========================================================================
# The narrator the dream pass injects
# ===========================================================================


class GGUFNarrator:
    """Rephrases a candidate pattern's factual description in plain words.

    The dream pass builds a complete, factual description of every candidate it
    proposes and, when a narrator is present, offers it the chance to say the
    same thing more readably. This narrator asks the model to rephrase the given
    description without adding any claim or changing any figure, and returns the
    rephrasing only when it is non-empty. On any fault it returns nothing, and the
    dream pass keeps its own built-in description, so narration can only ever
    improve readability and never alters or blocks a candidate.
    """

    def __init__(self, completer) -> None:
        self._completer = completer
        self.version = getattr(completer, "version", None)

    def narrate(self, *, pattern_type: str, template: str) -> str:
        """Return a plain-language rephrasing of the description, or an empty string."""
        try:
            prompt = self._build_prompt(pattern_type, template)
            text = self._completer.complete(prompt, max_tokens=DEFAULT_NARRATOR_MAX_TOKENS)
            cleaned = text.strip()
            return cleaned
        except Exception:  # noqa: BLE001 - narration is a convenience, never load-bearing
            logger.exception("the desktop narrator failed; the built-in description will stand")
            return ""

    @staticmethod
    def _build_prompt(pattern_type: str, template: str) -> str:
        return "\n".join(
            [
                "You are rephrasing one factual sentence from an environmental",
                "monitoring report so it reads clearly for a general reader. Keep",
                "every figure and every named quantity exactly as written. Do not",
                "add any new claim, number, or interpretation. Return only the",
                "rephrased sentence.",
                "",
                f"Pattern type: {pattern_type}",
                f"Sentence: {template}",
            ]
        )


# ===========================================================================
# Loading
# ===========================================================================


def _extract_json_array(text: str):
    """The first balanced JSON array in a block of text, parsed, or None.

    A language model often wraps its array in a short preamble or a code fence.
    Scanning for the first opening bracket and matching it to its close, with
    string-awareness so a bracket inside a quoted value does not confuse the
    match, recovers the array from that surrounding text. A block with no
    balanced array, or one that does not parse as JSON, returns None.
    """
    if not text:
        return None
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    parsed = json.loads(snippet)
                except (ValueError, TypeError):
                    return None
                return parsed if isinstance(parsed, list) else None
    return None


def _resolve_model_file(settings) -> Path:
    """The concrete GGUF file to load, from the configured desktop model path.

    The configured path may name a folder (the common case, where the setup step
    places a downloaded model in the models directory) or a single file. A folder
    is searched for GGUF files: exactly one is used directly, and when several are
    present the first by name is chosen so the loader is deterministic, while the
    management interface is where a specific model is selected. A path that names
    a file is used as given. A missing model, or a folder with no GGUF file,
    raises a clear error a caller turns into running without the model.
    """
    entry = settings.raw.get("desktop_models", {}).get("llm", {})
    configured = entry.get("path")
    if not configured:
        raise LLMError(
            "no desktop language model is configured under desktop_models.llm.path."
        )

    model_path = Path(configured)
    if not model_path.is_absolute():
        model_path = Path(settings.repo_root) / model_path

    if model_path.is_dir():
        candidates = sorted(model_path.glob("*.gguf"))
        if not candidates:
            raise LLMError(
                f"no GGUF model file was found in {model_path}. Download a .gguf "
                f"model into that folder, or set desktop_models.llm.path to a file."
            )
        if len(candidates) > 1:
            logger.info(
                "several GGUF models are present in %s; using %s. Select a specific "
                "model in the interface to pin one.",
                model_path,
                candidates[0].name,
            )
        return candidates[0]

    if not model_path.exists():
        raise LLMError(
            f"the desktop language model was not found at {model_path}. Download a "
            f".gguf model there, or set desktop_models.llm.path to its folder."
        )
    return model_path


def _import_llama():
    try:
        from llama_cpp import Llama  # imported here so this module loads without it
        return Llama
    except Exception as exc:  # noqa: BLE001
        raise LLMDependencyError(
            "llama.cpp (the llama-cpp-python package) is required to run the desktop "
            "language model, but it is not installed."
        ) from exc


def load_completer(
    settings,
    *,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> LlamaCompleter:
    """Load the configured desktop GGUF model into a completer.

    Resolves the model file from the configuration, loads it through llama.cpp on
    the CPU, and wraps it so the interpreter and narrator can share one loaded
    model rather than each loading its own copy. The version stamped on the
    completer prefers a version declared in the configuration and falls back to
    the model file's name, so every interpretation and narration can record which
    model produced it. A missing model or a missing runtime raises, which a
    caller turns into running with no language model.
    """
    model_file = _resolve_model_file(settings)
    Llama = _import_llama()

    entry = settings.raw.get("desktop_models", {}).get("llm", {})
    version = entry.get("version") or model_file.stem

    try:
        llm = Llama(model_path=str(model_file), n_ctx=int(context_tokens), verbose=False)
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"the desktop language model at {model_file} could not be loaded: {exc}") from exc

    logger.info("desktop language model loaded from %s (version %s)", model_file, version)
    return LlamaCompleter(llm, version=version)


def build_interpreter(settings, *, completer: Optional[LlamaCompleter] = None) -> GGUFInterpreter:
    """Build the verification interpreter, loading the model when none is passed.

    Pass a shared completer to reuse one loaded model across the interpreter and
    the narrator; with none passed, the model is loaded here.
    """
    if completer is None:
        completer = load_completer(settings)
    return GGUFInterpreter(completer)


def build_narrator(settings, *, completer: Optional[LlamaCompleter] = None) -> GGUFNarrator:
    """Build the dream-pass narrator, loading the model when none is passed.

    Pass a shared completer to reuse one loaded model across the narrator and the
    interpreter; with none passed, the model is loaded here.
    """
    if completer is None:
        completer = load_completer(settings)
    return GGUFNarrator(completer)
