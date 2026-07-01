"""Audtheia field-station capture composer.

Path: audtheia/pipeline/composer.py

A field station's rule is that every trigger captures every sense at once. When
the camera closes an encounter, the station still records the sound around it,
where it is, and what its sensors read. When the hydrophone opens an event, the
station still records the frames in view at that moment, where it is, and what
its sensors read. One trigger, one complete multimodal record.

The detection loop, though, calls a single capture and takes a single result,
and each capture module was built to fill only its own part of that result: the
audio capture fills the audio fields, the location-and-environment capture fills
the location and sensor fields. This composer is the piece that makes the
one-call, one-result loop honour the capture-everything rule. It holds the
separate capture legs, runs them together for the same event window, and merges
their single-purpose results into one.

Two properties matter here.

  Every leg runs, and one leg's failure never sinks the others. The legs run in
  parallel, and each is isolated: if a sensor bank throws or an audio device is
  unplugged, that leg contributes nothing and is logged, while the legs that did
  succeed are still merged into the record. A partial record of what was
  actually captured is worth far more than no record at all, and the missing
  parts are simply absent, which a later reader sees as not captured rather than
  as a measured value.

  The legs do not overwrite each other. Each leg fills a different part of the
  result, so merging is a matter of taking each leg's own fields. If two legs
  ever set the same field, that is a wiring mistake, so the merge keeps the first
  and logs the collision rather than silently letting one leg clobber another.

The composer is reached by the detection loop through the same small capture
interface every leg already implements, so it drops in wherever a single capture
did, and the station gains complete multimodal capture with no change to the
loop.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from audtheia.pipeline.monitor import CaptureResult, TrackEvent

__all__ = [
    "CaptureLeg",
    "CaptureComposer",
    "ComposedTriggerSink",
    "merge_capture_results",
]

logger = logging.getLogger("audtheia.pipeline.composer")


# A capture leg is anything that, given an event window, returns the part of the
# capture result it is responsible for. The audio capture and the
# location-and-environment capture are the two legs today; a future leg fits the
# same shape.
CaptureLeg = Callable[[str, str], CaptureResult]


# The scalar fields of a capture result, paired so the merge can copy each from
# whichever leg filled it and detect the mistake of two legs filling the same
# one. The environmental-readings list is merged separately by concatenation,
# since more than one leg could in principle contribute channels.
_SCALAR_FIELDS = (
    "audio_clip_path",
    "audio_true_duration_seconds",
    "audio_capped",
    "gps_latitude",
    "gps_longitude",
    "gps_elevation",
    "gps_status",
    "acoustic_model_version",
    "gbif_snapshot_date",
    "iucn_fetch_date",
)


def merge_capture_results(results: list[CaptureResult]) -> CaptureResult:
    """Combine several single-purpose capture results into one.

    Each scalar field is taken from the one leg that set it; the environmental
    readings from every leg are gathered together. If two legs set the same
    scalar field, the first is kept and the collision is logged, because that can
    only mean two legs were wired to fill the same part of the record.
    """
    merged = CaptureResult()
    for result in results:
        for name in _SCALAR_FIELDS:
            incoming = getattr(result, name)
            if incoming is None:
                continue
            current = getattr(merged, name)
            if current is None:
                setattr(merged, name, incoming)
            elif current != incoming:
                logger.warning(
                    "two capture legs both set %s (%r and %r); keeping the first",
                    name,
                    current,
                    incoming,
                )
        if result.environmental_readings:
            merged.environmental_readings.extend(result.environmental_readings)
    return merged


class CaptureComposer:
    """Runs several capture legs for one event window and merges their results.

    The legs run in parallel, since each waits on its own device, and each is
    isolated so one leg's failure never denies the record the others captured.
    The composer is deliberately unaware of what any leg does: it only knows each
    leg returns the part of the result it owns, and that the parts merge into a
    whole.
    """

    def __init__(self, legs: list[tuple[str, CaptureLeg]]) -> None:
        # Each leg carries a short name used only in the log, so a failure names
        # the leg that failed.
        self._legs = list(legs)

    def compose(self, first_seen: str, last_seen: str) -> CaptureResult:
        """Capture every leg for one event window and return one merged result."""
        if not self._legs:
            return CaptureResult()

        results: list[CaptureResult] = []
        with ThreadPoolExecutor(max_workers=len(self._legs)) as pool:
            futures = {
                pool.submit(self._run_leg, name, leg, first_seen, last_seen): name
                for name, leg in self._legs
            }
            for future in futures:
                results.append(future.result())

        return merge_capture_results(results)

    @staticmethod
    def _run_leg(
        name: str,
        leg: CaptureLeg,
        first_seen: str,
        last_seen: str,
    ) -> CaptureResult:
        # A leg that raises contributes an empty result rather than bringing down
        # the whole capture. The empty result merges to nothing, so the record
        # carries exactly what the working legs captured and no invented values.
        try:
            return leg(first_seen, last_seen)
        except Exception:  # noqa: BLE001 - a leg fault is isolated and logged, never fatal
            logger.exception("capture leg %s failed; its fields will be absent", name)
            return CaptureResult()


class ComposedTriggerSink:
    """The vision loop's capture: every sense for every visual trigger.

    The detection loop calls this when an encounter closes. It composes the
    audio capture and the location-and-environment capture for the encounter's
    window and returns one merged result. The encounter's own frames are already
    held by the loop, so the visual sense of a visual trigger needs no capture
    here; this sink adds the other senses the capture-everything rule requires.
    """

    def __init__(
        self,
        *,
        acoustic_sink=None,
        environment_capture=None,
    ) -> None:
        self._acoustic_sink = acoustic_sink
        self._environment_capture = environment_capture

    def on_event(self, event: TrackEvent) -> CaptureResult:
        legs: list[tuple[str, CaptureLeg]] = []

        if self._acoustic_sink is not None:
            # The audio leg reads the sound around the encounter from the shared
            # ring buffer, keyed by the encounter itself.
            legs.append(("acoustic", lambda fs, ls: self._acoustic_sink.on_event(event)))

        if self._environment_capture is not None:
            legs.append(
                ("environment", self._environment_capture.capture)
            )

        composer = CaptureComposer(legs)
        return composer.compose(event.first_seen, event.last_seen)
