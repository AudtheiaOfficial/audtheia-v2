"""Desktop capture drivers for Audtheia.

Path: audtheia/pipeline/drivers.py

The field runner (audtheia/pipeline/__main__.py) builds a station's senses from
an optional drivers module through small factory functions. This module is that
drivers module for an ordinary desktop computer with no field hardware. It turns
a normal video source, a connected webcam, a network stream, or a saved video
file, into the same frame stream the detection loop already expects, and it runs
the screening detection model through ONNX Runtime on the desktop instead of a
field accelerator. Everything downstream, the object tracker, event aggregation,
quality control, and storage, is unchanged and never learns that the frames came
from a webcam rather than a field camera.

The two heavy libraries, OpenCV for video and ONNX Runtime for detection, are
imported lazily inside the factory functions. Importing this module therefore
never requires them, so a field station that reaches the same seam without those
libraries installed loads it and simply falls back to no capture, while a clear
message names the package to install if a desktop capture is actually started
without it.

The frame source and the detector each take their underlying backend as an
argument, so both can be exercised end to end against a scripted feed and a
scripted model output with no camera and no model file present; the factory
functions supply the real backends.
"""

from __future__ import annotations

import ast
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from audtheia.pipeline.monitor import Frame, RawDetection, ISO_FORMAT

logger = logging.getLogger("audtheia.pipeline.drivers")

__all__ = [
    "OpenCVFrameSource",
    "OnnxYoloDetector",
    "build_frame_source",
    "build_detector",
    "CaptureError",
    "CaptureConfigError",
    "CaptureDependencyError",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_NMS_IOU",
    "LETTERBOX_FILL",
]

# Fallbacks for a model whose input size or activation floor cannot be read from
# its own metadata. Six hundred and forty is the standard YOLO input edge; the
# confidence floor is only a fallback, since the real one is read from the
# station's tracker activation threshold. These are algorithm constants, not
# per-deployment values, so they live here rather than in the configuration.
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS_IOU = 0.45
LETTERBOX_FILL = 114


class CaptureError(RuntimeError):
    """A desktop capture could not start for an operational reason."""


class CaptureConfigError(CaptureError):
    """A desktop capture is missing a required piece of configuration."""


class CaptureDependencyError(CaptureError):
    """A desktop capture needs a library that is not installed."""


# ---------------------------------------------------------------------------
# Lazy imports of the heavy backends.
# ---------------------------------------------------------------------------


def _import_cv2():
    try:
        import cv2  # imported here so this module loads without OpenCV present
        return cv2
    except Exception as exc:  # noqa: BLE001 - any import failure means it is unusable
        raise CaptureDependencyError(
            "OpenCV is required to read a webcam, stream, or video file on the "
            "desktop, but it is not installed. Install opencv-python-headless."
        ) from exc


def _import_onnxruntime():
    try:
        import onnxruntime as ort  # imported here so this module loads without it
        return ort
    except Exception as exc:  # noqa: BLE001
        raise CaptureDependencyError(
            "ONNX Runtime is required to run the detection model on the desktop, "
            "but it is not installed. Install onnxruntime."
        ) from exc


# ---------------------------------------------------------------------------
# Time helpers, matching the storage layer's format exactly.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def _iso_from(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime(ISO_FORMAT)


# ===========================================================================
# Frame source: a webcam, a network stream, or a video file
# ===========================================================================


def _parse_video_spec(spec: str) -> tuple:
    """Turn a configured video string into an OpenCV target and a live flag.

    Accepted forms, so a deployment can name its source plainly:
      webcam        or  webcam:2      a connected camera by index (default 0)
      url:rtsp://.. or  http://..     a network stream
      file:/path    or  /path/clip.mp4  a saved video file
    A bare number is a camera index; a bare rtsp/http address is a stream;
    anything else is treated as a file path. The live flag is true for a camera
    or a stream, where timestamps are the real wall-clock moment each frame is
    read, and false for a file, where timestamps advance from the file's own
    frame rate so a replayed clip yields sensible event durations.
    """
    s = str(spec).strip()
    lowered = s.lower()
    if lowered.startswith("webcam"):
        rest = s.split(":", 1)[1].strip() if ":" in s else ""
        return (int(rest) if rest.isdigit() else 0), True
    if lowered.startswith("stream:"):
        return s.split(":", 1)[1].strip(), True
    if lowered.startswith("url:"):
        return s.split(":", 1)[1].strip(), True
    if lowered.startswith("file:"):
        return s.split(":", 1)[1].strip(), False
    if s.isdigit():
        return int(s), True
    if lowered.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://")):
        return s, True
    return s, False


def _best_stream_url(info: dict) -> str:
    """Pick a directly-readable video stream address from yt-dlp's page info.

    yt-dlp returns either a single resolved address or a list of formats. A
    format that carries video and has an address is preferred, choosing the
    highest resolution, so the live feed OpenCV opens is the best one the page
    offers rather than an audio-only or thumbnail track.
    """
    if not info:
        raise CaptureError("the stream page returned no information to open")
    if info.get("url"):
        return info["url"]
    formats = info.get("formats") or []
    candidates = [f for f in formats if f.get("url") and f.get("vcodec") not in (None, "none")]
    if not candidates:
        raise CaptureError("the stream page offered no video stream to open")
    candidates.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
    return candidates[0]["url"]


def _resolve_stream_url(page_url: str, *, ydl_factory=None) -> str:
    """Resolve a web-page video address to a direct stream address via yt-dlp.

    The extractor is created through a small factory so a test can supply its own
    without reaching the network. When yt-dlp is not installed, a clear message
    names it; when the page cannot be resolved, the reason is reported rather than
    failing obscurely deep inside OpenCV.
    """
    if ydl_factory is None:
        try:
            from yt_dlp import YoutubeDL  # imported here so this module loads without yt-dlp
        except Exception as exc:  # noqa: BLE001
            raise CaptureDependencyError(
                "yt-dlp is required to read a 'stream:' page address (for example a "
                "YouTube or Pixcams live page), but it is not installed. Install yt-dlp, "
                "or give a direct camera, file, or stream URL instead."
            ) from exc

        def ydl_factory():  # noqa: E306 - a tiny default factory
            return YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True})

    try:
        with ydl_factory() as ydl:
            info = ydl.extract_info(page_url, download=False)
    except CaptureError:
        raise
    except Exception as exc:  # noqa: BLE001 - any extractor failure is reported plainly
        raise CaptureError(f"yt-dlp could not resolve a stream from {page_url!r}: {exc}") from exc
    return _best_stream_url(info)


class OpenCVFrameSource:
    """A frame source backed by an OpenCV capture.

    The capture is any object exposing OpenCV's read (returning an ok flag and a
    BGR image) and release, so a test drives this with a scripted capture and no
    camera. OpenCV yields BGR pixels; the detection loop and the frame writer
    expect RGB, so each frame is converted once here, which is the only colour
    handling the rest of the pipeline needs.
    """

    def __init__(self, capture, *, live: bool, fps: Optional[float] = None, base_time: Optional[datetime] = None) -> None:
        self._capture = capture
        self._live = bool(live)
        self._fps = float(fps) if fps and fps > 0 else None
        self._base_time = base_time or datetime.now(timezone.utc)
        self._index = 0

    def read(self) -> Optional[Frame]:
        ok, image_bgr = self._capture.read()
        if not ok or image_bgr is None:
            return None

        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        else:
            image_rgb = np.ascontiguousarray(image_bgr)

        index = self._index
        self._index += 1

        if self._live or self._fps is None:
            captured_at = _now_iso()
        else:
            captured_at = _iso_from(self._base_time + timedelta(seconds=index / self._fps))

        # The desktop clock is disciplined by the operating system, so a desktop
        # capture's timestamps are trusted rather than provisional.
        return Frame(index=index, image=image_rgb, captured_at=captured_at, time_provisional=0)

    def close(self) -> None:
        release = getattr(self._capture, "release", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001 - closing a source must never raise
                logger.debug("releasing the video capture raised; ignoring", exc_info=True)


def build_frame_source(settings, station: dict) -> OpenCVFrameSource:
    """Open the station's configured desktop video source."""
    source = settings.capture_source(station)
    video = source.get("video")
    if not video:
        raise CaptureConfigError(
            "this station has no capture.source.video configured; a desktop "
            "capture needs one, for example 'webcam:0', 'url:rtsp://...', or "
            "'file:/path/to/clip.mp4'."
        )

    target, live = _parse_video_spec(video)
    if video.strip().lower().startswith("stream:"):
        # A page address (a YouTube or Pixcams live page) is not a media stream
        # OpenCV can open; yt-dlp resolves it to the direct stream first.
        target = _resolve_stream_url(target)
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(target)
    if not capture.isOpened():
        raise CaptureError(
            f"could not open the video source {video!r}. Check the camera index, "
            f"the stream address, or the file path."
        )

    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = reported_fps if reported_fps and reported_fps > 0 else station.get("capture", {}).get("fps")
    return OpenCVFrameSource(capture, live=live, fps=fps)


# ===========================================================================
# Detector: a YOLO model run through ONNX Runtime on the desktop
# ===========================================================================


class OnnxYoloDetector:
    """A per-frame detector that runs a YOLO model through ONNX Runtime.

    The session is any object exposing ONNX Runtime's run and get_inputs, so a
    test drives the detector's own preprocessing and decoding with a scripted
    model output and no model file. The model is the desktop screening model,
    equivalent in role to the accelerator's model on a field station: it turns a
    frame into raw per-object boxes, which the tracker then associates into
    events exactly as it does for field detections.

    The decode targets the standard YOLO export whose output holds, per candidate
    box, the box centre and size followed by one score per class. Boxes are
    scored against a confidence floor, mapped back out of the letterboxed input
    to original-frame pixels, and reduced by non-maximum suppression.
    """

    def __init__(
        self,
        session,
        *,
        class_names: dict,
        input_size: tuple = DEFAULT_INPUT_SIZE,
        conf_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_NMS_IOU,
        input_name: Optional[str] = None,
    ) -> None:
        self._session = session
        self._class_names = {int(k): str(v) for k, v in dict(class_names).items()}
        self._in_w = int(input_size[0])
        self._in_h = int(input_size[1])
        self._conf = float(conf_threshold)
        self._iou = float(iou_threshold)
        self._input_name = input_name or session.get_inputs()[0].name

    @property
    def class_names(self) -> dict:
        return dict(self._class_names)

    def detect(self, frame: Frame) -> list:
        blob, scale, pad = self._preprocess(frame.image)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._postprocess(outputs[0], scale, pad, frame.image.shape[1], frame.image.shape[0])

    def close(self) -> None:
        self._session = None

    # -- preprocessing and decoding -------------------------------------

    def _preprocess(self, image_rgb: np.ndarray):
        """Letterbox the frame to the model's square input, keeping aspect."""
        cv2 = _import_cv2()
        h0, w0 = image_rgb.shape[:2]
        scale = min(self._in_w / w0, self._in_h / h0)
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h))

        canvas = np.full((self._in_h, self._in_w, 3), LETTERBOX_FILL, dtype=np.uint8)
        pad_x = (self._in_w - new_w) // 2
        pad_y = (self._in_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(blob), scale, (pad_x, pad_y)

    def _postprocess(self, output, scale, pad, orig_w, orig_h) -> list:
        pred = np.asarray(output, dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        # Normalize to one row per candidate box. A standard export is
        # (4 + classes, boxes); some are already (boxes, 4 + classes).
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.shape[0] == 0 or pred.shape[1] < 5:
            return []

        boxes = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = confidences >= self._conf
        if not np.any(keep):
            return []
        boxes, class_ids, confidences = boxes[keep], class_ids[keep], confidences[keep]

        pad_x, pad_y = pad
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, orig_w)
        y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, orig_h)
        x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, orig_w)
        y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, orig_h)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        detections = []
        for i in self._nms(xyxy, confidences, self._iou):
            cid = int(class_ids[i])
            detections.append(
                RawDetection(
                    x1=float(x1[i]), y1=float(y1[i]), x2=float(x2[i]), y2=float(y2[i]),
                    confidence=float(confidences[i]),
                    class_id=cid,
                    class_name=self._class_names.get(cid, str(cid)),
                )
            )
        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
        if boxes.shape[0] == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]
        return keep


def _class_names_from_session(session) -> dict:
    """Read a YOLO export's class-name map from its own ONNX metadata.

    A standard export stores the names as a string form of a mapping under the
    'names' key. When it is absent or unreadable, an empty map is returned and
    the loop falls back to the numeric class index as the label, so a model
    missing its metadata still runs.
    """
    try:
        meta = session.get_modelmeta().custom_metadata_map
    except Exception:  # noqa: BLE001 - a model without metadata is allowed
        return {}
    raw = meta.get("names") if isinstance(meta, dict) else None
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {i: str(v) for i, v in enumerate(parsed)}
    return {}


def build_detector(settings, station: dict) -> OnnxYoloDetector:
    """Load the station's desktop screening model into an ONNX Runtime detector."""
    entry = settings.desktop_visual_model(station)
    path = entry.get("path")
    if not path:
        raise CaptureConfigError(
            "this station has no models.visual_desktop.path configured; a desktop "
            "capture needs an ONNX detection model to screen frames."
        )

    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = Path(settings.repo_root) / model_path
    if not model_path.exists():
        raise CaptureConfigError(
            f"the desktop detection model was not found at {model_path}. Run setup "
            f"to download it, or place your own ONNX model there."
        )

    ort = _import_onnxruntime()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    class_names = _class_names_from_session(session)
    if not class_names:
        logger.warning(
            "the desktop detection model has no class-name metadata; detections "
            "will be labelled by their numeric class index until names are added."
        )

    in_w, in_h = DEFAULT_INPUT_SIZE
    shape = session.get_inputs()[0].shape
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        if isinstance(shape[3], int):
            in_w = shape[3]
        if isinstance(shape[2], int):
            in_h = shape[2]

    # The screening confidence floor tracks the station's own tracker activation
    # threshold, so the detector emits exactly the detections the tracker would
    # consider, and one configured value governs both.
    conf = float(station.get("capture", {}).get("bytetrack", {}).get("track_activation_threshold", DEFAULT_CONFIDENCE))

    return OnnxYoloDetector(
        session,
        class_names=class_names,
        input_size=(in_w, in_h),
        conf_threshold=conf,
    )
