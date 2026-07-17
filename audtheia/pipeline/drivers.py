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
import json
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
    "OnnxRfDetrDetector",
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
    s = _unquote_spec(spec)
    lowered = s.lower()
    if lowered.startswith("webcam"):
        rest = s.split(":", 1)[1].strip() if ":" in s else ""
        return (int(rest) if rest.isdigit() else 0), True
    if lowered.startswith("stream:"):
        return _unquote_spec(s.split(":", 1)[1]), True
    if lowered.startswith("url:"):
        return _unquote_spec(s.split(":", 1)[1]), True
    if lowered.startswith("file:"):
        return _unquote_spec(s.split(":", 1)[1]), False
    if s.isdigit():
        return int(s), True
    if lowered.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://")):
        return s, True
    return s, False


def _unquote_spec(text) -> str:
    """Strip one matched pair of surrounding quotes, then whitespace.

    Windows "Copy as path" wraps a path in double quotes, and users paste that,
    so a quoted `file:"C:\\clip.mp4"` must resolve like the bare path.
    """
    t = str(text).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        t = t[1:-1].strip()
    return t


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
    low = video.strip().lower()
    cv2 = _import_cv2()
    if low.startswith("stream:"):
        # A page address (a YouTube or Pixcams live page) is not a media stream
        # OpenCV can open; yt-dlp resolves it to the direct stream first.
        target = _resolve_stream_url(target)
    capture = cv2.VideoCapture(target)
    # A plain web address pasted without a prefix may be a web *page* (a YouTube
    # link, a live-cam page) rather than a direct media stream. If such a source
    # does not open directly, resolve it through yt-dlp and try once more, so a
    # webcam, a direct stream URL, a page link, or a file all work whether or not
    # the user prefixes them. An explicit 'url:' is honored as a direct address
    # and is never second-guessed; an explicit 'stream:' was already resolved.
    if (not capture.isOpened()
            and not low.startswith(("url:", "stream:"))
            and low.startswith(("http://", "https://"))):
        try:
            resolved = _resolve_stream_url(target)
        except Exception:  # noqa: BLE001 - yt-dlp missing or resolution failed
            resolved = None
        if resolved and resolved != target:
            capture = cv2.VideoCapture(resolved)
    if not capture.isOpened():
        raise CaptureError(
            f"could not open the video source {video!r}. Check the camera index, "
            f"the stream address, or the file path."
        )

    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = reported_fps if reported_fps and reported_fps > 0 else station.get("capture", {}).get("fps")

    # A finite recording reports a positive total frame count; a true live camera
    # or feed reports zero or less. When a source the spec marked live is in fact
    # a finite recording with a known frame rate (for example a saved clip reached
    # through a 'stream:' page that resolves to a video-on-demand), its frames are
    # timestamped from that frame rate rather than the wall clock, so a clip the
    # desktop processes slowly still yields event durations equal to the real
    # elapsed video time instead of the processing time. A genuine live source
    # keeps wall-clock timestamps, which are its true time base.
    if live and fps and fps > 0:
        try:
            frame_total = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:  # noqa: BLE001 - a source that cannot report a count stays live
            frame_total = 0.0
        if frame_total > 0:
            live = False

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


# ===========================================================================
# Detector: an RF-DETR model run through ONNX Runtime on the desktop
#
# The same RF-DETR export that verifies can also screen, so a station whose only
# model is RF-DETR still drives desktop capture. These are model-family constants
# for RF-DETR, matching the verifier adapter's preprocessing.
# ===========================================================================

RFDETR_INPUT_SIZE = (560, 560)
RFDETR_MEAN = (0.485, 0.456, 0.406)
RFDETR_STD = (0.229, 0.224, 0.225)


def _rfdetr_sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _split_rfdetr_outputs(outputs):
    """Return (boxes (N,4), logits (N,classes)) from an RF-DETR export's outputs.

    A standard RF-DETR export returns two two-dimensional tensors: per-query boxes
    (a last dimension of four) and per-query class logits. Either order is
    accepted, so the box tensor is the one whose last dimension is four.
    """
    boxes = None
    logits = None
    for out in outputs:
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            continue
        if arr.shape[1] == 4:
            boxes = arr
        else:
            logits = arr
    return boxes, logits


class OnnxRfDetrDetector:
    """A per-frame screening detector that runs an RF-DETR model through ONNX Runtime.

    RF-DETR emits per-query class logits, scored with a sigmoid, and center-form
    boxes. This decodes both into the same per-object boxes the tracker consumes
    from the YOLO detector, so either model family drives desktop capture with no
    other change. The session is any object exposing ONNX Runtime's run and
    get_inputs, so the decode is testable against a scripted output with no model.
    """

    def __init__(self, session, *, class_names: dict, input_size: tuple = RFDETR_INPUT_SIZE,
                 conf_threshold: float = DEFAULT_CONFIDENCE, iou_threshold: float = DEFAULT_NMS_IOU,
                 input_name: Optional[str] = None) -> None:
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
        blob = self._preprocess(frame.image)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._postprocess(outputs, frame.image.shape[1], frame.image.shape[0])

    def close(self) -> None:
        self._session = None

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize to the model's square input and normalize, as RF-DETR expects."""
        cv2 = _import_cv2()
        resized = cv2.resize(image_rgb, (self._in_w, self._in_h))
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - np.array(RFDETR_MEAN, dtype=np.float32)) / np.array(RFDETR_STD, dtype=np.float32)
        blob = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def _postprocess(self, outputs, orig_w, orig_h) -> list:
        boxes_t, logits_t = _split_rfdetr_outputs(outputs)
        if boxes_t is None or logits_t is None or logits_t.size == 0:
            return []
        scores = _rfdetr_sigmoid(logits_t)  # (queries, classes)
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        keep = confidences >= self._conf
        if not np.any(keep):
            return []
        boxes = boxes_t[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        # RF-DETR normally emits boxes normalized to [0, 1]; some exports emit
        # input-pixel coordinates. When the values are clearly larger than one, they
        # are input pixels, so bring them back to normalized before scaling out.
        if float(np.max(np.abs(boxes))) > 2.0:
            cx, bw = cx / self._in_w, bw / self._in_w
            cy, bh = cy / self._in_h, bh / self._in_h
        x1 = np.clip((cx - bw / 2) * orig_w, 0, orig_w)
        y1 = np.clip((cy - bh / 2) * orig_h, 0, orig_h)
        x2 = np.clip((cx + bw / 2) * orig_w, 0, orig_w)
        y2 = np.clip((cy + bh / 2) * orig_h, 0, orig_h)

        detections = []
        for i in OnnxYoloDetector._nms(np.stack([x1, y1, x2, y2], axis=1), confidences, self._iou):
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


def _looks_like_rfdetr(session) -> bool:
    """Whether a loaded model looks like an RF-DETR export rather than YOLO.

    An RF-DETR export returns two tensors (boxes and class logits); a YOLO export
    returns one. The output count is the reliable discriminator, and it lets a
    station screen with whichever detector family the placed model belongs to.
    """
    try:
        return len(session.get_outputs()) >= 2
    except Exception:  # noqa: BLE001 - an odd session falls back to the YOLO decode
        return False


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


def _labels_from_file(path: Path) -> dict:
    """Read an index-to-name map from a labels file placed beside the model.

    A `.json` file may hold either an index-ordered list of names or an explicit
    {index: name} map. A `.txt` or `.names` file holds one name per line, in class
    order, with blank lines and lines beginning with '#' ignored. An unreadable or
    empty file yields an empty map, so the caller falls back cleanly.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in parsed.items()}
        if isinstance(parsed, (list, tuple)):
            return {i: str(v) for i, v in enumerate(parsed)}
        return {}
    names = [ln.strip() for ln in text.splitlines()]
    names = [n for n in names if n and not n.startswith("#")]
    return {i: n for i, n in enumerate(names)}


def _class_names_from_config(entry: dict, model_path: Path) -> dict:
    """Resolve class names from the model's settings entry or a sibling file.

    This is the path for a model whose ONNX carries no embedded names, such as an
    RF-DETR export. Names are taken, in order of preference, from an inline
    `class_names` list or map on the model entry, then from an explicit
    `labels_path`, then from a file sitting beside the model named after it
    (`<model>.labels.json`, `<model>.labels.txt`, or `<model>.names`). The first
    source that yields any names wins, so a deployment can point the interface at
    names without re-exporting the model.
    """
    inline = entry.get("class_names")
    if isinstance(inline, dict):
        return {int(k): str(v) for k, v in inline.items()}
    if isinstance(inline, (list, tuple)):
        return {i: str(v) for i, v in enumerate(inline)}

    candidates: list[Path] = []
    labels_path = entry.get("labels_path")
    if labels_path:
        p = Path(labels_path)
        candidates.append(p if p.is_absolute() else model_path.parent / p)
    stem = model_path.stem
    for suffix in (".labels.json", ".labels.txt", ".names"):
        candidates.append(model_path.parent / f"{stem}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            names = _labels_from_file(candidate)
            if names:
                return names
    return {}


def build_detector(settings, station: dict):
    """Load the station's desktop screening model into an ONNX Runtime detector.

    The model may be a YOLO or an RF-DETR export; the loader detects which and
    returns the matching detector, so a station screens with whatever ONNX
    classifier is placed for it.
    """
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
        class_names = _class_names_from_config(entry, model_path)
    if not class_names:
        logger.warning(
            "the desktop detection model has no class-name metadata and no labels "
            "file beside it; detections will be labelled by their numeric class "
            "index. Add names inline on the model's settings entry (class_names), "
            "or place '%s.labels.json' / '%s.labels.txt' next to the model.",
            model_path.stem,
            model_path.stem,
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

    # A station may place a YOLO or an RF-DETR ONNX; the model's own output shape
    # decides which decoder screens with it, so either family works with no other
    # configuration.
    if _looks_like_rfdetr(session):
        return OnnxRfDetrDetector(
            session,
            class_names=class_names,
            input_size=(in_w, in_h),
            conf_threshold=conf,
        )
    return OnnxYoloDetector(
        session,
        class_names=class_names,
        input_size=(in_w, in_h),
        conf_threshold=conf,
    )
