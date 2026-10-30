import hashlib
import json
import threading
from functools import lru_cache

import numpy as np
from PIL import Image
from backend.config import manifest_path, model_path
from backend.errors import ServiceError
from backend.geometry import letterbox, point_to_basic
from backend.schemas import ViewSpec
from backend.tiling import TILE, detect_view

_session_lock = threading.Lock()
# Evaluated and rejected models (see docs/BASELINE_V1.md). Never serve these.
REJECTED_MODEL_SHA256 = {'b7cb1fc9851b674467aeeb0f43224a74f9373434c6b6e46bcaec692463ac57ae'}
# yolo26-end2end: our own exports, NMS built in, rows [x1,y1,x2,y2,conf,cls].
# yolov5: the pretrained pLitter model, raw rows [cx,cy,w,h,objectness,class scores...] without NMS.
FORMATS = ('yolo26-end2end', 'yolov5')
YOLOV5_NMS_IOU = .45
YOLOV5_MAX_ROWS = 300


def output_width(info: dict) -> int:
    return 5 + len(info['class_map']) if info['format'] == 'yolov5' else 6


def _valid_class_map(class_map) -> bool:
    return (isinstance(class_map, dict) and sorted(class_map) == [str(i) for i in range(len(class_map))]
            and all(v in ('litter', None) for v in class_map.values()) and 'litter' in class_map.values())


def yolov5_rows(raw: np.ndarray, class_map: dict, threshold: float, nms_iou: float = YOLOV5_NMS_IOU) -> np.ndarray:
    """Raw YOLOv5 output [1, N, 5+classes] -> [1, M, 6] litter rows, like an end-to-end export.

    Score = objectness x class score, over the classes the manifest maps to litter (pLitter's
    trash bin class is dropped). Overlapping boxes are then removed with NMS.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 5 + len(class_map):
        raise ServiceError('model_output_invalid', 'The model produced an unsupported output shape.')
    raw = raw[0][np.isfinite(raw[0]).all(axis=1)]
    litter = np.array([class_map[str(i)] == 'litter' for i in range(len(class_map))])
    scores = (raw[:, 4:5] * raw[:, 5:])[:, litter].max(axis=1)
    raw, scores = raw[scores >= threshold], scores[scores >= threshold]
    boxes = np.stack([raw[:, 0] - raw[:, 2] / 2, raw[:, 1] - raw[:, 3] / 2, raw[:, 0] + raw[:, 2] / 2, raw[:, 1] + raw[:, 3] / 2], axis=1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order, kept = np.argsort(-scores), []
    while order.size and len(kept) < YOLOV5_MAX_ROWS:
        best, order = order[0], order[1:]
        kept.append(best)
        inter = (np.clip(np.minimum(boxes[best, 2], boxes[order, 2]) - np.maximum(boxes[best, 0], boxes[order, 0]), 0, None)
                 * np.clip(np.minimum(boxes[best, 3], boxes[order, 3]) - np.maximum(boxes[best, 1], boxes[order, 1]), 0, None))
        order = order[inter / (area[best] + area[order] - inter + 1e-9) < nms_iou]
    rows = np.zeros((1, len(kept), 6), dtype=np.float32)
    rows[0, :, :4], rows[0, :, 4] = boxes[kept], scores[kept]
    return rows


def output_rows(info: dict):
    """raw model output -> [1, N, 6] rows [x1,y1,x2,y2,conf,0] in letterboxed pixels."""
    if info['format'] != 'yolov5':
        return lambda raw: raw
    threshold, nms_iou = float(info.get('confidence_threshold', .35)), float(info.get('nms_iou', YOLOV5_NMS_IOU))
    return lambda raw: yolov5_rows(raw, info['class_map'], threshold, nms_iou)

def manifest() -> dict:
    try:
        data = json.loads(manifest_path().read_text(encoding='utf-8'))
        if not model_path().is_file() or data.get('trained') is not True or data.get('classes') != ['litter'] or data.get('format') not in FORMATS or data.get('input_size') != 640 or not data.get('version') or len(data.get('sha256', '')) != 64:
            raise ValueError('Model not configured')
        if data['format'] == 'yolov5' and (not _valid_class_map(data.get('class_map')) or not .1 <= float(data.get('nms_iou', YOLOV5_NMS_IOU)) <= .9):
            raise ValueError('Invalid class map')
        if data['sha256'] in REJECTED_MODEL_SHA256:
            raise ValueError('Rejected model')
        threshold = float(data.get('confidence_threshold', .35))
        if not .05 <= threshold <= .95:
            raise ValueError('Invalid threshold')
        tiles = data.get('tiling')
        if tiles is not None:
            face, tile, stride = (tiles.get(k) for k in ('face_size', 'tile', 'stride'))
            if not all(isinstance(v, int) for v in (face, tile, stride)) or tile != TILE or not TILE < face <= 2048 or not 0 < stride <= tile:
                raise ValueError('Invalid tiling')
        return data
    except (OSError, ValueError, TypeError):
        raise ServiceError('model_not_ready', 'The trained litter model has not been installed yet. Street exploration is still available.') from None

@lru_cache(maxsize=1)
def load_session(path: str, expected_sha: str, width: int = 6):
    import onnxruntime as ort
    with _session_lock:
        with open(path, 'rb') as file:
            actual = hashlib.file_digest(file, 'sha256').hexdigest()
        if actual != expected_sha:
            raise ServiceError('model_checksum_failed', 'The model does not match its training manifest.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or len(outputs) != 1 or outputs[0].shape[-1] != width:
                raise ValueError('Unsupported model signature')
            return session
        except Exception:
            raise ServiceError('model_load_failed', 'The model export is incompatible. Export with a fixed 640-pixel input and the output format named in the manifest.') from None

def decode(output: np.ndarray, image: Image.Image, scale: float, left: int, top: int, view: ViewSpec, threshold: float):
    rows = np.asarray(output)
    if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[2] != 6:
        raise ServiceError('model_output_invalid', 'The model produced an unsupported output shape.')
    detections = []
    for row in sorted(rows[0], key=lambda r: -float(r[4])):
        x1, y1, x2, y2, confidence, label = (float(v) for v in row)
        if not np.isfinite(row).all() or confidence < threshold or confidence > 1 or label != 0:
            continue
        x1, x2 = (np.clip((x - left) / scale / image.width, 0, 1) for x in (x1, x2))
        y1, y2 = (np.clip((y - top) / scale / image.height, 0, 1) for y in (y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({'id': f'litter-{len(detections)+1}', 'label': 'litter', 'confidence': round(confidence, 4), 'box': [float(x1), float(y1), float(x2), float(y2)], 'tag': point_to_basic((x1+x2)/2, (y1+y2)/2, view)})
        if len(detections) >= 100:
            break
    return detections

def view_size(info: dict) -> int:
    """Longest side to render each view at: 640, or larger when the model is used with tiling."""
    return (info.get('tiling') or {}).get('face_size', TILE)


def _run(session, tensor):
    try:
        return session.run(None, {session.get_inputs()[0].name: tensor})[0]
    except Exception:
        raise ServiceError('inference_failed', 'The model could not finish this scan. Please retry.') from None


def detect(image: Image.Image, view: ViewSpec, info: dict):
    session = load_session(str(model_path()), info['sha256'], output_width(info))
    threshold = float(info.get('confidence_threshold', .35))
    rows = output_rows(info)
    tiles = info.get('tiling')
    if tiles:
        pairs = detect_view(image, lambda tensor: rows(_run(session, tensor))[0], threshold, tiles['tile'], tiles['stride'])
        return [{'id': f'litter-{i+1}', 'label': 'litter', 'confidence': round(confidence, 4), 'box': [float(v) for v in box],
                 'tag': point_to_basic((box[0]+box[2])/2, (box[1]+box[3])/2, view)}
                for i, (box, confidence) in enumerate(pairs[:100])]
    tensor, scale, left, top = letterbox(image)
    return decode(rows(_run(session, tensor)), image, scale, left, top, view, threshold)
