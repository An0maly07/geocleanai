"""Evaluate labelled held-out Pune views using the exact deployed preprocessing.

Input: folder/images/*.jpg and folder/labels/*.txt (YOLO format, class 0).
Every image, including clean images, must have an explicit label file.

--model accepts the deployed ONNX file or, for validation sweeps only, a PyTorch .pt
checkpoint run on the identical letterboxed input.

The scoring rule is unchanged from the first baseline run: predictions are taken in
descending confidence order and each claims the unmatched label box with the highest
IoU, counting as a true positive when that IoU is at least the threshold.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.geometry import letterbox
from backend.tiling import detect_view

INPUT_SIZE = 640
PARITY_MIN_IOU = .99
# Decided 26 Sept 2026 after the baseline diagnosis: compare boxes after clipping both to the
# image (as the app does), and accept sub-pixel rounding on small boxes.
PARITY_MAX_CORNER_PX = 1.0
PARITY_RULE = 'counts equal; boxes clipped to the image; each pair IoU >= 0.99 or every corner within 1 px'
# Shorter box side in detector input pixels.
SIZE_BUCKETS = (('<8', 0, 8), ('8-16', 8, 16), ('16-32', 16, 32), ('>=32', 32, float('inf')))


def iou(a, b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    # Model coordinates are NumPy scalars; keep report values JSON-compatible.
    return float(inter / union) if union > 0 else 0.0


def read_labels(label_file: Path) -> list[list[float]]:
    """YOLO centre-format labels as normalized [x1, y1, x2, y2]. Empty file = verified clean."""
    truths = []
    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        if cls != 0:
            raise ValueError('Evaluation labels must use class 0: litter.')
        truths.append([x-w/2, y-h/2, x+w/2, y+h/2])
    return truths


def decode_rows(rows, width: int, height: int, scale: float, left: int, top: int, confidence: float):
    """End-to-end model rows in letterboxed pixels -> [(normalized box, confidence)], highest first."""
    predictions = []
    for row in sorted(rows, key=lambda r: -r[4]):
        if row[4] < confidence or row[5] != 0:
            continue
        box = [(row[0]-left)/scale/width, (row[1]-top)/scale/height, (row[2]-left)/scale/width, (row[3]-top)/scale/height]
        predictions.append(([float(v) for v in np.clip(box, 0, 1)], float(row[4])))
    return predictions


def match(predictions: list[list[float]], truths: list[list[float]], iou_threshold: float = .5):
    """Greedy one-to-one matching; predictions must already be in descending confidence order.

    Returns (prediction index, truth index, IoU) for each true positive.
    """
    unmatched = set(range(len(truths)))
    matches = []
    for p, box in enumerate(predictions):
        # Sorted so exact IoU ties always resolve to the lowest label index.
        best = max(sorted(unmatched), key=lambda i: iou(box, truths[i]), default=None)
        if best is not None:
            overlap = iou(box, truths[best])
            if overlap >= iou_threshold:
                unmatched.remove(best)
                matches.append((p, best, overlap))
    return matches


def short_side_px(box: list[float], width: int, height: int, input_size: int = INPUT_SIZE) -> float:
    scale = input_size / max(width, height)
    return min((box[2]-box[0])*width, (box[3]-box[1])*height) * scale


def size_bucket(side: float) -> str:
    return next(name for name, low, high in SIZE_BUCKETS if low <= side < high)


def parity(pt_rows, ort_rows, min_iou: float = PARITY_MIN_IOU, bounds=None) -> dict:
    """Compare PyTorch and ONNX rows (letterboxed pixels) with strict one-to-one pairing.

    Pairs are formed greedily by highest IoU across all pairs, so no ONNX box can be reused.
    bounds = (x1, y1, x2, y2) of the real image inside the letterboxed input: both sets of boxes
    are clipped to it first, because PyTorch clips and raw ONNX output does not. A pair passes at
    IoU >= min_iou or with every corner within PARITY_MAX_CORNER_PX. The old unclipped, IoU-only
    result is kept as passed_strict_unclipped.
    """
    strict = _pair_rows(pt_rows, ort_rows)
    if bounds is not None:
        clip = lambda rows: [[*np.clip(r[:4], [bounds[0], bounds[1]] * 2, [bounds[2], bounds[3]] * 2), *r[4:]] for r in rows]
        pt_rows, ort_rows = clip(pt_rows), clip(ort_rows)
    report = _pair_rows(pt_rows, ort_rows)
    report['passed'] = len(pt_rows) == len(ort_rows) and all(p['iou'] >= min_iou or p['max_corner_px'] <= PARITY_MAX_CORNER_PX for p in report['pairs'])
    report['passed_strict_unclipped'] = len(strict['pairs']) == len(pt_rows) == len(ort_rows) and all(p['iou'] >= min_iou for p in strict['pairs'])
    return report


def _pair_rows(pt_rows, ort_rows) -> dict:
    candidates = sorted(((iou(a[:4], b[:4]), i, j) for i, a in enumerate(pt_rows) for j, b in enumerate(ort_rows)), reverse=True)
    used_pt, used_ort, pairs = set(), set(), []
    for overlap, i, j in candidates:
        if i in used_pt or j in used_ort:
            continue
        used_pt.add(i); used_ort.add(j)
        a, b = pt_rows[i], ort_rows[j]
        pairs.append({'iou': float(overlap),
                      'max_corner_px': float(max(abs(float(a[k]) - float(b[k])) for k in range(4))),
                      'confidence_diff': float(abs(float(a[4]) - float(b[4]))),
                      'pt_box_short_side_px': float(min(a[2]-a[0], a[3]-a[1]))})
    return {'pt_count': len(pt_rows), 'onnx_count': len(ort_rows),
            'unpaired_pt': len(pt_rows) - len(used_pt), 'unpaired_onnx': len(ort_rows) - len(used_ort),
            'pairs': pairs,
            'min_matched_iou': min((p['iou'] for p in pairs), default=1.0),
            'max_corner_px': max((p['max_corner_px'] for p in pairs), default=0.0),
            'max_confidence_diff': max((p['confidence_diff'] for p in pairs), default=0.0)}


def summarize(records: list[dict]) -> dict:
    """Totals, size-bucket recall and clean-view false alarms from per-image records."""
    tp = sum(r['tp'] for r in records); fp = sum(r['fp'] for r in records); fn = sum(r['fn'] for r in records)
    buckets = {name: {'labels': 0, 'matched': 0} for name, _, _ in SIZE_BUCKETS}
    for record in records:
        matched = {m['truth'] for m in record['matches']}
        for index, truth in enumerate(record['truths']):
            buckets[truth['size_bucket']]['labels'] += 1
            buckets[truth['size_bucket']]['matched'] += index in matched
    for bucket in buckets.values():
        bucket['recall'] = bucket['matched'] / bucket['labels'] if bucket['labels'] else None
    clean = [r for r in records if not r['truths']]
    return {'true_positives': tp, 'false_positives': fp, 'false_negatives': fn, 'images': len(records),
            'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
            'recall_by_size': buckets,
            'clean_views': len(clean),
            'clean_view_false_positives': sum(r['fp'] for r in clean),
            'clean_views_with_false_alarm': sum(r['fp'] > 0 for r in clean)}


def wilson(successes: int, total: int, z: float = 1.96):
    """95% Wilson interval for a proportion; ignores correlation, so it is a lower bound on uncertainty."""
    if not total:
        return None
    p = successes / total
    centre = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * ((p*(1-p)/total + z*z/(4*total*total)) ** .5) / (1 + z*z/total)
    return [round(centre-half, 4), round(centre+half, 4)]


def panorama_bootstrap(records: list[dict], samples: int = 2000, seed: int = 0) -> dict:
    """Resample whole panoramas (views from one panorama are correlated) for 95% intervals."""
    panoramas = {}
    for record in records:
        totals = panoramas.setdefault(record['image'].split('_')[0], [0, 0, 0])
        for index, key in enumerate(('tp', 'fp', 'fn')):
            totals[index] += record[key]
    keys = sorted(panoramas)
    rng = np.random.default_rng(seed)
    precision, recall = [], []
    for _ in range(samples):
        tp, fp, fn = np.sum([panoramas[keys[i]] for i in rng.integers(0, len(keys), len(keys))], axis=0)
        if tp + fp:
            precision.append(tp/(tp+fp))
        if tp + fn:
            recall.append(tp/(tp+fn))
    interval = lambda values: [round(float(v), 4) for v in np.percentile(values, [2.5, 97.5])] if values else None
    return {'panoramas': len(keys), 'samples': samples, 'precision_95': interval(precision), 'recall_95': interval(recall)}


def _sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def sweep(records: list[dict], thresholds, iou_threshold: float = .5) -> list[dict]:
    """Re-score saved low-threshold predictions at each confidence threshold."""
    rows = []
    for threshold in thresholds:
        rescored = []
        for record in records:
            kept = [p['box'] for p in record['predictions'] if p['confidence'] >= threshold]
            matches = match(kept, [t['box'] for t in record['truths']], iou_threshold)
            rescored.append({'tp': len(matches), 'fp': len(kept)-len(matches), 'fn': len(record['truths'])-len(matches),
                             'truths': record['truths'], 'matches': [{'truth': t} for _, t, _ in matches]})
        summary = summarize(rescored)
        rows.append({'threshold': float(threshold), **{k: summary[k] for k in (
            'true_positives', 'false_positives', 'false_negatives', 'precision', 'recall',
            'recall_by_size', 'clean_view_false_positives')}})
    return rows


def choose_threshold(rows: list[dict], min_precision: float = .85) -> dict:
    """Lowest threshold meeting the precision floor; otherwise the maximum-F1 threshold."""
    qualifying = [r for r in rows if r['precision'] is not None and r['precision'] >= min_precision]
    if qualifying:
        return {**min(qualifying, key=lambda r: r['threshold']), 'rule': f'lowest threshold with precision >= {min_precision}'}
    f1 = lambda r: 2*r['precision']*r['recall']/(r['precision']+r['recall']) if r['precision'] and r['recall'] else 0.0
    return {**max(rows, key=lambda r: (f1(r), -r['threshold'])), 'rule': f'no threshold reached precision {min_precision}; maximum F1'}


def load_runner(model_path: Path, confidence: float, manifest: Path | None = None):
    """Return (run(tensor) -> rows in letterboxed pixels, versions). ONNX for deployment, .pt for validation sweeps.

    manifest: the model's manifest.json, required for raw YOLOv5 exports (the pretrained pLitter model),
    whose output is decoded and NMS-filtered with the deployed backend code.
    """
    if Path(model_path).suffix == '.pt':
        import ultralytics
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        def run(tensor):
            square = Image.fromarray((tensor[0].transpose(1,2,0)*255).round().astype('uint8'))
            return model.predict(square, imgsz=INPUT_SIZE, rect=False, conf=confidence, nms=False, max_det=100, verbose=False)[0].boxes.data.cpu().numpy()
        return run, {'ultralytics': getattr(ultralytics, '__version__', None)}
    import onnxruntime as ort
    from backend.detector import output_rows  # imported here: it loads app config, which Colab may not need
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    info = json.loads(Path(manifest).read_text(encoding='utf-8')) if manifest else {'format': 'yolo26-end2end'}
    # Decode at the evaluation threshold, not the deployed one, so sweeps see every candidate.
    rows = output_rows({**info, 'confidence_threshold': confidence})
    def run(tensor):
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if info['format'] != 'yolov5' and np.shape(output)[-1] != 6:
            raise ValueError('This model has raw YOLOv5 output. Pass --manifest with its manifest.json.')
        return rows(output)[0]
    return run, {'onnxruntime': getattr(ort, '__version__', None)}


def evaluate(args):
    iou_threshold = getattr(args, 'iou', .5)
    split = getattr(args, 'split', None)
    image_dir = args.dataset/'images'/split if split else args.dataset/'images'
    label_dir = args.dataset/'labels'/split if split else args.dataset/'labels'
    run, versions = load_runner(args.model, args.confidence, getattr(args, 'manifest', None))
    records, parity_records = [], []
    torch_model = None
    if getattr(args, 'pt', None):
        import ultralytics
        from ultralytics import YOLO
        torch_model = YOLO(str(args.pt))
        versions['ultralytics'] = getattr(ultralytics, '__version__', None)
    tiled = getattr(args, 'tiled', False)
    if tiled and torch_model:
        raise ValueError('Check ONNX/PyTorch parity on untiled 640px views; tiling reuses the same model per tile.')
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in ('.jpg','.jpeg','.png'):
            continue
        label_file = label_dir/f'{path.stem}.txt'
        if not label_file.is_file():
            raise ValueError(f'Missing reviewed label file for {path.name}. Use an empty file for a verified clean view.')
        image = Image.open(path).convert('RGB')
        started = time.perf_counter()
        if tiled:
            # The same tiling code as the deployed API; detector scale is the rendered view size.
            predictions = detect_view(image, lambda tensor: np.asarray(run(tensor)).reshape(-1, 6), args.confidence, stride=getattr(args, 'stride', 384))
            detector_px = max(image.size)
        else:
            tensor, scale, left, top = letterbox(image)
            rows = np.asarray(run(tensor)).reshape(-1, 6)
            predictions = decode_rows(rows, image.width, image.height, scale, left, top, args.confidence)
            detector_px = INPUT_SIZE
        latency = (time.perf_counter()-started)*1000
        truths = read_labels(label_file)
        matches = match([box for box, _ in predictions], truths, iou_threshold)
        tp = len(matches)
        fp, fn = len(predictions)-tp, len(truths)-tp
        records.append({'image': path.name, 'tp': tp, 'fp': fp, 'fn': fn, 'latency_ms': round(latency,1),
                        'predictions': [{'box': box, 'confidence': conf} for box, conf in predictions],
                        'truths': [{'box': box, 'short_side_px': short_side_px(box, image.width, image.height, detector_px),
                                    'size_bucket': size_bucket(short_side_px(box, image.width, image.height, detector_px))} for box in truths],
                        'matches': [{'prediction': p, 'truth': t, 'iou': o} for p, t, o in matches]})
        if torch_model:
            # Compare on the identical letterboxed 640px RGB image, not a second resize.
            square = Image.fromarray((tensor[0].transpose(1,2,0)*255).round().astype('uint8'))
            boxes = torch_model.predict(square, imgsz=INPUT_SIZE, rect=False, conf=args.confidence, nms=False, max_det=100, device='cpu', verbose=False)[0].boxes
            pt_rows = boxes.data.cpu().numpy()
            ort_rows = rows[rows[:,4] >= args.confidence]
            bounds = (left, top, left + image.width * scale, top + image.height * scale)
            parity_records.append({'image': path.name, **parity(pt_rows, ort_rows, bounds=bounds)})
    if not records:
        raise ValueError('No evaluation images found.')
    lock = args.dataset/'evaluation-lock.json'
    summary = summarize(records)
    result = {**summary, 'iou_threshold': iou_threshold, 'confidence_threshold': args.confidence,
              'precision_wilson_95': wilson(summary['true_positives'], summary['true_positives']+summary['false_positives']),
              'recall_wilson_95': wilson(summary['true_positives'], summary['true_positives']+summary['false_negatives']),
              'panorama_bootstrap': panorama_bootstrap(records),
              'model_sha256': _sha256(Path(args.model)) if Path(args.model).is_file() else None,
              'evaluation_lock_sha256': _sha256(lock) if lock.is_file() else None,
              'versions': versions,
              'median_inference_ms': float(np.median([r['latency_ms'] for r in records])),
              'mode': f"tiled {max(image.size)}px views, 640px tiles, stride {getattr(args, 'stride', 384)}" if tiled else '640px views',
              'per_image': records, 'parity': parity_records, 'parity_rule': PARITY_RULE,
              'parity_passed': all(r['passed'] for r in parity_records) if parity_records else None,
              'parity_passed_strict_unclipped': all(r['passed_strict_unclipped'] for r in parity_records) if parity_records else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=_json_safe))
    print(f'Evaluated {len(records)} images. Precision={result["precision"]}; recall={result["recall"]}. Results: {args.output}')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/pune-evaluation.json'))
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--split', help='Subfolder of images/ and labels/, e.g. val for a YOLO dataset')
    parser.add_argument('--tiled', action='store_true', help='Views rendered above 640px: detect with overlapping 640px tiles')
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--pt', type=Path, help='Optional PyTorch best.pt for export parity checks')
    parser.add_argument('--manifest', type=Path, help="The model's manifest.json; required for raw YOLOv5 exports")
    evaluate(parser.parse_args())
