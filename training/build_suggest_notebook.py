"""Build notebook C: zero-shot litter box suggestions for labellers, scored before use.

The detector (OWLv2, prompted with litter descriptions) is never trained here and its boxes
never become labels by themselves: the labelling page shows them as suggestions that a person
accepts or rejects. The notebook first measures the suggestions against views the team already
labelled, so everyone knows how far to trust them.
"""
import argparse
import json
from pathlib import Path

from build_colab_notebooks_v2 import ROOT, notebook

QUERIES = ['a piece of litter on the ground', 'a plastic bottle', 'a plastic bag', 'a food wrapper',
           'a paper cup', 'a crumpled piece of paper', 'a discarded can', 'a cigarette butt',
           'a pile of garbage', 'trash on the street']


def cells(bundle: str, digest: str) -> list[tuple[str, str]]:
    return [
        ('markdown', '''
            # GeoCleanAI — C. AI box suggestions for labelling

            A ready-made open-vocabulary detector (**OWLv2**) looks for litter from text prompts. Nothing is trained.
            1. The suggestions are **scored against the views the team already labelled**, so you know how far to trust them.
            2. They are saved to `suggestions.json` for the labelling page, where a person accepts or rejects every box.

            Use a GPU runtime (Runtime → Change runtime type → T4). Then **Runtime → Run all**. About 30–45 minutes with tiles.
        '''),
        ('code', f'''
            from google.colab import drive
            drive.mount('/content/drive')
            import hashlib, json, time, zipfile
            from datetime import datetime, timezone
            from pathlib import Path
            DRIVE = Path('/content/drive/MyDrive/GeoCleanAI')
            BUNDLE, WORK = DRIVE/'{bundle}', Path('/content/suggest')
            OUT = DRIVE/'suggestions-{Path(bundle).stem}'; OUT.mkdir(exist_ok=True)
            assert BUNDLE.is_file(), 'Upload {bundle} into My Drive/GeoCleanAI first.'

            def file_hash(path):
                with Path(path).open('rb') as stream:
                    return hashlib.file_digest(stream, 'sha256').hexdigest()

            if not (WORK/'.ok').is_file():
                assert file_hash(BUNDLE) == '{digest}', '{bundle} is incomplete or a different version.'
                with zipfile.ZipFile(BUNDLE) as archive:
                    for name in archive.namelist():
                        assert (WORK/name).resolve().is_relative_to(WORK.resolve()), 'Unsafe archive path'
                    archive.extractall(WORK)
                (WORK/'.ok').write_text('ok')
            MANIFEST = json.loads((WORK/'suggest-manifest.json').read_text())
            VIEWS, LABELS = MANIFEST['views'], MANIFEST['labels']
            print(len(VIEWS), 'views;', len(LABELS), 'already labelled (used only to score the suggestions)')
        '''),
        ('code', f'''
            import torch, torchvision, transformers
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            assert torch.cuda.is_available(), 'Choose a GPU runtime: Runtime → Change runtime type → T4 GPU.'
            DETECTOR = 'google/owlv2-base-patch16-ensemble'
            QUERIES = {json.dumps(QUERIES)}
            processor = Owlv2Processor.from_pretrained(DETECTOR)
            model = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to('cuda').eval()
            print('transformers', transformers.__version__, '| torch', torch.__version__)
        '''),
        ('markdown', '''
            ## Detect
            Each 1280 px view is checked whole and as four overlapping tiles, because most Pune litter is tiny.
            Set `TILES = False` to halve the time. Progress is saved to Drive every 100 views; if Colab disconnects, run all again and it continues.
        '''),
        ('code', '''
            from PIL import Image
            from torchvision.ops import nms
            TILES, MIN_SCORE, BATCH = True, 0.05, 5  # one view and its four tiles per GPU batch
            # Boxes far larger than anything the team drew are scene-level guesses ("a pile of garbage" = the whole street).
            human = [b for label in LABELS.values() for b in label['boxes']]
            MAX_SIDE = min(1.0, 1.5 * max(max(b[2] - b[0], b[3] - b[1]) for b in human)) if human else 1.0

            def crops(image):
                size = image.width
                yield 0, 0, size, image
                if TILES:
                    tile = int(size * .6)  # 2x2 tiles overlapping by 20%
                    for x in (0, size - tile):
                        for y in (0, size - tile):
                            yield x, y, tile, image.crop((x, y, x + tile, y + tile))

            @torch.no_grad()
            def run(images):
                inputs = processor(text=[QUERIES] * len(images), images=images, return_tensors='pt').to('cuda')
                outputs = model(**inputs)
                # Class-agnostic: the best-matching prompt decides each box's score. Inputs are square, so
                # boxes normalised to the padded input are normalised to the crop.
                return outputs.logits.sigmoid().max(-1).values.cpu(), outputs.pred_boxes.cpu()

            def suggest(view):
                image = Image.open(WORK/'views'/view['file_name']).convert('RGB')
                parts = list(crops(image))
                found, scores = [], []
                for start in range(0, len(parts), BATCH):
                    chunk = parts[start:start + BATCH]
                    all_scores, all_boxes = run([crop for *_, crop in chunk])
                    for (x, y, size, _), s, b in zip(chunk, all_scores, all_boxes):
                        keep = s >= MIN_SCORE
                        cx, cy, w, h = b[keep].unbind(-1)
                        scale = size / image.width
                        found.append(torch.stack([(cx - w / 2) * scale + x / image.width, (cy - h / 2) * scale + y / image.width,
                                                  (cx + w / 2) * scale + x / image.width, (cy + h / 2) * scale + y / image.width], -1).clamp(0, 1))
                        scores.append(s[keep])
                boxes, scores = torch.cat(found), torch.cat(scores)
                sides = torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
                ok = (sides <= MAX_SIDE) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes, scores = boxes[ok], scores[ok]
                keep = nms(boxes, scores, 0.5)[:100]
                return [[round(float(v), 5) for v in box] + [round(float(score), 4)] for box, score in zip(boxes[keep], scores[keep])]

            partial = OUT/'partial.json'
            RESULTS = json.loads(partial.read_text()) if partial.is_file() else {}
            started = time.time()
            for i, view in enumerate(VIEWS):
                if view['file_name'] in RESULTS:
                    continue
                RESULTS[view['file_name']] = suggest(view)
                if len(RESULTS) % 100 == 0 or i == len(VIEWS) - 1:
                    partial.write_text(json.dumps(RESULTS))
                    print(f'{len(RESULTS)}/{len(VIEWS)} views · {(time.time() - started) / 60:.1f} min')
            print('Done. Largest allowed box side:', round(MAX_SIDE, 3))
        '''),
        ('markdown', '''
            ## Score the suggestions on the team's labels
            Same matching rule as model evaluation: one suggestion per labelled box, at IoU ≥ 0.5 (exact enough to accept as is) and ≥ 0.3 (roughly on the item).
            Sizes are in pixels at 640 px, matching the project's size buckets.
        '''),
        ('code', '''
            from torchvision.ops import box_iou

            def score(threshold, iou_needed):
                tp = fp = fn = clean_alarm_views = 0
                by_size = {'<8': [0, 0], '8-16': [0, 0], '16-32': [0, 0], '>=32': [0, 0]}
                for name, label in LABELS.items():
                    predicted = [b[:4] for b in RESULTS[name] if b[4] >= threshold]
                    truth = label['boxes']
                    matched = set()
                    if predicted and truth:
                        ious = box_iou(torch.tensor(predicted), torch.tensor(truth))
                        for p in range(len(predicted)):  # predictions are already sorted by score
                            best = max(((float(ious[p, t]), t) for t in range(len(truth)) if t not in matched), default=(0, None))
                            if best[0] >= iou_needed:
                                matched.add(best[1])
                    tp += len(matched); fp += len(predicted) - len(matched); fn += len(truth) - len(matched)
                    clean_alarm_views += label['status'] == 'clean' and bool(predicted)
                    for t, box in enumerate(truth):
                        side = max(box[2] - box[0], box[3] - box[1]) * 640
                        bucket = '<8' if side < 8 else '8-16' if side < 16 else '16-32' if side < 32 else '>=32'
                        by_size[bucket][0] += 1; by_size[bucket][1] += t in matched
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                return {'threshold': threshold, 'iou': iou_needed, 'tp': tp, 'fp': fp, 'fn': fn,
                        'precision': round(precision, 3), 'recall': round(recall, 3), 'f1': round(f1, 3),
                        'clean_views_with_suggestions': clean_alarm_views,
                        'recall_by_size': {k: f'{m}/{n}' for k, (n, m) in by_size.items()}}

            THRESHOLDS = [round(t / 100, 2) for t in range(5, 55, 5)]
            TABLE = [score(t, iou) for iou in (0.5, 0.3) for t in THRESHOLDS]
            clean_total = sum(label['status'] == 'clean' for label in LABELS.values())
            for row in TABLE:
                print(f"IoU {row['iou']} · threshold {row['threshold']:.2f} · precision {row['precision']:.2f} · recall {row['recall']:.2f}"
                      f" · F1 {row['f1']:.2f} · clean views with suggestions {row['clean_views_with_suggestions']}/{clean_total} · by size {row['recall_by_size']}")
            CHOSEN = max((r for r in TABLE if r['iou'] == 0.5), key=lambda r: (r['f1'], r['precision']))
            print('\\nSuggestion threshold (best F1 at IoU 0.5):', json.dumps(CHOSEN, indent=1))
            if CHOSEN['precision'] < 0.3:
                print('WARNING: fewer than 3 in 10 suggestions are right. Labellers will mostly reject them; treat them as hints only.')
        '''),
        ('markdown', '## Save and download `suggestions.json`\nPut the downloaded file in your project folder and run the import command from `docs/ASSISTED_LABELLING.md`.'),
        ('code', '''
            result = {'detector': DETECTOR, 'queries': QUERIES, 'tiles': TILES, 'view_size': MANIFEST['view_size'],
                      'threshold': CHOSEN['threshold'], 'evaluation': {'chosen': CHOSEN, 'table': TABLE,
                      'labelled_views': len(LABELS), 'note': 'Scored on human labels of the same batch (train and validation views).'},
                      'created_at': datetime.now(timezone.utc).isoformat(),
                      'versions': {'transformers': transformers.__version__, 'torch': torch.__version__},
                      'views': {v['file_name']: {'sha256': v['sha256'], 'boxes': RESULTS[v['file_name']]} for v in VIEWS}}
            (OUT/'suggestions.json').write_text(json.dumps(result))
            from google.colab import files
            files.download(str(OUT/'suggestions.json'))
            print('Saved', OUT/'suggestions.json')
        '''),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/colab/geocleanai-suggest-v1.zip')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/colab/GeoCleanAI_C_Suggestions.ipynb')
    args = parser.parse_args()
    digest = args.bundle.with_name(args.bundle.name + '.sha256').read_text().strip()
    built = notebook(cells(args.bundle.name, digest), args.output.stem)
    args.output.write_text(json.dumps(built, indent=1), encoding='utf-8')
    print(f'Wrote {args.output} ({len(built["cells"])} cells; all code cells compile)')


if __name__ == '__main__':
    main()
