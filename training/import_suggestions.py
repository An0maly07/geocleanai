"""Check suggestions.json from notebook C and install it for the labelling page.

Suggestions are matched to images by checksum: a view whose image changed gets none. The file
is installed as <batch>/suggestions.json; an earlier one is kept with a timestamp.
"""
import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def install(source: Path, batch: Path) -> dict:
    data = json.loads(source.read_text(encoding='utf-8'))
    for key in ('detector', 'threshold', 'views', 'evaluation'):
        if key not in data:
            raise ValueError(f'{source.name} is missing "{key}"; use the file downloaded from notebook C.')
    if not .01 <= float(data['threshold']) <= .99:
        raise ValueError('The suggestion threshold must be between 0.01 and 0.99.')
    manifest = json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
    local = {view['file_name']: sha256(batch/'images'/view['file_name']) for view in manifest['views']}
    matched, changed, unknown, boxes = 0, [], [], 0
    for name, item in data['views'].items():
        if name not in local:
            unknown.append(name)
            continue
        if item['sha256'] != local[name]:
            changed.append(name)
            continue
        for box in item['boxes']:
            if len(box) != 5 or not all(math.isfinite(v) for v in box) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError(f'Invalid suggestion box in {name}: {box}')
        matched += 1
        boxes += sum(box[4] >= data['threshold'] for box in item['boxes'])
    if unknown:
        raise ValueError(f'{len(unknown)} suggested views are not in {batch.name}, e.g. {unknown[0]}; wrong batch?')
    target = batch/'suggestions.json'
    if target.is_file():
        shutil.copy2(target, batch/f"suggestions-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json")
    shutil.copyfile(source, target)
    chosen = data['evaluation'].get('chosen', {})
    return {'installed': str(target), 'views_with_suggestions': matched, 'skipped_changed_images': len(changed),
            'boxes_shown_at_threshold': boxes, 'threshold': data['threshold'],
            'scored_precision': chosen.get('precision'), 'scored_recall': chosen.get('recall'),
            'next': 'Restart the labelling page: python -m training.review_server --data ' + batch.as_posix()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='suggestions.json downloaded from notebook C')
    parser.add_argument('--batch', type=Path, default=Path('data/pune-label-v2'))
    args = parser.parse_args()
    print(json.dumps(install(args.file, args.batch), indent=2))
