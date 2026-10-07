"""Build the Pune-focused dataset for the second fine-tuning stage (pune-focus-v1). No model involved.

- train: every Pune training view, repeated, plus a fixed random sample of TACO photos and
  street-like TACO tiles (litter-v2), so the model keeps what it learned about litter in general.
- val: the Pune validation split, so early stopping and best.pt are judged on Pune streets.
- test: the TACO test photos, only for train.py's supporting TACO metric.
TACO photos keep their v1 names and bytes, so the Colab bundle restores them from the v1 data
zip instead of uploading them again (see package_colab_v2.py).
"""
import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def checked_split(folder: Path, role: str) -> list[dict]:
    lock = json.loads((folder/'evaluation-lock.json').read_text())
    if lock['role'] != role:
        raise ValueError(f'{folder} is not a {role} split.')
    for row in lock['files']:
        if sha256(folder/'images'/row['image']) != row['image_sha256'] or sha256(folder/'labels'/f"{Path(row['image']).stem}.txt") != row['label_sha256']:
            raise ValueError(f"{row['image']} changed since the split was made.")
    return lock['files']


def copy_pair(images: Path, labels: Path, stem: str, suffix: str, output: Path, split: str, target_stem: str | None = None):
    target_stem = target_stem or stem
    shutil.copy2(images/f'{stem}{suffix}', output/'images'/split/f'{target_stem}{suffix}')
    shutil.copy2(labels/f'{stem}.txt', output/'labels'/split/f'{target_stem}.txt')


def build(output: Path, pune_train: Path, pune_val: Path, taco: Path, tiles: Path,
          taco_originals: int = 300, taco_tiles: int = 300, pune_repeat: int = 2, seed: int = 42) -> dict:
    if output.exists():
        raise ValueError(f'{output} exists; build a new version instead of replacing it.')
    rng = random.Random(seed)
    train_rows, val_rows = checked_split(pune_train, 'pune_training'), checked_split(pune_val, 'pune_validation')
    if {r['location_group'] for r in train_rows} & {r['location_group'] for r in val_rows}:
        raise ValueError('A location group appears in both Pune training and validation.')
    for split in ('train', 'val', 'test'):
        (output/'images'/split).mkdir(parents=True); (output/'labels'/split).mkdir(parents=True)
    records = []
    for k in range(pune_repeat):
        for row in train_rows:
            stem = Path(row['image']).stem
            copy_pair(pune_train/'images', pune_train/'labels', stem, '.jpg', output, 'train', f'pune{k}_{stem}')
            records.append({'name': f'pune{k}_{row["image"]}', 'split': 'train', 'kind': 'pune', 'groups': [f"pune:{row['location_group']}"]})
    # litter-v2 lists the v1 TACO photos by file name (kind 'original'); their bytes are copied from v1 itself.
    tile_records = json.loads((tiles/'splits.json').read_text())['images']
    by_split = lambda split: sorted((r for r in tile_records if r['split'] == split and r['kind'] == 'original'), key=lambda r: r['name'])
    for row in rng.sample(by_split('train'), min(taco_originals, len(by_split('train')))) + by_split('test'):
        name = Path(row['name'])
        copy_pair(taco/'images'/row['split'], taco/'labels'/row['split'], name.stem, name.suffix, output, row['split'])
        records.append({'name': row['name'], 'split': row['split'], 'kind': 'original', 'groups': row.get('groups', [])})
    derived = sorted((r for r in tile_records if r['split'] == 'train' and r['kind'] == 'derived'), key=lambda r: r['name'])
    for row in rng.sample(derived, min(taco_tiles, len(derived))):
        name = Path(row['name'])
        copy_pair(tiles/'images/train', tiles/'labels/train', name.stem, name.suffix, output, 'train')
        records.append({'name': row['name'], 'split': 'train', 'kind': 'derived', 'groups': row['groups']})
    for row in val_rows:
        copy_pair(pune_val/'images', pune_val/'labels', Path(row['image']).stem, '.jpg', output, 'val')
        records.append({'name': row['image'], 'split': 'val', 'kind': 'pune', 'groups': [f"pune:{row['location_group']}"]})
    (output/'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: litter\n')
    (output/'splits.json').write_text(json.dumps({'images': records}, indent=1))
    counts = {split: {kind: sum(r['split'] == split and r['kind'] == kind for r in records) for kind in ('pune', 'original', 'derived')} for split in ('train', 'val', 'test')}
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'pune_repeat': pune_repeat,
              'sources': {'pune_train': str(pune_train), 'pune_val': str(pune_val), 'taco': str(taco), 'tiles': str(tiles)},
              'pune_train_lock_sha256': sha256(pune_train/'evaluation-lock.json'), 'pune_val_lock_sha256': sha256(pune_val/'evaluation-lock.json'),
              'counts': counts, 'pune_share_of_train': round(counts['train']['pune'] / sum(counts['train'].values()), 3)}
    (output/'build-report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT/'data/pune-focus-v1')
    parser.add_argument('--pune-train', type=Path, default=ROOT/'data/pune-train-v2')
    parser.add_argument('--pune-val', type=Path, default=ROOT/'data/pune-val-v2')
    parser.add_argument('--taco', type=Path, default=ROOT/'data/litter')
    parser.add_argument('--tiles', type=Path, default=ROOT/'data/litter-v2')
    parser.add_argument('--taco-originals', type=int, default=300)
    parser.add_argument('--taco-tiles', type=int, default=300)
    parser.add_argument('--pune-repeat', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.pune_train, args.pune_val, args.taco, args.tiles,
                           args.taco_originals, args.taco_tiles, args.pune_repeat), indent=2))
