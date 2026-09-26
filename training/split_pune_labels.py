"""Split newly labelled Pune views into training and validation sets by location group.

--all-train (decided 26 September 2026: no train/val split of the labelled set): every
exported view goes to one training set instead, with the same guards against the locked test.

Rule (fixed before labelling; uses label counts only, never model results):
- Whole location groups only: sequences that pass within 100 m of each other never straddle
  the split, so the same street cannot be in both sets.
- Validation should hold about a third of the panoramas AND between 20% and 45% of the litter
  regions (so it can measure recall), with at least one litter view.
- Among qualifying group choices, the one closest to a third of panoramas wins; ties go to
  fewer groups, then alphabetical order, so the result is deterministic.
The locked Pune test set is never read or changed here.
"""
import argparse
import hashlib
import itertools
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REGION_SHARE = (.20, .45)
PANORAMA_SHARE = 1 / 3


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def group_statistics(views: list[dict], groups: dict[str, str], label_dir: Path) -> dict[str, dict]:
    stats = defaultdict(lambda: {'panoramas': set(), 'views': 0, 'litter_views': 0, 'regions': 0})
    for view in views:
        group = groups[view['image_id']]
        regions = sum(1 for line in (label_dir/f"{Path(view['file_name']).stem}.txt").read_text().splitlines() if line.strip())
        stats[group]['panoramas'].add(view['image_id'])
        stats[group]['views'] += 1
        stats[group]['litter_views'] += regions > 0
        stats[group]['regions'] += regions
    return {g: {**s, 'panoramas': len(s['panoramas'])} for g, s in stats.items()}


def choose_validation(stats: dict[str, dict]) -> tuple[list[str], dict]:
    names = sorted(stats)
    total_panoramas = sum(s['panoramas'] for s in stats.values())
    total_regions = sum(s['regions'] for s in stats.values())
    largest = max(names, key=lambda g: (stats[g]['panoramas'], g))
    best, fallback = None, None
    for size in range(1, len(names)):
        for subset in itertools.combinations([n for n in names if n != largest], size):
            panoramas = sum(stats[g]['panoramas'] for g in subset) / total_panoramas
            regions = sum(stats[g]['regions'] for g in subset) / total_regions if total_regions else 0
            litter_views = sum(stats[g]['litter_views'] for g in subset)
            key = (abs(panoramas - PANORAMA_SHARE), size, subset)
            if REGION_SHARE[0] <= regions <= REGION_SHARE[1] and litter_views > 0:
                best = min(best, key) if best else key
            fallback = min(fallback, key) if fallback else key
    chosen = best or fallback
    subset = list(chosen[2])
    summary = {'rule': __doc__.split('Rule')[1].split('The locked')[0].strip(),
               'met_region_constraint': best is not None,
               'validation_panorama_share': round(sum(stats[g]['panoramas'] for g in subset) / total_panoramas, 3),
               'validation_region_share': round(sum(stats[g]['regions'] for g in subset) / total_regions, 3) if total_regions else None}
    return subset, summary


def write_split(export: Path, views: list[dict], output: Path, role: str, parent: dict):
    (output/'images').mkdir(parents=True); (output/'labels').mkdir()
    files = []
    for view in views:
        name = view['file_name']; label = f'{Path(name).stem}.txt'
        shutil.copy2(export/'images'/name, output/'images'/name)
        shutil.copy2(export/'labels'/label, output/'labels'/label)
        files.append({'image': name, 'sequence': view['sequence'], 'location_group': view['location_group'],
                      'image_sha256': sha256(output/'images'/name), 'label_sha256': sha256(output/'labels'/label)})
    (output/'manifest.json').write_text(json.dumps({'views': views}, indent=2))
    lock = {**parent, 'role': role, 'views': len(files), 'files': files, 'created_at': datetime.now(timezone.utc).isoformat()}
    (output/'evaluation-lock.json').write_text(json.dumps(lock, indent=2))


def load_views(export: Path, candidates: Path, test_manifest: Path) -> list[dict]:
    """Exported views with their location group; refuses anything from the locked test set."""
    views = json.loads((export/'manifest.json').read_text())['views']
    groups = {c['id']: c['location_group'] for c in json.loads(candidates.read_text())}
    test_sequences = {v['sequence'] for v in json.loads(test_manifest.read_text())['views']}
    leaked = [v['file_name'] for v in views if v['sequence'] in test_sequences]
    if leaked:
        raise ValueError(f'Views from locked test sequences: {leaked[:5]}')
    test_lock = test_manifest.with_name('evaluation-lock.json')
    if test_lock.is_file():
        test_hashes = {row['image_sha256'] for row in json.loads(test_lock.read_text())['files']}
        leaked = [v['file_name'] for v in views if sha256(export/'images'/v['file_name']) in test_hashes]
        if leaked:
            raise ValueError(f'Views identical to locked test images: {leaked[:5]}')
    for view in views:
        view['location_group'] = groups[view['image_id']]
    return views


def all_training(export: Path, candidates: Path, train_out: Path, test_manifest: Path) -> dict:
    """Every labelled view for training; model selection then relies on the TACO proxy."""
    if train_out.exists():
        raise ValueError(f'{train_out} exists; splits are never overwritten.')
    views = load_views(export, candidates, test_manifest)
    stats = group_statistics(views, {v['image_id']: v['location_group'] for v in views}, export/'labels')
    parent = {'source_export': str(export), 'candidates_sha256': sha256(candidates),
              'split_rule': {'rule': 'No split (decided 26 September 2026): all labelled views train; the locked test set is excluded.'}}
    write_split(export, views, train_out, 'pune_training', parent)
    return {'training': {k: sum(s[k] for s in stats.values()) for k in ('panoramas', 'views', 'litter_views', 'regions')},
            'location_groups': len(stats)}


def split(export: Path, candidates: Path, train_out: Path, val_out: Path, test_manifest: Path) -> dict:
    for folder in (train_out, val_out):
        if folder.exists():
            raise ValueError(f'{folder} exists; splits are never overwritten.')
    views = load_views(export, candidates, test_manifest)
    groups = {v['image_id']: v['location_group'] for v in views}
    stats = group_statistics(views, groups, export/'labels')
    validation, summary = choose_validation(stats)
    parent = {'source_export': str(export), 'candidates_sha256': sha256(candidates), 'split_rule': summary}
    write_split(export, [v for v in views if v['location_group'] in validation], val_out, 'pune_validation', parent)
    write_split(export, [v for v in views if v['location_group'] not in validation], train_out, 'pune_training', parent)
    count = lambda part: {k: sum(stats[g][k] for g in stats if (g in validation) == part) for k in ('panoramas', 'views', 'litter_views', 'regions')}
    return {**summary, 'validation_groups': validation, 'validation': count(True), 'training': count(False)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', type=Path, default=Path('data/pune-label-v2-export'))
    parser.add_argument('--candidates', type=Path, default=Path('artifacts/pune-label-v2/candidates.json'))
    parser.add_argument('--train-output', type=Path, default=Path('data/pune-train-v1'))
    parser.add_argument('--val-output', type=Path, default=Path('data/pune-val-v1'))
    parser.add_argument('--test-manifest', type=Path, default=Path('data/pune-test-v1/manifest.json'))
    parser.add_argument('--all-train', action='store_true', help='No validation split: every labelled view goes to --train-output')
    args = parser.parse_args()
    if args.all_train:
        print(json.dumps(all_training(args.export, args.candidates, args.train_output, args.test_manifest), indent=2))
    else:
        print(json.dumps(split(args.export, args.candidates, args.train_output, args.val_output, args.test_manifest), indent=2))
