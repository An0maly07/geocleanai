"""Convert pLitterStreet (COCO boxes) into a one-class, group-separated YOLO dataset.

pLitterStreet: https://doi.org/10.5281/zenodo.8288500, CC BY 4.0. Credit the authors
(see docs/DATA_CREDITS.md) wherever the images or a model trained on them is shown.

All litter categories merge into class 0 `litter`. Categories named in --not-litter
(default: any name containing "bin") are never written as litter boxes; their images
stay in the dataset, so a bin-only photo becomes a clean negative. Run --list first
to see every category and its box count before converting.

Images are grouped by their folder inside --images (one folder per city or route),
or by --group-regex on the file path, so no street lands in both train and val.
The split and copying are done by prepare_dataset.prepare.
"""
import argparse
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

try:
    from training.prepare_dataset import prepare
except ImportError:  # run as a script from training/ or in Colab
    from prepare_dataset import prepare

LICENSE = {'id': 1, 'name': 'CC BY 4.0 (pLitterStreet)', 'url': 'https://creativecommons.org/licenses/by/4.0/'}


def load(paths: list[Path]) -> list[dict]:
    return [json.loads(path.read_text(encoding='utf-8')) for path in paths]


def category_counts(cocos: list[dict]) -> Counter:
    counts = Counter()
    for coco in cocos:
        names = {row['id']: row['name'] for row in coco.get('categories', [])}
        counts.update(names.get(ann['category_id'], f'unknown-{ann["category_id"]}') for ann in coco['annotations'])
    return counts


def is_not_litter(name: str, not_litter: list[str]) -> bool:
    lowered = name.lower()
    return any(term.lower() in lowered for term in not_litter)


def group_for(file_name: str, pattern: re.Pattern | None) -> str:
    if pattern:
        match = pattern.search(file_name)
        if not match:
            raise ValueError(f'--group-regex does not match {file_name}')
        return match.group(1) if match.groups() else match.group(0)
    parent = Path(file_name).parent.as_posix()
    if parent in ('', '.'):
        raise ValueError(f'{file_name} is not inside a folder. Pass --group-regex to say how photos are grouped.')
    return parent


def merge(cocos: list[dict], not_litter: list[str], group_regex: str | None = None) -> tuple[dict, dict, Counter]:
    """Merge COCO files into one litter-only COCO with new unique ids, plus groups.json content."""
    pattern = re.compile(group_regex) if group_regex else None
    images, annotations, groups, seen = [], [], {}, {}
    kept = Counter()
    for coco in cocos:
        names = {row['id']: row['name'] for row in coco.get('categories', [])}
        new_ids = {}
        for item in coco['images']:
            file_name = item['file_name']
            if file_name in seen:  # the same photo listed in two annotation files
                new_ids[item['id']] = seen[file_name]
                continue
            new_id = len(images) + 1
            new_ids[item['id']] = seen[file_name] = new_id
            images.append({'id': new_id, 'file_name': file_name, 'width': item['width'], 'height': item['height'], 'license': LICENSE['id']})
            groups[str(new_id)] = group_for(file_name, pattern)
        for ann in coco['annotations']:
            name = names.get(ann['category_id'], f'unknown-{ann["category_id"]}')
            if is_not_litter(name, not_litter):
                kept['not-litter (skipped)'] += 1
                continue
            kept['litter'] += 1
            annotations.append({'id': len(annotations) + 1, 'image_id': new_ids[ann['image_id']], 'bbox': ann['bbox'], 'category_id': 0})
    merged = {'images': images, 'annotations': annotations, 'licenses': [LICENSE],
              'categories': [{'id': 0, 'name': 'litter'}]}
    return merged, groups, kept


def prepare_from_files(annotation_files: list[Path], images: Path, output: Path, not_litter: list[str],
                       group_regex: str | None = None, seed: int = 42, cocos: list[dict] | None = None) -> dict:
    merged, groups, kept = merge(cocos or load(annotation_files), not_litter, group_regex)
    print(dict(kept), '| images:', len(merged['images']), '| groups:', len(set(groups.values())))
    with tempfile.TemporaryDirectory() as scratch:
        annotations, group_file = Path(scratch)/'plitter-litter.json', Path(scratch)/'groups.json'
        annotations.write_text(json.dumps(merged), encoding='utf-8')
        group_file.write_text(json.dumps(groups), encoding='utf-8')
        counts = prepare(annotations, images, group_file, output, seed)
    (output/'groups.json').write_text(json.dumps(groups, indent=1), encoding='utf-8')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotations', type=Path, nargs='+', required=True, help='One or more pLitterStreet COCO JSON files')
    parser.add_argument('--images', type=Path, required=True, help='Folder that file_name paths are relative to')
    parser.add_argument('--output', type=Path, help='Empty folder for the YOLO dataset')
    parser.add_argument('--not-litter', nargs='*', default=['bin'], help='Category name parts that are not litter (default: bin)')
    parser.add_argument('--group-regex', help='Regex on file_name; group 1 (or the whole match) is the location/sequence group')
    parser.add_argument('--list', action='store_true', help='Only print categories and box counts')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cocos = load(args.annotations)
    for name, count in category_counts(cocos).most_common():
        print(f'{count:7d}  {name}  ->  {"not litter" if is_not_litter(name, args.not_litter) else "litter"}')
    if args.list:
        return
    if not args.output:
        parser.error('--output is required unless --list is given')
    print(prepare_from_files(args.annotations, args.images, args.output, args.not_litter, args.group_regex, args.seed, cocos))


if __name__ == '__main__':
    main()
