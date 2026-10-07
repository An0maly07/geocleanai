"""Split a labelling batch into self-contained packs for a team, and merge the results back.

make:  one pack per member. Each pack holds the review page, that member's panoramas (whole
       panoramas, with originals for the high-detail view), a shared calibration set that every
       member labels, and that member's share of the TACO review sheet.
merge: copies each member's saved reviews back into the batch. A review is accepted only if
       its image hash matches the batch image and the view was assigned to that member; an
       existing different review is never overwritten (reported as a conflict). Calibration
       views are not merged automatically: an agreement report shows where members differ, the
       team agrees on the answer, and one person records it in the main batch.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ['training/review_server.py', 'training/review.html', 'backend/__init__.py', 'backend/geometry.py',
        'backend/schemas.py', 'backend/panorama.py']
REQUIREMENTS = 'fastapi==0.141.1\nuvicorn==0.53.0\nPillow==12.3.0\nnumpy==2.5.3\npydantic==2.13.5\n'
WINDOWS_START = '''@echo off
cd /d "%~dp0"
if not exist .venv (py -3 -m venv .venv || python -m venv .venv)
.venv\\Scripts\\python -m pip install -q -r requirements.txt
echo Opening http://127.0.0.1:8765 - refresh the page if it is blank at first.
start "" http://127.0.0.1:8765
.venv\\Scripts\\python -m training.review_server --data data
'''
UNIX_START = '''#!/bin/sh
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q -r requirements.txt
echo "Open http://127.0.0.1:8765 in your browser (refresh if blank at first)."
.venv/bin/python -m training.review_server --data data
'''
README = '''GeoCleanAI labelling pack for {member}
=====================================

Needs Python 3.11 or newer (python.org). Windows: double-click START-LABELLING.bat.
Mac/Linux: run  sh start-labelling.sh . Then open http://127.0.0.1:8765 .

1. PUNE VIEWS ({views} views). {calibration_note}
   - Enter your name. Use the original-detail view to inspect; label litter you can see there,
     even if tiny.
   - One tight box per distinguishable item (wrappers, bottles, cups, cans, cigarette litter,
     bags on the ground, paper). One box for a touching clump of tiny pieces that cannot be
     told apart. Never group separate items across clean ground.
   - NOT litter: leaves, stones, shadows, road markings, bins themselves, bags people carry,
     construction material, flags, people.
   - "I checked: no visible litter" only after scanning the WHOLE view.
   - "Skip" with a short note if you genuinely cannot tell (blur, darkness).
   Everything saves automatically to data/reviews.

2. TACO SHEET (taco/review.csv, {taco} rows). Open each taco/queue/<image_id>.jpg (magenta boxes)
   and fill four columns:
   status: usable | difficult | unusable   (unusable = wrong/misaligned labels, corrupt, not litter; needs a note)
   annotation_complete: yes | no   (yes only if EVERY visible litter item has a box)
   tags: any of  small clutter shadows leaves road_markings low_light glare occluded cluster
   notes: free text
   Blur or difficulty alone is NOT "unusable".

3. SEND BACK: zip the "data/reviews" folder and "taco/review.csv" (or the whole pack folder)
   and send it to the project owner. Do not edit or rename the images.
'''


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def parse_ranges(text: str | None, members: list[str]) -> dict[str, tuple[int, int]] | None:
    """'1-240,241-480,...' (1-based, inclusive) -> {member: (start, end)} in member order."""
    if not text:
        return None
    parts = [tuple(int(v) for v in part.split('-')) for part in text.split(',')]
    if len(parts) != len(members) or any(len(p) != 2 or p[0] < 1 or p[1] < p[0] for p in parts):
        raise ValueError(f'Give one start-end range per member: {text}')
    return dict(zip(members, parts))


def make(batch: Path, members: list[str], output: Path, taco_queue: Path | None, calibration_count: int = 2, seed: int = 42, root: Path = ROOT,
         photo_ranges: dict | None = None, taco_ranges: dict | None = None, calibration_photos: tuple[int, int] | None = None) -> dict:
    """photo_ranges / calibration_photos use the photo numbers shown on the review page (1-based)."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; packs are never rebuilt over earlier ones.')
    if any((batch/'reviews').glob('*.json')):
        raise ValueError('The batch already has reviews; split only an unreviewed batch.')
    manifest = json.loads((batch/'manifest.json').read_text())
    views = manifest['views']
    panoramas = sorted({v['image_id'] for v in views})
    by_panorama = {p: [v for v in views if v['image_id'] == p] for p in panoramas}
    in_photos = lambda start, end: list(dict.fromkeys(v['image_id'] for v in views[start-1:end]))
    rng = random.Random(seed)
    if calibration_photos:
        calibration = in_photos(*calibration_photos)
    else:
        # Calibration: prefer one older (litter-likely) capture and one recent one, when available.
        creators = sorted({v['contributor'] for v in views})
        calibration = []
        for creator in creators:
            pool = sorted(p for p in panoramas if by_panorama[p][0]['contributor'] == creator and p not in calibration)
            if pool and len(calibration) < calibration_count:
                calibration.append(rng.choice(pool))
        while len(calibration) < calibration_count:
            calibration.append(rng.choice(sorted(set(panoramas) - set(calibration))))
    assignment = {m: [] for m in members}
    if photo_ranges:
        for member, (start, end) in photo_ranges.items():
            chosen = in_photos(start, end)
            partial = [p for p in chosen if not all(start <= views.index(v) + 1 <= end for v in by_panorama[p])]
            if partial:
                raise ValueError(f'Range {start}-{end} for {member} splits panoramas {partial}; use whole panoramas.')
            assignment[member] = [p for p in chosen if p not in calibration]
        assigned = [p for ps in assignment.values() for p in ps]
        if len(assigned) != len(set(assigned)) or set(assigned) | set(calibration) != set(panoramas):
            raise ValueError('Photo ranges must cover every panorama exactly once.')
    else:
        rest = [p for p in panoramas if p not in calibration]
        rest.sort(key=lambda p: (by_panorama[p][0].get('sequence') or '', p))
        for index, panorama in enumerate(rest):
            assignment[members[index % len(members)]].append(panorama)
    taco_rows = list(csv.DictReader(taco_queue.open(encoding='utf-8'))) if taco_queue else []
    if taco_ranges:
        taco_share = {m: taco_rows[start-1:end] for m, (start, end) in taco_ranges.items()}
    else:
        taco_share = {m: taco_rows[i::len(members)] for i, m in enumerate(members)}
    output.mkdir(parents=True, exist_ok=True)
    summary = {'batch': str(batch), 'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed,
               'calibration_panoramas': calibration, 'members': {}}
    for member in members:
        pack = output/f'pack-{member}'
        for relative in CODE:
            (pack/relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, pack/relative)
        (pack/'data/images').mkdir(parents=True); (pack/'data/originals').mkdir(); (pack/'data/reviews').mkdir()
        chosen = calibration + assignment[member]
        pack_views = []
        for panorama in chosen:
            original = root/Path(by_panorama[panorama][0]['source_file'].replace('\\', '/'))
            shutil.copy2(original, pack/'data/originals'/original.name)
            for view in by_panorama[panorama]:
                shutil.copy2(batch/'images'/view['file_name'], pack/'data/images'/view['file_name'])
                pack_views.append({**view, 'source_file': f'data/originals/{original.name}'})
        (pack/'data/manifest.json').write_text(json.dumps({**manifest, 'views': pack_views}, indent=2))
        calibration_views = [v['file_name'] for v in pack_views if v['image_id'] in calibration]
        assigned_views = [v['file_name'] for v in pack_views if v['image_id'] not in calibration]
        if taco_rows:
            (pack/'taco/queue').mkdir(parents=True)
            with (pack/'taco/review.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=list(taco_rows[0])); writer.writeheader(); writer.writerows(taco_share[member])
            for row in taco_share[member]:
                shutil.copy2(taco_queue.parent/'queue'/f"{row['image_id']}.jpg", pack/'taco/queue'/f"{row['image_id']}.jpg")
        info = {'member': member, 'calibration': calibration_views, 'assigned': assigned_views,
                'image_sha256': {v['file_name']: sha256(pack/'data/images'/v['file_name']) for v in pack_views},
                'taco_rows': [r['image_id'] for r in taco_share[member]]}
        (pack/'pack.json').write_text(json.dumps(info, indent=2))
        (pack/'requirements.txt').write_text(REQUIREMENTS)
        (pack/'START-LABELLING.bat').write_text(WINDOWS_START.replace('\n', '\r\n'))
        (pack/'start-labelling.sh').write_text(UNIX_START)
        calibration_note = (f'Start with the {len(calibration_views)} CALIBRATION views (names in pack.json -> "calibration"); '
                            'every team member labels these same views so the team can compare. Then label the rest.'
                            if calibration_views else 'Label only the photo numbers your team assigned to you (shown at the top of the page).')
        (pack/'README.txt').write_text(README.format(member=member, views=len(pack_views), calibration_note=calibration_note, taco=len(taco_share[member])))
        archive = output/f'geocleanai-label-pack-{member}.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(pack.rglob('*')):
                if path.is_file():
                    bundle.write(path, Path(f'pack-{member}')/path.relative_to(pack))
        summary['members'][member] = {'panoramas': len(chosen), 'views': len(pack_views), 'assigned_views': len(assigned_views),
                                      'calibration_views': len(calibration_views), 'taco_rows': len(taco_share[member]),
                                      'zip_mb': round(archive.stat().st_size / 1e6, 1)}
    (output/'assignments.json').write_text(json.dumps({**summary, 'assignment': assignment}, indent=2))
    return summary


def _box_agreement(a: list, b: list) -> float | None:
    """F1 of one-to-one box matches at IoU >= 0.5 between two reviewers (None if both have no boxes)."""
    if not a and not b:
        return None
    def iou(x, y):
        inter = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(x[1], y[1]))
        union = (x[2]-x[0])*(x[3]-x[1]) + (y[2]-y[0])*(y[3]-y[1]) - inter
        return inter / union if union > 0 else 0
    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, matched = set(), set(), 0
    for overlap, i, j in pairs:
        if overlap >= .5 and i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); matched += 1
    return 2 * matched / (len(a) + len(b))


def _pack_root(source: Path, work: Path) -> Path:
    if source.suffix == '.zip':
        target = work/source.stem
        with zipfile.ZipFile(source) as bundle:
            for name in bundle.namelist():
                if not (target/name).resolve().is_relative_to(target.resolve()):
                    raise ValueError(f'Unsafe path in {source.name}')
            bundle.extractall(target)
        source = target
    found = list(source.rglob('pack.json'))
    if len(found) != 1:
        raise ValueError(f'{source} must contain exactly one pack.json')
    return found[0].parent


def merge(sources: list[Path], batch: Path, taco_queue: Path | None, report_path: Path) -> dict:
    manifest = json.loads((batch/'manifest.json').read_text())
    batch_hashes = {v['file_name']: sha256(batch/'images'/v['file_name']) for v in manifest['views']}
    (batch/'reviews').mkdir(exist_ok=True)
    report = {'merged': 0, 'unchanged': 0, 'conflicts': [], 'rejected': [], 'calibration': {}, 'taco': {'filled': 0, 'conflicts': []}}
    taco_updates = {}
    work = report_path.parent/'_unpacked'
    for source in sources:
        root = _pack_root(source, work)
        info = json.loads((root/'pack.json').read_text())
        member = info['member']
        for review_file in sorted((root/'data/reviews').glob('*.json')):
            name = review_file.name.removesuffix('.json')
            review = json.loads(review_file.read_text())
            if batch_hashes.get(name) is None or review.get('sha256') != batch_hashes[name]:
                report['rejected'].append({'member': member, 'view': name, 'reason': 'image hash does not match the batch'})
                continue
            if review.get('status') in (None, 'unreviewed'):
                continue
            if name in info['calibration']:
                report['calibration'].setdefault(name, {})[member] = {'status': review['status'], 'boxes': review['boxes'], 'reviewer': review.get('reviewer')}
                continue
            if name not in info['assigned']:
                report['rejected'].append({'member': member, 'view': name, 'reason': 'view was not assigned to this member'})
                continue
            target = batch/'reviews'/review_file.name
            if target.exists():
                existing = json.loads(target.read_text())
                if (existing['status'], existing['boxes']) == (review['status'], review['boxes']):
                    report['unchanged'] += 1
                else:
                    report['conflicts'].append({'member': member, 'view': name, 'existing_reviewer': existing.get('reviewer')})
                continue
            shutil.copy2(review_file, target)
            report['merged'] += 1
        csv_path = root/'taco/review.csv'
        if csv_path.is_file():
            for row in csv.DictReader(csv_path.open(encoding='utf-8')):
                if any(row[k].strip() for k in ('status', 'annotation_complete', 'tags', 'notes')):
                    taco_updates[row['image_id']] = {**row, '_member': member}
    # Calibration agreement: status agreement and pairwise box F1.
    agreement = []
    for view, answers in report['calibration'].items():
        members = sorted(answers)
        statuses = {answers[m]['status'] for m in members}
        pairs = [_box_agreement(answers[a]['boxes'], answers[b]['boxes']) for i, a in enumerate(members) for b in members[i+1:]]
        pairs = [p for p in pairs if p is not None]
        agreement.append({'view': view, 'reviewers': len(members), 'statuses_agree': len(statuses) == 1,
                          'statuses': {m: answers[m]['status'] for m in members},
                          'box_count': {m: len(answers[m]['boxes']) for m in members},
                          'mean_box_f1': round(sum(pairs) / len(pairs), 3) if pairs else None})
    report['calibration_summary'] = {'views': len(agreement), 'status_agreement': sum(a['statuses_agree'] for a in agreement),
                                     'views_to_discuss': [a['view'] for a in agreement if not a['statuses_agree'] or (a['mean_box_f1'] is not None and a['mean_box_f1'] < .7)],
                                     'per_view': agreement}
    if taco_queue and taco_updates:
        rows = list(csv.DictReader(taco_queue.open(encoding='utf-8')))
        fields = list(rows[0])
        for row in rows:
            update = taco_updates.get(row['image_id'])
            if not update:
                continue
            for key in ('status', 'annotation_complete', 'tags', 'notes'):
                new = update[key].strip()
                if new and row[key].strip() and row[key].strip() != new:
                    report['taco']['conflicts'].append({'image_id': row['image_id'], 'field': key, 'member': update['_member']})
                elif new and not row[key].strip():
                    row[key] = new
            report['taco']['filled'] += 1
        backup = taco_queue.with_name(f"{taco_queue.stem}.before-merge-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.csv")
        shutil.copy2(taco_queue, backup)
        with taco_queue.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        report['taco']['backup'] = str(backup)
    shutil.rmtree(work, ignore_errors=True)
    report_path.write_text(json.dumps(report, indent=2))
    return {k: v for k, v in report.items() if k not in ('calibration',)} | {'calibration_summary': {k: v for k, v in report['calibration_summary'].items() if k != 'per_view'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    make_cmd = sub.add_parser('make')
    make_cmd.add_argument('--batch', type=Path, default=Path('data/pune-label-v2'))
    make_cmd.add_argument('--members', nargs='+', required=True, help='Short names, e.g. A B C D')
    make_cmd.add_argument('--output', type=Path, default=Path('artifacts/label-packs-v2'))
    make_cmd.add_argument('--taco-queue', type=Path, default=Path('data/litter-review/review-queue.csv'))
    make_cmd.add_argument('--calibration', type=int, default=2)
    make_cmd.add_argument('--photo-ranges', help='Per member, in order: e.g. 1-240,241-480,481-720,721-960 (photo numbers on the review page)')
    make_cmd.add_argument('--taco-ranges', help='Per member TACO sheet rows, e.g. 1-58,59-116,117-174,175-232')
    make_cmd.add_argument('--calibration-photos', help='Photos every member labels, e.g. 1-6')
    merge_cmd = sub.add_parser('merge')
    merge_cmd.add_argument('packs', nargs='+', type=Path, help='Returned pack folders or zips')
    merge_cmd.add_argument('--batch', type=Path, default=Path('data/pune-label-v2'))
    merge_cmd.add_argument('--taco-queue', type=Path, default=Path('data/litter-review/review-queue.csv'))
    merge_cmd.add_argument('--report', type=Path, default=Path('artifacts/label-packs-v2/merge-report.json'))
    args = parser.parse_args()
    if args.command == 'make':
        calibration_photos = tuple(int(v) for v in args.calibration_photos.split('-')) if args.calibration_photos else None
        print(json.dumps(make(args.batch, args.members, args.output, args.taco_queue if args.taco_queue.is_file() else None, args.calibration,
                              photo_ranges=parse_ranges(args.photo_ranges, args.members), taco_ranges=parse_ranges(args.taco_ranges, args.members),
                              calibration_photos=calibration_photos), indent=2))
    else:
        print(json.dumps(merge(args.packs, args.batch, args.taco_queue if args.taco_queue.is_file() else None, args.report), indent=2))
