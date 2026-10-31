"""Team check of Claude's fine-tune labels (plan task 2B.3).

build: copy a sample of labelled views into a new review folder. Claude's boxes go into
suggestions.json, so the existing review page (training/review_server.py) shows them as dashed
boxes the reviewer accepts, rejects or replaces, and views Claude marked clean show no boxes.
The sample is half views Claude marked as litter and half views marked clean, so both wrong
boxes and missed litter can show up.

compare: after the review, count how many checked views needed a fix. A view needs a fix when
the reviewer's litter/clean decision differs from Claude's, a Claude box was rejected, or the
reviewer drew a box of their own. If more than 1 in 5 checked views needed a fix, the labels
fail the check and Claude re-labels.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

FAIL_RATE = .2


def sample(views: list[dict], labels: dict, fraction: float = .1, seed: int = 0) -> list[dict]:
    """fraction of all views, half from views Claude labelled as litter and half from clean views."""
    rng = random.Random(seed)
    litter = [v for v in views if labels[v['file_name'][:-4]]]
    clean = [v for v in views if not labels[v['file_name'][:-4]]]
    half = round(len(views) * fraction / 2)
    picked = rng.sample(litter, min(half, len(litter))) + rng.sample(clean, min(half, len(clean)))
    rng.shuffle(picked)
    return picked


def build(source: Path, output: Path, fraction: float = .1, seed: int = 0) -> dict:
    if output.exists():
        raise ValueError('Choose a new folder; an earlier check may hold reviews.')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    labels = json.loads((source / 'claude_labels.json').read_text(encoding='utf-8'))
    picked = sample(manifest['views'], labels, fraction, seed)
    (output / 'images').mkdir(parents=True)
    suggestions = {}
    for view in picked:
        name = view['file_name']
        shutil.copy2(source / 'images' / name, output / 'images' / name)
        digest = hashlib.sha256((output / 'images' / name).read_bytes()).hexdigest()
        suggestions[name] = {'sha256': digest, 'boxes': [[x1, y1, x2, y2, 1.0] for _, x1, y1, x2, y2 in labels[name[:-4]]]}
    (output / 'manifest.json').write_text(json.dumps({
        'analysis_version': manifest['analysis_version'], 'views': picked, 'source': str(source), 'seed': seed,
        'notice': "Team check of Claude's labels. Dashed boxes are Claude's; accept, reject or redraw them."}, indent=2), encoding='utf-8')
    (output / 'suggestions.json').write_text(json.dumps({
        'detector': "Claude's labels (drawn by eye by an AI, not a detector). Check every box and the whole photo",
        'threshold': 0, 'views': suggestions}, indent=2), encoding='utf-8')
    return {'views': len(picked), 'claude_litter_views': sum(bool(s['boxes']) for s in suggestions.values()),
            'claude_boxes': sum(len(s['boxes']) for s in suggestions.values()), 'output': str(output)}


def needs_fix(claude_boxes: list, review: dict) -> list[str]:
    """Reasons this view's Claude labels needed fixing, judged from the saved review."""
    reasons = []
    reviewer_litter = review['status'] == 'litter'
    if reviewer_litter != bool(claude_boxes):
        reasons.append('reviewer marked litter' if reviewer_litter else 'reviewer marked clean')
    sources = review.get('box_sources') or ['hand'] * len(review['boxes'])
    accepted = sources.count('ai')
    if reviewer_litter and accepted < len(claude_boxes):
        reasons.append(f'{len(claude_boxes) - accepted} Claude box(es) rejected')
    if sources.count('hand'):
        reasons.append(f"{sources.count('hand')} box(es) drawn by the reviewer")
    return reasons


def compare(check: Path) -> dict:
    manifest = json.loads((check / 'manifest.json').read_text(encoding='utf-8'))
    suggestions = json.loads((check / 'suggestions.json').read_text(encoding='utf-8'))['views']
    rows, skipped, unreviewed = [], 0, 0
    for view in manifest['views']:
        name = view['file_name']
        path = check / 'reviews' / f'{name}.json'
        review = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'status': 'unreviewed'}
        if review['status'] == 'unreviewed':
            unreviewed += 1
            continue
        if review['status'] == 'skip':
            skipped += 1
            continue
        if review.get('sha256') != suggestions[name]['sha256']:
            raise ValueError(f'{name} changed after it was reviewed')
        reasons = needs_fix(suggestions[name]['boxes'], review)
        rows.append({'view': name, 'reviewer': review['reviewer'], 'fix': reasons})
    checked = len(rows)
    fixed = sum(bool(r['fix']) for r in rows)
    rate = fixed / checked if checked else None
    return {'checked': checked, 'needed_fix': fixed, 'fix_rate': rate, 'skipped': skipped, 'unreviewed': unreviewed,
            'passed': None if rate is None or unreviewed else rate <= FAIL_RATE,
            'rule': f'labels pass when at most {FAIL_RATE:.0%} of checked views need a fix', 'views': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='Make the review folder')
    b.add_argument('--source', type=Path, default=Path('data/pune-finetune-v1'))
    b.add_argument('--output', type=Path, default=Path('data/pune-finetune-v1-check'))
    b.add_argument('--fraction', type=float, default=.1)
    b.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('compare', help='Score the finished review')
    c.add_argument('--check', type=Path, default=Path('data/pune-finetune-v1-check'))
    args = parser.parse_args()
    if args.command == 'build':
        print(build(args.source, args.output, args.fraction, args.seed))
    else:
        result = compare(args.check)
        (args.check / 'check-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print({k: v for k, v in result.items() if k != 'views'})
