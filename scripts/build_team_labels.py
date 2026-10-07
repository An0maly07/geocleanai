"""Publish the team's hand-drawn Pune labels for the viewer's "Team labels" mode.

These are human labels, not model output. Only panoramas whose six views were all reviewed
are included, so a missing marker never stands in for an unreviewed direction. Boxes seen in
two overlapping views are merged with the same rule the live analysis uses, and their panorama
positions come from the same geometry, so markers sit where a model's would.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.panorama import ANALYSIS_VERSION, merge_detections, panorama_views


def build(batch: Path, candidates: Path, output: Path) -> dict:
    manifest = json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
    places = {str(c['id']): c for c in json.loads(candidates.read_text(encoding='utf-8'))}
    views = dict(panorama_views())
    by_panorama: dict[str, dict[str, dict]] = {}
    for view in manifest['views']:
        review_path = batch/'reviews'/f"{view['file_name']}.json"
        if review_path.is_file():
            by_panorama.setdefault(view['image_id'], {})[view['view_id']] = {**view, 'review': json.loads(review_path.read_text(encoding='utf-8'))}
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    index, skipped = {}, []
    for image_id, reviewed in sorted(by_panorama.items()):
        if set(reviewed) != set(views):
            skipped.append(image_id)
            continue
        boxes = [{'view_id': view_id, 'label': 'litter', 'confidence': 1.0, 'box': box}
                 for view_id, item in reviewed.items() if item['review']['status'] == 'litter'
                 for box in item['review']['boxes']]
        detections = merge_detections(boxes, views)
        evidence = []
        for view_id in sorted({d['view_id'] for d in detections}):
            name = reviewed[view_id]['file_name']
            shutil.copyfile(batch/'images'/name, output/name)
            evidence.append({'id': view_id, 'view': views[view_id].model_dump(), 'preview': f'/team-labels/{name}'})
        place = places[image_id]
        index[image_id] = {
            'panorama': {'id': image_id, 'lat': place['lat'], 'lng': place['lng'], 'captured_at': place['captured_at'],
                         'creator': place['creator'], 'is_pano': True},
            'analysis_version': ANALYSIS_VERSION, 'detections': detections, 'views': evidence,
            'reviewers': sorted({item['review']['reviewer'].strip() for item in reviewed.values()}),
            'unclear_views': sorted(v for v, item in reviewed.items() if item['review']['status'] == 'skip'),
        }
    (output/'index.json').write_text(json.dumps(index, indent=1), encoding='utf-8')
    return {'panoramas': len(index), 'with_litter': sum(bool(p['detections']) for p in index.values()),
            'markers': sum(len(p['detections']) for p in index.values()), 'partly_reviewed_skipped': len(skipped)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=Path('data/pune-label-v2'))
    parser.add_argument('--candidates', type=Path, default=Path('artifacts/pune-label-v2/candidates.json'))
    parser.add_argument('--output', type=Path, default=Path('public/team-labels'))
    args = parser.parse_args()
    print(json.dumps(build(args.batch, args.candidates, args.output), indent=2))
