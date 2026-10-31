"""Select and render Pune views for fine-tuning the pretrained pLitter model (plan task 2B.1).

Panoramas come from four areas with recent Mapillary 360° photos. One panorama is kept per
~50 m spot (the newest), and any panorama within 300 m of a test panorama is dropped, so no
street in the Ravet or PCMC test sets is trained on. Each kept panorama gives the four
horizontal views at 640 px; "up" (sky) and "down" (camera car) are skipped.

Location groups join panoramas within 100 m of each other, so training/split_pune_labels.py
never puts the same street in both training and validation. No label files are written: a
missing label means the view has not been labelled yet, not that it is clean.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from backend.panorama import ANALYSIS_VERSION, panorama_views
from backend.providers import distance_m

# Area centres, checked 30 Oct 2026 for 2024+ coverage; each is searched on a 3 x 3 grid of
# 500 m-radius Mapillary queries spaced OFFSET degrees apart.
AREAS = {'Hinjewadi': (18.591, 73.739), 'Punawale': (18.630, 73.735),
         'Chikhali': (18.682, 73.810), 'Dehu Road': (18.678, 73.733)}
OFFSET = .008
SPOT_M = 50
EXCLUDE_M = 300
GROUP_M = 100
VIEWS = ('front', 'right', 'back', 'left')
TEST_SETS = (Path('artifacts/ravet-scan/results.json'), Path('artifacts/tiling-test/pcmc_panoramas.json'))


def newest_per_spot(panoramas: list[dict], spot_m: int = SPOT_M) -> list[dict]:
    """Keep the newest panorama in each ~spot_m grid cell."""
    lat_step = spot_m / 111_320
    cells = {}
    for p in panoramas:
        lng_step = lat_step / max(.1, abs(math.cos(math.radians(p['lat']))))
        key = (round(p['lat'] / lat_step), round(p['lng'] / lng_step))
        if key not in cells or p['captured_at'] > cells[key]['captured_at']:
            cells[key] = p
    return sorted(cells.values(), key=lambda p: (p['lat'], p['lng'], p['id']))


def far_from(panoramas: list[dict], points: list[tuple[float, float]], min_m: int = EXCLUDE_M) -> list[dict]:
    """Drop panoramas closer than min_m to any test point; record the nearest distance on the rest."""
    kept = []
    for p in panoramas:
        nearest = min((distance_m(p['lat'], p['lng'], lat, lng) for lat, lng in points), default=None)
        if nearest is None or nearest >= min_m:
            kept.append({**p, 'nearest_test_m': nearest})
    return kept


def location_groups(panoramas: list[dict], link_m: int = GROUP_M) -> dict[str, str]:
    """Single-link groups: panoramas within link_m of each other share a group."""
    parent = {p['id']: p['id'] for p in panoramas}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(panoramas):
        for b in panoramas[i + 1:]:
            if distance_m(a['lat'], a['lng'], b['lat'], b['lng']) < link_m:
                parent[root(a['id'])] = root(b['id'])
    roots = sorted({root(p['id']) for p in panoramas})
    names = {r: f'g{n:03d}' for n, r in enumerate(roots)}
    return {p['id']: names[root(p['id'])] for p in panoramas}


def test_points(paths) -> list[tuple[float, float]]:
    points = []
    for path in paths:
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
        points += [(float(r['lat']), float(r['lng'])) for r in rows]
    if not points:
        raise ValueError('No test panoramas found; refusing to select without the exclusion zone.')
    return points


def collect(areas: dict[str, tuple[float, float]]) -> list[dict]:
    from backend import providers
    found = {}
    for area, (lat, lng) in areas.items():
        for dlat in (-OFFSET, 0, OFFSET):
            for dlng in (-OFFSET, 0, OFFSET):
                for row in providers.nearby(lat + dlat, lng + dlng)['panoramas']:
                    found.setdefault(row['id'], {**row, 'area': area})
    return list(found.values())


def render(panoramas: list[dict], groups: dict[str, str], output: Path) -> list[dict]:
    from backend import providers
    from backend.geometry import perspective
    (output / 'images').mkdir(parents=True, exist_ok=True)
    views = dict(panorama_views())
    records = []
    for p in panoramas:
        item, _ = providers.get_image(p['id'], original=True)
        captured = datetime.fromtimestamp(p['captured_at'] / 1000, timezone.utc).date().isoformat()
        with providers.download_panorama(item) as source:
            size = list(source.size)
            for view_id in VIEWS:
                name = f"{p['id']}_{view_id}.jpg"
                with perspective(source, views[view_id]) as crop:
                    crop.convert('RGB').save(output / 'images' / name, format='JPEG', quality=92)
                records.append({
                    'file_name': name, 'image_id': p['id'], 'view_id': view_id, 'capture_date': captured,
                    'contributor': p['creator'], 'source_url': f"https://www.mapillary.com/app/?pKey={p['id']}",
                    'panorama_size': size, 'sequence': None, 'area': p['area'], 'lat': p['lat'], 'lng': p['lng'],
                    'location_group': groups[p['id']], 'nearest_test_m': p['nearest_test_m'], 'review_status': 'unreviewed',
                })
    return records


def select(output: Path, exclude=TEST_SETS, areas=AREAS, limit: int | None = None) -> dict:
    if any((output / 'reviews').glob('*.json')) or any((output / 'labels').glob('*.txt')):
        raise ValueError('This folder already has labels or reviews. Select into a new folder.')
    points = test_points(exclude)
    found = collect(areas)
    kept = far_from(newest_per_spot(found), points)[:limit]
    groups = location_groups(kept)
    records = render(kept, groups, output)
    rule = {'spot_m': SPOT_M, 'exclude_m': EXCLUDE_M, 'group_m': GROUP_M, 'views': list(VIEWS),
            'areas': areas, 'test_sets': [str(p) for p in exclude], 'test_panoramas': len(points),
            'min_distance_to_test_m': min(p['nearest_test_m'] for p in kept) if kept else None}
    (output / 'manifest.json').write_text(json.dumps({
        'analysis_version': ANALYSIS_VERSION, 'selection': rule, 'views': records,
        'notice': 'Fine-tune candidates. Missing labels are unlabelled, not clean.'}, indent=2), encoding='utf-8')
    # Same shape split_pune_labels.py reads: one row per panorama with its location group.
    (output / 'candidates.json').write_text(json.dumps(
        [{'id': p['id'], 'location_group': groups[p['id']], 'area': p['area'], 'lat': p['lat'], 'lng': p['lng']} for p in kept],
        indent=2), encoding='utf-8')
    by_area = {a: sum(p['area'] == a for p in kept) for a in areas}
    return {'found': len(found), 'panoramas': len(kept), 'views': len(records), 'groups': len(set(groups.values())),
            'by_area': by_area, 'min_distance_to_test_m': rule['min_distance_to_test_m']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('data/pune-finetune-v1'))
    parser.add_argument('--exclude', type=Path, nargs='+', default=list(TEST_SETS),
                        help='JSON lists of test panoramas (with lat/lng) to keep at least 300 m away from')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    print(select(args.output, args.exclude, limit=args.limit))
