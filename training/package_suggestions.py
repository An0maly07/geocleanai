"""Package a labelling batch for AI box suggestions in Colab. No model runs here.

Each view is re-rendered at 1280 px from its original panorama (the labelling page's detail
size), so a zero-shot detector sees small litter with as many pixels as the source holds.
Human labels go along only so the notebook can score the suggestions before anyone relies on
them. Suggestions are help for labellers, never labels: a person accepts or rejects every box.
"""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIZE = 1280


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def render(job: tuple[str, list[tuple[str, str]]]) -> list[tuple[str, bytes]]:
    from backend.geometry import perspective
    from backend.panorama import panorama_views
    source, wanted = job
    views = dict(panorama_views())
    rendered = []
    with Image.open(source) as panorama:
        for name, view_id in wanted:
            buffer = io.BytesIO()
            perspective(panorama, views[view_id], longest_side=SIZE).save(buffer, format='JPEG', quality=92)
            rendered.append((name, buffer.getvalue()))
    return rendered


def package(batch: Path, output: Path, workers: int = 4) -> dict:
    if output.exists():
        raise ValueError(f'{output.name} already exists; package a new version instead of replacing it.')
    manifest = json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
    jobs: dict[str, list[tuple[str, str]]] = {}
    views, labels = [], {}
    for view in manifest['views']:
        name = view['file_name']
        source = Path(view['source_file'])
        if not view.get('original_resolution') or not source.is_file():
            raise ValueError(f'{name} needs its original panorama ({source}).')
        jobs.setdefault(str(source), []).append((name, view['view_id']))
        views.append({'file_name': name, 'image_id': view['image_id'], 'view_id': view['view_id'],
                      'sha256': sha256(batch/'images'/name)})
        review_path = batch/'reviews'/f'{name}.json'
        if review_path.is_file():
            review = json.loads(review_path.read_text(encoding='utf-8'))
            if review['status'] in ('litter', 'clean'):
                labels[name] = {'status': review['status'], 'boxes': review['boxes']}
    output.parent.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as archive:
            for rendered in (pool.map if pool else map)(render, jobs.items()):
                for name, data in rendered:
                    archive.writestr(f'views/{name}', data)
            archive.writestr('suggest-manifest.json', json.dumps({
                'batch': batch.name, 'created_at': datetime.now(timezone.utc).isoformat(), 'view_size': SIZE,
                'views': views, 'labels': labels,
                'notice': 'labels are for scoring suggestions only; suggestions never become labels without a person.'}, indent=1))
    finally:
        if pool:
            pool.shutdown()
    digest = sha256(output)
    output.with_name(output.name + '.sha256').write_text(digest + '\n')
    return {'archive': str(output), 'views': len(views), 'labelled_for_scoring': len(labels),
            'megabytes': round(output.stat().st_size / 1e6, 1), 'sha256': digest}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=ROOT/'data/pune-label-v2')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/colab/geocleanai-suggest-v1.zip')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(package(args.batch, args.output, args.workers), indent=2))
