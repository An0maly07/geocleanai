"""Local human review: python -m training.review_server [--port 8765]."""
import argparse
import hashlib
import json
import math
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

from backend.geometry import perspective
from backend.panorama import ANALYSIS_VERSION, panorama_views


class Review(BaseModel):
    status: str
    boxes: list[list[float]] = Field(default_factory=list)
    reviewer: str = Field(min_length=1, max_length=100)
    notes: str = Field(default='', max_length=2000)
    revision: int = Field(default=0, ge=0)
    # Where each box came from: 'hand' (drawn) or 'ai' (an accepted suggestion). Empty for older clients.
    box_sources: list[str] = Field(default_factory=list)
    suggestions_shown: int = Field(default=0, ge=0, le=500)


class ReviewStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.views = manifest['views']
        self.version = manifest['analysis_version']
        self.lock = threading.Lock()
        self.inspection_lock = threading.Lock()
        self.inspections = {}
        self.paths = {}
        self.hashes = {}
        self.reviews = self.root / 'reviews'
        self.reviews.mkdir(exist_ok=True)
        for row in self.views:
            name = row['file_name']
            path = (self.root / 'images' / name).resolve()
            if Path(name).name != name or not path.is_relative_to(self.root / 'images') or name in self.paths:
                raise ValueError('Invalid or duplicate image name in manifest')
            self.paths[name] = path
            self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            source = Path(row.get('source_file', ''))
            if source.is_file() and self.version == ANALYSIS_VERSION:
                with Image.open(source) as panorama:
                    detail_size = min(1920, max(640, panorama.width // 3))
                self.inspections[name] = (source, row['view_id'], detail_size)
        self.assist, self.suggestions = None, {}
        if (self.root / 'suggestions.json').is_file():
            self.load_suggestions(json.loads((self.root / 'suggestions.json').read_text(encoding='utf-8')))

    def load_suggestions(self, data):
        """AI box suggestions for labellers. Only boxes for unchanged images, at or above the scored threshold."""
        threshold = float(data['threshold'])
        for name, item in data['views'].items():
            if name in self.hashes and item['sha256'] == self.hashes[name]:
                self.suggestions[name] = [box for box in item['boxes'] if box[4] >= threshold]
        chosen = data.get('evaluation', {}).get('chosen', {})
        self.assist = {'detector': data['detector'], 'threshold': threshold, 'views': len(self.suggestions),
                       'precision': chosen.get('precision'), 'recall': chosen.get('recall')}

    def inspection(self, name):
        if name not in self.inspections:
            raise HTTPException(404, 'Original detail is unavailable')
        source, view_id, size = self.inspections[name]
        stat = source.stat()
        key = hashlib.sha256(f'{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{self.version}:{size}:{view_id}'.encode()).hexdigest()[:24]
        folder = self.root / 'inspection'
        target = folder / f'{key}.jpg'
        with self.inspection_lock:
            if not target.exists():
                folder.mkdir(exist_ok=True)
                with Image.open(source) as panorama:
                    detail = perspective(panorama, dict(panorama_views())[view_id], longest_side=size)
                temporary = target.with_suffix('.tmp')
                detail.save(temporary, format='JPEG', quality=97, subsampling=0)
                temporary.replace(target)
        return target

    def read(self, name):
        if name not in self.paths:
            raise HTTPException(404, 'Unknown image')
        path = self.reviews / f'{name}.json'
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'status': 'unreviewed', 'boxes': [], 'revision': 0, 'reviewer': '', 'notes': ''}
        if result.get('sha256', self.hashes[name]) != self.hashes[name]:
            return {**result, 'status': 'unreviewed', 'boxes': [],
                    'notes': 'Image changed. Review this image again.'}
        return result

    def save(self, name, review: Review):
        if review.status not in {'litter', 'clean', 'skip', 'unreviewed'}:
            raise HTTPException(422, 'Choose litter, clean, skip or unreviewed')
        if not review.reviewer.strip():
            raise HTTPException(422, 'Enter the reviewer name')
        if (review.status == 'litter') != bool(review.boxes):
            raise HTTPException(422, 'Litter requires boxes; other states must have no boxes')
        if review.status == 'skip' and not review.notes.strip():
            raise HTTPException(422, 'Explain why this view is unclear')
        if len(review.boxes) > 500:
            raise HTTPException(422, 'Too many boxes')
        for box in review.boxes:
            if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
                raise HTTPException(422, 'Boxes must stay inside the image')
            if box[0] >= box[2] or box[1] >= box[3]:
                raise HTTPException(422, 'Boxes must have positive width and height')
        if review.box_sources and (len(review.box_sources) != len(review.boxes) or not set(review.box_sources) <= {'hand', 'ai'}):
            raise HTTPException(422, 'Each box needs a source: hand or ai')
        with self.lock:
            previous = self.read(name)
            if review.revision != previous['revision']:
                raise HTTPException(409, 'This review changed in another tab. Reload before saving.')
            record = {**review.model_dump(), 'reviewer': review.reviewer.strip(),
                      'revision': review.revision + 1, 'sha256': self.hashes[name],
                      'reviewed_at': datetime.now(timezone.utc).isoformat()}
            path = self.reviews / f'{name}.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
            temporary.replace(path)
            return record

    def export(self, output: Path):
        if output.exists():
            raise ValueError('Choose a new output folder to protect previous exports')
        selected = []
        for view in self.views:
            review = self.read(view['file_name'])
            if review['status'] in {'litter', 'clean'}:
                if not view.get('original_resolution') or not view.get('sequence'):
                    raise ValueError('Reviewed views need original imagery and capture sequence metadata')
                selected.append((view, review))
        if not selected:
            raise ValueError('No human-reviewed views to export')
        (output / 'images').mkdir(parents=True)
        (output / 'labels').mkdir()
        records = []
        for view, review in selected:
            name = view['file_name']
            shutil.copy2(self.paths[name], output / 'images' / name)
            labels = [f'0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}'
                      for x1, y1, x2, y2 in review['boxes']]
            (output / 'labels' / f'{Path(name).stem}.txt').write_text('\n'.join(labels), encoding='utf-8')
            records.append({**view, 'review': review})
        (output / 'manifest.json').write_text(json.dumps({
            'analysis_version': self.version, 'views': records,
            'notice': 'Reviewed candidate pool. Assign whole capture groups before training or evaluation.'
        }, indent=2), encoding='utf-8')
        return {'exported': len(records), 'output': str(output)}


def create_app(root: Path):
    store = ReviewStore(root)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        # Loopback binding plus Host and custom-header checks prevent remote
        # webpages from sending cross-origin writes to the local reviewer.
        from fastapi.responses import JSONResponse
        if request.url.hostname not in {'localhost', '127.0.0.1', 'testserver'}:
            return JSONResponse({'detail': 'Local access only'}, status_code=403)
        if request.method == 'POST':
            origin = request.headers.get('origin')
            if request.headers.get('x-review-client') != 'geocleanai' or (
                origin and origin != str(request.base_url).rstrip('/')
            ):
                return JSONResponse({'detail': 'Open the local review page to save'}, status_code=403)
            if int(request.headers.get('content-length', '0')) > 100_000:
                return JSONResponse({'detail': 'Review is too large'}, status_code=413)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def page():
        return FileResponse(Path(__file__).with_name('review.html'))

    @app.get('/api/views')
    def views():
        rows = []
        for row in store.views:
            review = store.read(row['file_name'])
            # Suggestions only on photos nobody has reviewed, so saved labels are never second-guessed by the AI.
            suggestions = store.suggestions.get(row['file_name'], []) if review['status'] == 'unreviewed' else []
            rows.append({**row, 'review': review, 'suggestions': suggestions,
                         'inspection_available': row['file_name'] in store.inspections})
        return rows

    @app.get('/api/assist')
    def assist():
        return store.assist or {}

    @app.get('/inspection/{name}')
    def inspection(name: str):
        return FileResponse(store.inspection(name))

    @app.get('/images/{name}')
    def image(name: str):
        if name not in store.paths:
            raise HTTPException(404, 'Unknown image')
        return FileResponse(store.paths[name])

    @app.post('/api/reviews/{name}')
    def save(name: str, review: Review):
        return store.save(name, review)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/pune-review'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--export', type=Path, help='Export reviewed views to a new folder instead of starting the UI')
    args = parser.parse_args()
    if args.export:
        print(ReviewStore(args.data).export(args.export))
    else:
        uvicorn.run(create_app(args.data), host='127.0.0.1', port=args.port)
