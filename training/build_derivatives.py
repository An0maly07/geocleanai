"""Build a street-like TACO dataset (litter-v2) from the prepared v1 split. No model involved.

Why: Pune litter is small and soft because it is far from a 360° camera. TACO photos are
sharp close-ups that cover only a metre or two of ground. Each derived 640x640 tile is made
from TACO photos DOWNSCALED (never upscaled) so their litter has the angular size of
5-30 cm items at 2-20 m, at the detector's pixels-per-degree. That target comes from camera
geometry, not from Pune test labels.

A photo shrunk that far is smaller than a tile, so a tile becomes a collage of several shrunk
photos from the SAME split. Each photo keeps at least MIN_PHOTO_PX pixels on its long side so
it still shows surrounding ground. Collage seams are artificial and are recorded as a known
limitation. Mild blur, noise, lighting change and JPEG compression are then applied; none of
these add detail.

Rules:
- Every tile uses parents from one split only; train originals are also kept as-is.
- A box cut by a crop keeps its label if >= 60% stays visible; otherwise the visible part is
  filled grey (114) so no unlabelled partial litter remains.
- Negative (litter-free) tiles only from photos a reviewer marked annotation_complete=yes,
  at most ~10% of training images, with a 16px margin around every box.
- Photos a reviewer marked 'unusable' leave training only; val and test stay complete.
- Validation becomes the street-like proxy built from val-split photos only.
- Pune training views (or their 640px tiles) with litter are repeated pune_repeat times.
  Litter-free Pune tiles are real street negatives (flags, people, road clutter) but would
  otherwise dominate once all ~960 labelled views train, so they are used once each and capped
  at pune_empty_share of the final training set, picked by seed.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.tiling import tile_grid

TILE = 640
GREY = (114, 114, 114)
# Detector pixels per degree of view: 100° faces rendered at 640 or 1024 pixels.
PROFILES = {'r640': 6.4, 'r1024': 10.24}
OBJECT_METRES = (.05, .30)
DISTANCE_METRES = (2.0, 20.0)
MIN_PHOTO_PX = 128
KEEP_VISIBLE = .6
# Below 3 px a single-pixel boundary error already drops IoU under 0.5, so such labels cannot be scored.
MIN_LABEL_PX = 3
NEGATIVE_MARGIN = 16
MAX_COLLAGE_ATTEMPTS = 60
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def target_pixels(rng: random.Random, px_per_degree: float) -> float:
    """Apparent size of a random litter item at a random street distance."""
    size, distance = log_uniform(rng, *OBJECT_METRES), log_uniform(rng, *DISTANCE_METRES)
    return math.degrees(2 * math.atan(size / (2 * distance))) * px_per_degree


def read_boxes(label: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label.read_text().splitlines():
        if line.strip():
            _, x, y, w, h = map(float, line.split())
            boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return boxes


def window_labels(boxes: list[list[float]], x0: float, y0: float, width: float = TILE, height: float | None = None):
    """Boxes (pixels) -> (kept boxes in window pixels, regions to grey out, tiny boxes dropped)."""
    height = width if height is None else height
    kept, grey, dropped = [], [], 0
    for x1, y1, x2, y2 in boxes:
        ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0+width), min(y2, y0+height)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        visible = (ix2-ix1)*(iy2-iy1) / max((x2-x1)*(y2-y1), 1e-9)
        local = [ix1-x0, iy1-y0, ix2-x0, iy2-y0]
        if visible < KEEP_VISIBLE:
            grey.append(local)
        elif min(local[2]-local[0], local[3]-local[1]) < MIN_LABEL_PX:
            dropped += 1  # Too small to score at IoU 0.5; leave it unlabelled.
        else:
            kept.append(local)
    return kept, grey, dropped


def degrade(image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    """Realistic capture losses. Returns the degraded image and the parameters used."""
    params = {'blur_sigma': round(rng.uniform(0, 1.0), 3), 'noise_sigma': round(rng.uniform(0, 3), 3),
              'brightness': round(rng.uniform(.8, 1.15), 3), 'contrast': round(rng.uniform(.85, 1.1), 3),
              'shadow': rng.random() < .2, 'jpeg_quality': rng.randint(60, 92)}
    if params['blur_sigma'] > .05:
        image = image.filter(ImageFilter.GaussianBlur(params['blur_sigma']))
    array = np.asarray(image, dtype=np.float32)
    mean = array.mean()
    array = (array - mean) * params['contrast'] + mean * params['brightness']
    if params['shadow']:
        # A soft darker band, like a building or tree shadow across the road.
        axis = rng.choice((0, 1)); start = rng.uniform(.1, .7); width = rng.uniform(.15, .4)
        ramp = np.linspace(0, 1, array.shape[axis])
        band = np.clip(1 - .45*np.exp(-((ramp - start - width/2)/(width/2))**2), .55, 1)
        array *= band[:, None, None] if axis == 0 else band[None, :, None]
    noise_rng = np.random.default_rng(rng.getrandbits(32))
    array += noise_rng.normal(0, params['noise_sigma'], array.shape)
    image = Image.fromarray(np.clip(array, 0, 255).round().astype('uint8'))
    buffer = io.BytesIO(); image.save(buffer, format='JPEG', quality=params['jpeg_quality'])
    return Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'), params


def yolo_lines(boxes, size=TILE):
    return [f'0 {(b[0]+b[2])/2/size:.8f} {(b[1]+b[3])/2/size:.8f} {(b[2]-b[0])/size:.8f} {(b[3]-b[1])/size:.8f}' for b in boxes]


def is_clear(boxes, x0, y0, margin=NEGATIVE_MARGIN):
    return all(b[2] < x0-margin or b[0] > x0+TILE+margin or b[3] < y0-margin or b[1] > y0+TILE+margin for b in boxes)


class Photo:
    """One parent photo with native boxes. Small renders come from a 640px cache to save decoding."""

    def __init__(self, image_id: int, path: Path, label: Path, cache: Path):
        self.id, self.path = image_id, path
        with Image.open(path) as image:
            self.size = image.size
            scale = min(1.0, TILE / max(image.size))
            image.convert('RGB').resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.BOX).save(cache/f'{image_id}.png')
        self.cache = cache/f'{image_id}.png'
        self.boxes = read_boxes(label, *self.size)

    def reference(self):
        return float(np.median([max(b[2]-b[0], b[3]-b[1]) for b in self.boxes])) if self.boxes else None

    def street_scale(self, rng: random.Random, px_per_degree: float) -> tuple[float, float]:
        """Scale giving street-distance litter, but never enlarging and never below MIN_PHOTO_PX."""
        target = target_pixels(rng, px_per_degree)
        reference = self.reference()
        scale = min(1.0, target / reference) if reference else rng.uniform(MIN_PHOTO_PX, TILE) / max(self.size)
        return min(1.0, max(scale, MIN_PHOTO_PX / max(self.size))), target

    def render(self, scale: float) -> tuple[Image.Image, list[list[float]]]:
        size = (max(1, round(self.size[0]*scale)), max(1, round(self.size[1]*scale)))
        source = self.cache if max(size) <= TILE else self.path
        with Image.open(source) as image:
            image = image.convert('RGB')
            rendered = image.resize(size, Image.Resampling.BOX) if image.size != size else image.copy()
        return rendered, [[v*scale for v in b] for b in self.boxes]


def crop_to(image, boxes, x0, y0, width, height):
    """Crop a region, returning the crop, kept labels (crop pixels) and grey regions applied."""
    crop = image.crop((x0, y0, x0+width, y0+height))
    kept, grey, dropped = window_labels(boxes, x0, y0, width, height)
    for gx1, gy1, gx2, gy2 in grey:
        crop.paste(GREY, (int(gx1), int(gy1), math.ceil(gx2), math.ceil(gy2)))
    return crop, kept, len(grey), dropped


def centred_origin(extent, size, centre, rng):
    if extent <= size:
        return 0
    return int(min(max(centre - rng.uniform(.2, .8)*size, 0), extent - size))


def clean_patch(photo: Photo, width: int, height: int, rng: random.Random):
    """A litter-free patch from a photo whose annotations were confirmed complete.

    Ground has no single "right" size, so the photo is rendered larger than a street-scale litter
    cell (still never enlarged beyond native) to leave room for windows clear of every box.
    """
    long_side = rng.uniform(2 * MIN_PHOTO_PX, min(max(photo.size), 4 * TILE))
    scale = min(1.0, long_side / max(photo.size))
    image, boxes = photo.render(scale)
    width, height = min(width, image.width), min(height, image.height)
    for _ in range(10):
        x0, y0 = rng.randint(0, image.width - width), rng.randint(0, image.height - height)
        if all(b[2] < x0-NEGATIVE_MARGIN or b[0] > x0+width+NEGATIVE_MARGIN or b[3] < y0-NEGATIVE_MARGIN or b[1] > y0+height+NEGATIVE_MARGIN for b in boxes):
            return image.crop((x0, y0, x0+width, y0+height)), scale
    return None, None


def make_tile(anchor: Photo, pool: list[Photo], rng: random.Random, px_per_degree: float,
              clean_pool: list[Photo] = (), clean_share: float = 0.0):
    """One labelled tile: a single crop when the scaled photo covers the tile, otherwise a collage.

    Collages use free-rectangle packing: each photo is cropped to fit the largest empty area,
    which is then split into the space to its right and below. Areas thinner than
    MIN_PHOTO_PX // 2 stay grey. A photo is used at most once per tile. With clean_share > 0,
    that share of areas after the first is filled with verified litter-free ground instead,
    so a tile holds less litter than a pure collage (street scenes are mostly clean).
    """
    canvas = Image.new('RGB', (TILE, TILE), GREY)
    labels, parts, used = [], [], set()
    free = [(0, 0, TILE, TILE)]
    minimum = MIN_PHOTO_PX // 2
    candidates = [anchor]
    for _ in range(MAX_COLLAGE_ATTEMPTS):
        if not free:
            break
        if candidates:
            photo = candidates.pop(0)
        else:
            unused = [p for p in pool if p.id not in used]
            if not unused:
                break
            photo = rng.choice(unused)
        used.add(photo.id)
        free.sort(key=lambda r: r[2]*r[3], reverse=True)
        x, y, room_w, room_h = free.pop(0)
        if parts and clean_pool and rng.random() < clean_share:
            source = rng.choice(clean_pool)
            patch, patch_scale = clean_patch(source, room_w, room_h, rng)
            if patch is not None and min(patch.size) >= minimum:
                canvas.paste(patch, (x, y))
                parts.append({'parent': source.id, 'clean': True, 'scale': round(patch_scale, 5), 'at': [x, y], 'size': list(patch.size), 'labels': 0})
                for rect in ((x + patch.width, y, room_w - patch.width, patch.height), (x, y + patch.height, room_w, room_h - patch.height)):
                    if rect[2] >= minimum and rect[3] >= minimum:
                        free.append(rect)
                used.discard(photo.id)
                candidates.insert(0, photo)
                continue
        scale, target = photo.street_scale(rng, px_per_degree)
        image, boxes = photo.render(scale)
        width, height = min(image.width, room_w), min(image.height, room_h)
        if width < minimum or height < minimum:
            free.append((x, y, room_w, room_h))
            continue
        # Centre larger photos on one of their litter items so the crop keeps a label.
        focus = rng.choice(boxes) if boxes else [image.width/2, image.height/2, image.width/2, image.height/2]
        x0 = centred_origin(image.width, width, (focus[0]+focus[2])/2, rng)
        y0 = centred_origin(image.height, height, (focus[1]+focus[3])/2, rng)
        crop, kept, greyed, dropped = crop_to(image, boxes, x0, y0, width, height)
        canvas.paste(crop, (x, y))
        labels += [[b[0]+x, b[1]+y, b[2]+x, b[3]+y] for b in kept]
        parts.append({'parent': photo.id, 'scale': round(scale, 5), 'target_px': round(target, 2), 'at': [x, y],
                      'size': [width, height], 'labels': len(kept), 'greyed': greyed, 'dropped_tiny': dropped})
        for rect in ((x + width, y, room_w - width, height), (x, y + height, room_w, room_h - height)):
            if rect[2] >= minimum and rect[3] >= minimum:
                free.append(rect)
    grey_share = float((np.asarray(canvas) == GREY).all(axis=2).mean())
    return canvas, labels, {'parts': parts, 'grey_share': round(grey_share, 3)}


def make_negative(photo: Photo, rng: random.Random):
    """A litter-free tile from a photo whose annotations a reviewer confirmed complete."""
    for _ in range(20):
        long_side = rng.uniform(TILE * 1.05, max(photo.size))
        scale = min(1.0, long_side / max(photo.size))
        image, boxes = photo.render(scale)
        if image.width < TILE or image.height < TILE:
            continue
        x0, y0 = rng.randint(0, image.width-TILE), rng.randint(0, image.height-TILE)
        if is_clear(boxes, x0, y0):
            return image.crop((x0, y0, x0+TILE, y0+TILE)), {'parent': photo.id, 'scale': round(scale, 5), 'window': [x0, y0]}
    return None, None


def read_review(path: Path | None) -> dict[int, dict]:
    if not path or not path.is_file():
        return {}
    rows = {}
    for row in csv.DictReader(path.open(encoding='utf-8')):
        status = row['status'].strip().lower()
        if status not in ('', 'usable', 'difficult', 'unusable'):
            raise ValueError(f"Image {row['image_id']}: status must be usable, difficult or unusable.")
        if status == 'unusable' and not row['notes'].strip():
            raise ValueError(f"Image {row['image_id']}: 'unusable' needs a note explaining why.")
        rows[int(row['image_id'])] = {'status': status, 'complete': row['annotation_complete'].strip().lower() == 'yes'}
    return rows


def build(source: Path, output: Path, profile: str, seed: int, per_image: int, review_path: Path | None,
          negative_share: float = .1, clean_share: float = .5, pune_train: Path | None = None, pune_repeat: int = 3,
          pune_empty_share: float = .15):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output folder to avoid mixing dataset versions.')
    split_data = json.loads((source/'splits.json').read_text())
    parents = {row['id']: row for row in split_data['images']}
    review = read_review(review_path)
    px_per_degree = PROFILES[profile]
    for split in ('train', 'val', 'test'):
        (output/'images'/split).mkdir(parents=True); (output/'labels'/split).mkdir(parents=True)
    records, derivatives = [], []

    def add(name, image, labels, split, kind, parent_ids, detail):
        image.save(output/'images'/split/name, quality=95)
        (output/'labels'/split/f'{Path(name).stem}.txt').write_text('\n'.join(yolo_lines(labels)))
        groups = sorted({parents[i]['group'] for i in parent_ids})
        records.append({'name': name, 'parents': sorted(set(parent_ids)), 'groups': groups, 'split': split, 'kind': kind})
        derivatives.append({'name': name, 'split': split, 'kind': kind, 'profile': profile, 'labels': len(labels), **detail})

    for image_id in sorted(parents):
        split = parents[image_id]['split']
        if split == 'test' or (split == 'train' and review.get(image_id, {}).get('status') != 'unusable'):
            # Originals: unchanged TACO test for comparability; train keeps its close-ups too.
            path = next(p for p in (source/'images'/split).glob(f'{image_id}.*') if p.suffix.lower() in IMAGE_SUFFIXES)
            shutil.copy2(path, output/'images'/split/path.name)
            shutil.copy2(source/'labels'/split/f'{image_id}.txt', output/'labels'/split/f'{image_id}.txt')
            records.append({'name': path.name, 'parents': [image_id], 'groups': [parents[image_id]['group']], 'split': split, 'kind': 'original'})

    def write_pune(name, target, window, parent, kind):
        """window None copies the whole view and its label file; otherwise (x0, y0) of a 640px tile."""
        if window is None:
            shutil.copy2(pune_train/'images'/name, output/'images/train'/target)
            shutil.copy2(pune_train/'labels'/f'{Path(name).stem}.txt', output/'labels/train'/f'{Path(target).stem}.txt')
        else:
            with Image.open(pune_train/'images'/name) as opened:
                view = opened.convert('RGB')
            boxes = read_boxes(pune_train/'labels'/f'{Path(name).stem}.txt', *view.size)
            crop, kept, _, _ = crop_to(view, boxes, *window, min(TILE, view.width), min(TILE, view.height))
            crop.save(output/'images/train'/target, quality=95)
            (output/'labels/train'/f'{Path(target).stem}.txt').write_text('\n'.join(yolo_lines(kept)))
        records.append({'name': target, **parent, 'split': 'train', 'kind': kind})

    pune_empty, pune_ambiguous = [], 0
    if pune_train:
        # Real Pune street views, already at detector scale. Pieces with litter are repeated so they
        # are not swamped by ~3,000 TACO images; they come only from the Pune training set.
        lock = json.loads((pune_train/'evaluation-lock.json').read_text())
        if lock['role'] != 'pune_training':
            raise ValueError(f'{pune_train} is not a Pune training split.')
        for row in lock['files']:
            name = row['image']
            if hashlib.sha256((pune_train/'images'/name).read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError(f'{name} changed since the Pune split was made.')
            with Image.open(pune_train/'images'/name) as opened:
                size = opened.size
            boxes = read_boxes(pune_train/'labels'/f'{Path(name).stem}.txt', *size)
            if max(size) <= TILE:
                pieces = [(name, None, 'litter' if boxes else 'empty')]
            else:
                # Views rendered above 640px (E3) are cut into the same overlapping 640px tiles the
                # app uses, with the usual rule for litter cut by a tile edge.
                pieces = []
                for index, (x0, y0) in enumerate(tile_grid(*size)):
                    kept, grey, dropped = window_labels(boxes, x0, y0, min(TILE, size[0]), min(TILE, size[1]))
                    # A tile with only greyed or too-small litter is neither a clean negative nor a usable positive.
                    kind = 'litter' if kept else ('empty' if not grey and not dropped else 'ambiguous')
                    pieces.append((f'{Path(name).stem}_t{index}.jpg', (x0, y0), kind))
            parent = {'parents': [f"pune:{Path(name).stem.rsplit('_', 1)[0]}"], 'groups': [f"pune:{row['location_group']}"]}
            for piece, window, kind in pieces:
                if kind == 'empty':
                    pune_empty.append((name, piece, window, parent))  # Written after TACO, once the cap is known.
                elif kind == 'ambiguous':
                    pune_ambiguous += 1
                else:
                    for k in range(pune_repeat):
                        write_pune(name, f'pune{k}_{piece}', window, parent, 'pune')

    with tempfile.TemporaryDirectory() as cache_dir:
        for split, count in (('train', per_image), ('val', 1)):
            usable = [i for i in sorted(parents) if parents[i]['split'] == split and not (split == 'train' and review.get(i, {}).get('status') == 'unusable')]
            photos = [Photo(i, next(p for p in (source/'images'/split).glob(f'{i}.*') if p.suffix.lower() in IMAGE_SUFFIXES),
                            source/'labels'/split/f'{i}.txt', Path(cache_dir)) for i in usable]
            clean_pool = [p for p in photos if review.get(p.id, {}).get('complete')]
            for photo in photos:
                for k in range(count):
                    rng = random.Random(f'{seed}-{profile}-{split}-{photo.id}-{k}')
                    tile, labels, detail = make_tile(photo, photos, rng, px_per_degree, clean_pool, clean_share)
                    tile, params = degrade(tile, rng)
                    add(f'{photo.id}_{profile}_{k}.jpg', tile, labels, split, 'derived', [p['parent'] for p in detail['parts']], {**detail, 'degradation': params})
            # Clean tiles only where a reviewer confirmed every litter item is labelled.
            allowed = int(negative_share * len(usable)) if split == 'train' else len(usable)
            made = 0
            for photo in photos:
                if made >= allowed or not review.get(photo.id, {}).get('complete'):
                    continue
                rng = random.Random(f'{seed}-{profile}-{split}-{photo.id}-negative')
                tile, detail = make_negative(photo, rng)
                if tile is not None:
                    tile, params = degrade(tile, rng)
                    add(f'{photo.id}_{profile}_neg.jpg', tile, [], split, 'negative', [photo.id], {**detail, 'degradation': params})
                    made += 1

    if pune_empty:
        # Cap litter-free Pune pieces at pune_empty_share of the final training set.
        others = sum(r['split'] == 'train' for r in records)
        allowed = min(len(pune_empty), int(pune_empty_share / (1 - pune_empty_share) * others))
        for name, piece, window, parent in random.Random(f'{seed}-{profile}-pune-empty').sample(pune_empty, allowed):
            write_pune(name, f'pune0_{piece}', window, parent, 'pune_empty')

    # Leakage guard: a parent photo or capture group must never appear in two splits.
    for key in ('parents', 'groups'):
        seen = {}
        for record in records:
            for value in record[key]:
                if seen.setdefault(value, record['split']) != record['split']:
                    raise AssertionError(f'{key[:-1]} {value} appears in two splits')
    (output/'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: litter\n', encoding='utf-8')
    counts = {split: {kind: sum(r['split'] == split and r['kind'] == kind for r in records) for kind in ('original', 'derived', 'negative', 'pune', 'pune_empty')} for split in ('train', 'val', 'test')}
    tiles = [d for d in derivatives if d['kind'] == 'derived']
    summary = {'profile': profile, 'px_per_degree': px_per_degree, 'seed': seed, 'per_image': per_image,
               'source_split_sha256': hashlib.sha256((source/'splits.json').read_bytes()).hexdigest(),
               'review_file': str(review_path) if review_path and review_path.is_file() else None, 'reviewed_images': len(review),
               'unusable_removed_from_train': sum(1 for i, r in review.items() if r['status'] == 'unusable' and parents.get(i, {}).get('split') == 'train'),
               'counts': counts, 'pune_train': str(pune_train) if pune_train else None, 'pune_repeat': pune_repeat if pune_train else 0,
               'pune_empty_share': pune_empty_share if pune_train else 0, 'pune_empty_available': len(pune_empty),
               'pune_ambiguous_tiles_skipped': pune_ambiguous,
               'mean_photos_per_tile': round(float(np.mean([len(d['parts']) for d in tiles])), 2) if tiles else None,
               'mean_grey_share': round(float(np.mean([d['grey_share'] for d in tiles])), 3) if tiles else None,
               'mean_labels_per_tile': round(float(np.mean([d['labels'] for d in tiles])), 1) if tiles else None,
               'mean_clean_share_of_parts': round(float(np.mean([sum(p.get('clean', False) for p in d['parts']) / len(d['parts']) for d in tiles])), 3) if tiles else None,
               'tiles_without_labels': sum(d['labels'] == 0 for d in tiles),
               'rules': {'object_metres': OBJECT_METRES, 'distance_metres': DISTANCE_METRES, 'min_photo_px': MIN_PHOTO_PX,
                         'keep_visible': KEEP_VISIBLE, 'min_label_px': MIN_LABEL_PX, 'negative_margin_px': NEGATIVE_MARGIN,
                         'negative_share': negative_share, 'clean_share': clean_share},
               'limitations': ['Collage seams between photos are artificial.',
                               'TACO photos show litter from close range; shrinking them does not recreate street perspective or 360° stitching artefacts.',
                               'Target sizes come from assumed item sizes (5-30 cm) and distances (2-20 m), not from Pune data.',
                               'Every TACO photo contains litter, so collage tiles hold far more litter per image than a street view; this may push the model towards false alarms.']}
    (output/'splits.json').write_text(json.dumps({**summary, 'images': records}, indent=2), encoding='utf-8')
    (output/'derivatives.json').write_text(json.dumps(derivatives, indent=2), encoding='utf-8')
    if (source/'credits.csv').is_file():
        shutil.copy2(source/'credits.csv', output/'credits.csv')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=Path('data/litter'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=sorted(PROFILES), default='r640')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per-image', type=int, default=2, help='Derived tiles per training photo')
    parser.add_argument('--review', type=Path, default=Path('data/litter-review/review-queue.csv'))
    parser.add_argument('--clean-share', type=float, default=.5, help='Share of collage areas filled with verified clean ground')
    parser.add_argument('--pune-train', type=Path, help='Pune training set from training/split_pune_labels.py (--all-train)')
    parser.add_argument('--pune-repeat', type=int, default=3)
    parser.add_argument('--pune-empty-share', type=float, default=.15, help='Cap on litter-free Pune pieces as a share of the training set')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.profile, args.seed, args.per_image, args.review,
                           clean_share=args.clean_share, pune_train=args.pune_train, pune_repeat=args.pune_repeat,
                           pune_empty_share=args.pune_empty_share), indent=2))
