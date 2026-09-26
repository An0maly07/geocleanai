import json
import random

import numpy as np
import pytest
from PIL import Image

from training.build_derivatives import (MIN_PHOTO_PX, TILE, Photo, build, crop_to, make_tile, target_pixels,
                                        window_labels, yolo_lines)


def test_target_sizes_follow_camera_geometry():
    rng = random.Random(0)
    sizes = [target_pixels(rng, 6.4) for _ in range(2000)]
    # 5 cm at 20 m is under 1 px; 30 cm at 2 m is about 55 px at 6.4 px/degree.
    assert .9 < min(sizes) and max(sizes) < 55.5
    assert np.median([target_pixels(rng, 10.24) for _ in range(2000)]) > np.median(sizes)


def test_window_keeps_mostly_visible_boxes_and_greys_cut_ones():
    boxes = [[10, 10, 50, 50],        # fully inside
             [600, 100, 700, 140],    # 40% visible -> greyed
             [620, 200, 650, 240],    # 67% visible -> kept, clipped
             [300, 300, 301.5, 330],  # 1.5px wide -> dropped
             [900, 900, 950, 950]]    # outside
    kept, grey, dropped = window_labels(boxes, 0, 0)
    assert kept == [[10, 10, 50, 50], [620, 200, 640, 240]]
    assert grey == [[600, 100, 640, 140]] and dropped == 1
    # Non-square windows clip on each axis separately.
    assert window_labels([[10, 90, 30, 130]], 0, 0, 200, 100)[1] == [[10, 90, 30, 100]]


def test_crop_greys_out_cut_litter():
    image = Image.new('RGB', (200, 200), (0, 0, 0))
    image.paste((255, 255, 255), (150, 20, 250, 60))
    crop, kept, greyed, _ = crop_to(image, [[150, 20, 250, 60]], 0, 0, 180, 180)
    assert kept == [] and greyed == 1
    assert crop.getpixel((160, 40)) == (114, 114, 114)


def make_source(tmp_path, splits, size=(2000, 1500)):
    source = tmp_path/'litter'
    rows = []
    for image_id, split in enumerate(splits):
        (source/'images'/split).mkdir(parents=True, exist_ok=True); (source/'labels'/split).mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', size, (40, 90, 40))
        image.paste((250, 250, 250), (900, 700, 1100, 800))
        image.save(source/'images'/split/f'{image_id}.jpg')
        (source/'labels'/split/f'{image_id}.txt').write_text(f'0 {1000/size[0]} {750/size[1]} {200/size[0]} {100/size[1]}\n')
        rows.append({'id': image_id, 'group': f'batch_{image_id}', 'split': split})
    (source/'splits.json').write_text(json.dumps({'images': rows}))
    return source


def test_collage_labels_land_on_the_litter(tmp_path):
    source = make_source(tmp_path, ['train'] * 3)
    photos = [Photo(i, source/f'images/train/{i}.jpg', source/f'labels/train/{i}.txt', tmp_path) for i in range(3)]
    for seed in range(5):
        tile, labels, detail = make_tile(photos[0], photos, random.Random(seed), 6.4)
        assert tile.size == (TILE, TILE) and len(detail['parts']) > 1  # Shrunk photos -> collage.
        assert all(p['scale'] <= 1 and max(p['size']) >= MIN_PHOTO_PX // 2 for p in detail['parts'])
        pixels = np.asarray(tile)
        for x1, y1, x2, y2 in labels:
            centre = pixels[int((y1+y2)/2), int((x1+x2)/2)]
            assert centre.min() > 200  # The white litter patch, not the green ground.
    _, x, y, w, h = map(float, yolo_lines([[100, 50, 140, 90]])[0].split())
    assert (x*TILE, y*TILE, w*TILE, h*TILE) == pytest.approx((120, 70, 40, 40))


def test_build_downscales_keeps_splits_and_uses_only_reviewed_negatives(tmp_path):
    source = make_source(tmp_path, ['train', 'train', 'train', 'val', 'test'])
    review = tmp_path/'review.csv'
    review.write_text('image_id,file_name,split,group,reasons,status,annotation_complete,tags,notes\n'
                      '0,x,train,batch_0,,usable,yes,,\n1,x,train,batch_1,,unusable,no,,labels misaligned\n')
    summary = build(source, tmp_path/'v2', 'r640', 42, 2, review, negative_share=1.0)
    records = json.loads((tmp_path/'v2/splits.json').read_text())['images']
    derivatives = json.loads((tmp_path/'v2/derivatives.json').read_text())
    assert summary['counts']['train'] == {'original': 2, 'derived': 4, 'negative': 1, 'pune': 0, 'pune_empty': 0}
    assert summary['counts']['val'] == {'original': 0, 'derived': 1, 'negative': 0, 'pune': 0, 'pune_empty': 0}
    assert summary['counts']['test'] == {'original': 1, 'derived': 0, 'negative': 0, 'pune': 0, 'pune_empty': 0}
    assert not any(1 in r['parents'] and r['split'] == 'train' for r in records)  # Unusable left training.
    assert all(p['scale'] <= 1 for d in derivatives for p in d.get('parts', []))  # Never enlarged.
    assert all(set(r['parents']) <= {3} for r in records if r['split'] == 'val')  # Val tiles use val photos only.
    assert (tmp_path/'v2/labels/train/0_r640_neg.txt').read_text() == ''
    assert all(Image.open(p).size == (TILE, TILE) for p in (tmp_path/'v2/images/train').glob('*_r640_*.jpg'))


def test_unusable_needs_a_reason(tmp_path):
    source = make_source(tmp_path, ['train', 'val', 'test'])
    review = tmp_path/'review.csv'
    review.write_text('image_id,file_name,split,group,reasons,status,annotation_complete,tags,notes\n0,x,train,b,,unusable,,,\n')
    with pytest.raises(ValueError, match='needs a note'):
        build(source, tmp_path/'v2', 'r640', 42, 1, review)


def test_clean_fill_adds_litter_free_ground_and_lowers_density(tmp_path):
    source = make_source(tmp_path, ['train'] * 4)
    photos = [Photo(i, source/f'images/train/{i}.jpg', source/f'labels/train/{i}.txt', tmp_path) for i in range(4)]
    plain = [len(make_tile(photos[0], photos, random.Random(s), 6.4)[1]) for s in range(8)]
    mixed, clean_parts = [], 0
    for seed in range(8):
        tile, labels, detail = make_tile(photos[0], photos, random.Random(seed), 6.4, clean_pool=photos, clean_share=.8)
        mixed.append(len(labels))
        pixels = np.asarray(tile)
        for part in detail['parts']:
            if part.get('clean'):
                clean_parts += 1
                x, y = part['at']; w, h = part['size']
                assert part['labels'] == 0
                assert pixels[y:y+h, x:x+w].min(axis=2).max() < 200  # No white litter inside a clean patch.
    assert clean_parts > 0
    assert np.mean(mixed) < np.mean(plain)


def test_pune_training_views_are_repeated_and_verified(tmp_path):
    from training.split_pune_labels import sha256
    source = make_source(tmp_path, ['train', 'val', 'test'])
    pune = tmp_path/'pune-train'; (pune/'images').mkdir(parents=True); (pune/'labels').mkdir()
    Image.new('RGB', (640, 640)).save(pune/'images/111_front.jpg'); (pune/'labels/111_front.txt').write_text('0 .5 .5 .1 .1')
    lock = {'role': 'pune_training', 'files': [{'image': '111_front.jpg', 'location_group': 'g1', 'image_sha256': sha256(pune/'images/111_front.jpg')}]}
    (pune/'evaluation-lock.json').write_text(json.dumps(lock))
    summary = build(source, tmp_path/'v2', 'r640', 42, 1, None, pune_train=pune, pune_repeat=3)
    assert summary['counts']['train']['pune'] == 3 and summary['counts']['val']['pune'] == 0
    assert (tmp_path/'v2/labels/train/pune2_111_front.txt').read_text() == '0 .5 .5 .1 .1'
    (pune/'evaluation-lock.json').write_text(json.dumps({**lock, 'role': 'pune_validation'}))
    with pytest.raises(ValueError, match='not a Pune training split'):
        build(source, tmp_path/'v3', 'r640', 42, 1, None, pune_train=pune)


def test_large_pune_views_are_cut_into_labelled_640_tiles(tmp_path):
    from training.split_pune_labels import sha256
    source = make_source(tmp_path, ['train', 'val', 'test'])
    pune = tmp_path/'pune-train-r1024'; (pune/'images').mkdir(parents=True); (pune/'labels').mkdir()
    view = Image.new('RGB', (1024, 1024), (30, 30, 30)); view.paste((255, 255, 255), (500, 500, 520, 520))
    view.save(pune/'images/111_front.jpg')
    (pune/'labels/111_front.txt').write_text(f'0 {510/1024} {510/1024} {20/1024} {20/1024}')
    lock = {'role': 'pune_training', 'files': [{'image': '111_front.jpg', 'location_group': 'g1', 'image_sha256': sha256(pune/'images/111_front.jpg')}]}
    (pune/'evaluation-lock.json').write_text(json.dumps(lock))
    summary = build(source, tmp_path/'v2', 'r1024', 42, 1, None, pune_train=pune, pune_repeat=1)
    assert summary['counts']['train']['pune'] == 4  # 2x2 tiles
    for index in range(4):
        tile = Image.open(tmp_path/f'v2/images/train/pune0_111_front_t{index}.jpg')
        assert tile.size == (640, 640)
        _, x, y, w, h = map(float, (tmp_path/f'v2/labels/train/pune0_111_front_t{index}.txt').read_text().split())
        assert (w*640, h*640) == pytest.approx((20, 20), abs=.01)
        assert min(tile.getpixel((round(x*640), round(y*640)))) > 200  # label sits on the white item


def test_litter_free_pune_tiles_are_used_once_and_capped(tmp_path):
    from training.split_pune_labels import sha256
    source = make_source(tmp_path, ['train', 'val', 'test'])
    pune = tmp_path/'pune-train-r1024'; (pune/'images').mkdir(parents=True); (pune/'labels').mkdir()
    files = []
    for index in range(6):
        name = f'{100 + index}_front.jpg'
        Image.new('RGB', (1024, 1024), (30, 30, 30)).save(pune/'images'/name)
        # One view has litter in its top-left tile only; the other five are clean.
        (pune/'labels'/f'{100 + index}_front.txt').write_text(f'0 {60/1024} {60/1024} {20/1024} {20/1024}' if index == 0 else '')
        files.append({'image': name, 'location_group': 'g1', 'image_sha256': sha256(pune/'images'/name)})
    (pune/'evaluation-lock.json').write_text(json.dumps({'role': 'pune_training', 'files': files}))
    summary = build(source, tmp_path/'v2', 'r1024', 42, 1, None, pune_train=pune, pune_repeat=2, pune_empty_share=.25)
    train = summary['counts']['train']
    assert train['pune'] == 2 and summary['pune_empty_available'] == 23  # 24 tiles, one with litter
    total = sum(train.values())
    assert 0 < train['pune_empty'] <= .25 * total
    names = [p.name for p in (tmp_path/'v2/images/train').glob('pune*_t*.jpg')]
    assert all(n.startswith('pune0_') for n in names if not n.startswith('pune1_100_front_t0'))
    for label in (tmp_path/'v2/labels/train').glob('pune0_10[1-5]_*.txt'):
        assert label.read_text() == ''
