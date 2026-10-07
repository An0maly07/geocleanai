import csv
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from training.label_packs import make, merge
from training.review_server import create_app

HEADERS = {'X-Review-Client': 'geocleanai'}


def make_batch(tmp_path):
    batch = tmp_path/'batch'; (batch/'images').mkdir(parents=True)
    (tmp_path/'artifacts').mkdir()
    views = []
    for index, (image_id, creator) in enumerate([('101', 'old'), ('102', 'new'), ('103', 'new'), ('104', 'new'), ('105', 'old')]):
        Image.new('RGB', (256, 128), (index * 40, 90, 90)).save(tmp_path/'artifacts'/f'panorama-{image_id}.jpg')
        for view_id in ('front', 'back'):
            name = f'{image_id}_{view_id}.jpg'
            Image.new('RGB', (64, 64), (index * 40, 10, 10)).save(batch/'images'/name)
            views.append({'file_name': name, 'image_id': image_id, 'view_id': view_id, 'contributor': creator, 'sequence': f's{image_id}',
                          'capture_date': '2026-01-01', 'source_url': 'x', 'panorama_size': [256, 128],
                          'source_file': f'artifacts\\panorama-{image_id}.jpg', 'original_resolution': True, 'review_status': 'unreviewed'})
    (batch/'manifest.json').write_text(json.dumps({'analysis_version': 'cube-100-v1', 'views': views}))
    taco = tmp_path/'review'; (taco/'queue').mkdir(parents=True)
    with (taco/'review-queue.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['image_id', 'file_name', 'split', 'group', 'reasons', 'status', 'annotation_complete', 'tags', 'notes'])
        writer.writeheader()
        for image_id in range(4):
            writer.writerow({'image_id': image_id, 'file_name': 'x', 'split': 'train', 'group': 'g', 'reasons': 'r', 'status': '', 'annotation_complete': '', 'tags': '', 'notes': ''})
            Image.new('RGB', (8, 8)).save(taco/'queue'/f'{image_id}.jpg')
    return batch, taco/'review-queue.csv'


def label(pack, monkeypatch, decisions):
    monkeypatch.chdir(pack)  # The start script runs the page from the pack folder.
    client = TestClient(create_app(pack/'data'))
    views = {v['file_name']: v for v in client.get('/api/views').json()}
    assert all(v['inspection_available'] for v in views.values())  # originals travel with the pack
    for name, (status, boxes) in decisions.items():
        body = {'status': status, 'boxes': boxes, 'reviewer': pack.name, 'notes': 'n', 'revision': views[name]['review']['revision']}
        assert client.post(f'/api/reviews/{name}', json=body, headers=HEADERS).status_code == 200


def test_packs_split_whole_panoramas_share_calibration_and_merge_back(tmp_path, monkeypatch):
    batch, taco = make_batch(tmp_path)
    summary = make(batch, ['A', 'B'], tmp_path/'packs', taco, calibration_count=1, root=tmp_path)
    calibration = summary['calibration_panoramas']
    assert len(calibration) == 1
    info = {m: json.loads((tmp_path/f'packs/pack-{m}/pack.json').read_text()) for m in ('A', 'B')}
    assert info['A']['calibration'] == info['B']['calibration']
    assigned = info['A']['assigned'] + info['B']['assigned']
    assert len(assigned) == len(set(assigned)) == 8  # every non-calibration view exactly once
    assert (tmp_path/'packs/geocleanai-label-pack-A.zip').is_file()

    cal = info['A']['calibration'][0]
    label(tmp_path/'packs/pack-A', monkeypatch, {cal: ('litter', [[.1, .1, .2, .2]]), info['A']['assigned'][0]: ('clean', [])})
    label(tmp_path/'packs/pack-B', monkeypatch, {cal: ('clean', []), info['B']['assigned'][0]: ('litter', [[.3, .3, .4, .4]])})
    # A review for an image that is not in the batch is rejected.
    (tmp_path/'packs/pack-B/data/reviews/999_front.jpg.json').write_text(json.dumps({'status': 'clean', 'boxes': [], 'sha256': 'x'}))
    with (tmp_path/'packs/pack-A/taco/review.csv').open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    rows[0].update(status='usable', annotation_complete='yes')
    with (tmp_path/'packs/pack-A/taco/review.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    monkeypatch.chdir(tmp_path)
    # A teammate sends back their labelled pack as a zip.
    import shutil
    returned = shutil.make_archive(str(tmp_path/'returned-B'), 'zip', tmp_path/'packs', 'pack-B')
    result = merge([tmp_path/'packs/pack-A', tmp_path/returned], batch, taco, tmp_path/'packs/merge.json')
    assert result['merged'] == 2 and len(result['rejected']) == 1
    assert not (batch/'reviews'/f'{cal}.json').exists()  # calibration is agreed by the team, not auto-merged
    assert result['calibration_summary']['views_to_discuss'] == [cal]
    merged_taco = {r['image_id']: r for r in csv.DictReader(taco.open(encoding='utf-8'))}
    assert merged_taco[rows[0]['image_id']]['annotation_complete'] == 'yes'

    # Merging again changes nothing; a different answer for a merged view is a conflict, not an overwrite.
    again = merge([tmp_path/'packs/pack-A'], batch, None, tmp_path/'packs/merge2.json')
    assert again['merged'] == 0 and again['unchanged'] == 1
    label(tmp_path/'packs/pack-A', monkeypatch, {info['A']['assigned'][0]: ('litter', [[.5, .5, .6, .6]])})
    monkeypatch.chdir(tmp_path)
    conflict = merge([tmp_path/'packs/pack-A'], batch, None, tmp_path/'packs/merge3.json')
    assert len(conflict['conflicts']) == 1
    assert json.loads((batch/'reviews'/f"{info['A']['assigned'][0]}.json").read_text())['status'] == 'clean'


def test_make_refuses_a_batch_that_already_has_reviews(tmp_path):
    batch, taco = make_batch(tmp_path)
    (batch/'reviews').mkdir(); (batch/'reviews/101_front.jpg.json').write_text('{}')
    with pytest.raises(ValueError, match='already has reviews'):
        make(batch, ['A'], tmp_path/'packs', taco, root=tmp_path)


def test_packs_follow_fixed_photo_ranges_and_shared_calibration(tmp_path):
    batch, taco = make_batch(tmp_path)  # 5 panoramas x 2 views = photos 1-10
    ranges = {'A': (1, 4), 'B': (5, 10)}
    summary = make(batch, ['A', 'B'], tmp_path/'packs', taco, root=tmp_path, photo_ranges=ranges,
                   taco_ranges={'A': (1, 1), 'B': (2, 4)}, calibration_photos=(1, 2))
    info = {m: json.loads((tmp_path/f'packs/pack-{m}/pack.json').read_text()) for m in 'AB'}
    assert summary['calibration_panoramas'] == ['101']
    assert info['A']['calibration'] == info['B']['calibration'] == ['101_front.jpg', '101_back.jpg']
    assert info['A']['assigned'] == ['102_front.jpg', '102_back.jpg']
    assert len(info['B']['assigned']) == 6 and info['A']['taco_rows'] == ['0'] and info['B']['taco_rows'] == ['1', '2', '3']
    with pytest.raises(ValueError, match='splits panoramas'):
        make(batch, ['A', 'B'], tmp_path/'packs2', taco, root=tmp_path, photo_ranges={'A': (1, 3), 'B': (4, 10)})
    with pytest.raises(ValueError, match='exactly once'):
        make(batch, ['A', 'B'], tmp_path/'packs3', taco, root=tmp_path, photo_ranges={'A': (1, 4), 'B': (5, 8)})
