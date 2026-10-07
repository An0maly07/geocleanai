import json

from PIL import Image

from backend.panorama import panorama_views
from scripts.build_team_labels import build


def make_batch(tmp_path, reviewed):
    batch = tmp_path/'batch'; (batch/'images').mkdir(parents=True); (batch/'reviews').mkdir()
    views = []
    for image_id in reviewed:
        for view_id, _ in panorama_views():
            name = f'{image_id}_{view_id}.jpg'
            Image.new('RGB', (64, 64)).save(batch/'images'/name)
            views.append({'file_name': name, 'image_id': image_id, 'view_id': view_id})
    (batch/'manifest.json').write_text(json.dumps({'views': views}))
    for image_id, statuses in reviewed.items():
        for view_id, status in statuses.items():
            boxes = [[.4, .4, .5, .5]] if status == 'litter' else []
            (batch/'reviews'/f'{image_id}_{view_id}.jpg.json').write_text(json.dumps({'status': status, 'boxes': boxes, 'reviewer': ' Asha '}))
    candidates = tmp_path/'candidates.json'
    candidates.write_text(json.dumps([{'id': i, 'lat': 18.5, 'lng': 73.8, 'captured_at': 1, 'creator': 'someone'} for i in reviewed]))
    return batch, candidates


def test_only_fully_reviewed_panoramas_are_published_with_their_evidence(tmp_path):
    full = {view_id: 'clean' for view_id, _ in panorama_views()} | {'front': 'litter', 'down': 'skip'}
    partial = {view_id: 'clean' for view_id, _ in list(panorama_views())[:5]}
    batch, candidates = make_batch(tmp_path, {'111': full, '222': partial})
    summary = build(batch, candidates, tmp_path/'out')
    index = json.loads((tmp_path/'out/index.json').read_text())
    assert summary == {'panoramas': 1, 'with_litter': 1, 'markers': 1, 'partly_reviewed_skipped': 1}
    entry = index['111']
    assert entry['reviewers'] == ['Asha'] and entry['unclear_views'] == ['down']
    assert entry['detections'][0]['view_id'] == 'front' and len(entry['detections'][0]['tag']) == 2
    assert entry['views'] == [{'id': 'front', 'view': dict(panorama_views())['front'].model_dump(mode='json'), 'preview': '/team-labels/111_front.jpg'}]
    assert (tmp_path/'out/111_front.jpg').is_file() and not (tmp_path/'out/111_back.jpg').exists()
