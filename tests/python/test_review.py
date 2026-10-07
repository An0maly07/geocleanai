import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from training.review_server import ReviewStore, create_app
from backend.panorama import ANALYSIS_VERSION


@pytest.fixture
def review_data(tmp_path):
    root = tmp_path / 'review'
    (root / 'images').mkdir(parents=True)
    views = []
    for i in range(3):
        name = f'{i}_front.jpg'
        Image.new('RGB', (640, 640), (30 + i, 40, 50)).save(root / 'images' / name)
        views.append({'file_name': name, 'image_id': str(i), 'view_id': 'front',
                      'original_resolution': True, 'sequence': f'sequence-{i}'})
    (root / 'manifest.json').write_text(json.dumps({'analysis_version': 'test', 'views': views}))
    return root


def post(client, name='0_front.jpg', **changes):
    payload = {'status': 'litter', 'boxes': [[.1, .2, .4, .6]], 'reviewer': 'Test reviewer', 'revision': 0}
    payload.update(changes)
    return client.post(f'/api/reviews/{name}', headers={'X-Review-Client': 'geocleanai'}, json=payload)


def test_explicit_review_and_export(review_data, tmp_path):
    client = TestClient(create_app(review_data))
    assert post(client).status_code == 200
    assert post(client, '1_front.jpg', status='clean', boxes=[]).status_code == 200
    assert post(client, '2_front.jpg', status='skip', boxes=[], notes='Blurred').status_code == 200
    assert not (review_data / 'labels').exists()
    output = tmp_path / 'export'
    assert ReviewStore(review_data).export(output)['exported'] == 2
    values = list(map(float, (output / 'labels/0_front.txt').read_text().split()))
    assert values == pytest.approx([0, .25, .4, .3, .4])
    assert (output / 'labels/1_front.txt').read_text() == ''
    assert not (output / 'labels/2_front.txt').exists()
    with pytest.raises(ValueError, match='previous exports'):
        ReviewStore(review_data).export(output)


def test_invalid_boxes_and_false_clean_are_rejected(review_data):
    client = TestClient(create_app(review_data))
    for changes in [{'status': 'clean'}, {'boxes': []}, {'boxes': [[.4, .2, .1, .5]]},
                    {'boxes': [[-.1, .2, .4, .6]]}, {'boxes': [[.1, .2, .4]]},
                    {'reviewer': '  '}, {'status': 'skip', 'boxes': []}]:
        assert post(client, **changes).status_code == 422
    assert client.get('/api/views').json()[0]['review']['status'] == 'unreviewed'


def test_stale_review_and_changed_photo(review_data):
    client = TestClient(create_app(review_data))
    assert post(client).status_code == 200
    assert post(client, status='clean', boxes=[]).status_code == 409
    assert TestClient(create_app(review_data)).get('/api/views').json()[0]['review']['status'] == 'litter'
    Image.new('RGB', (640, 640), 'white').save(review_data / 'images/0_front.jpg')
    store = ReviewStore(review_data)
    assert store.read('0_front.jpg')['status'] == 'unreviewed'
    with pytest.raises(ValueError, match='No human-reviewed'):
        store.export(review_data.parent / 'export')


def test_local_writes_and_manifest_only_files(review_data):
    client = TestClient(create_app(review_data))
    assert client.post('/api/reviews/0_front.jpg', json={}).status_code == 403
    assert client.post('/api/reviews/0_front.jpg', headers={
        'X-Review-Client': 'geocleanai', 'Origin': 'https://other.example'}, json={}).status_code == 403
    assert client.get('/images/.env.local').status_code == 404
    assert post(client, '../other').status_code == 404


def test_preview_cannot_be_exported_as_evaluation(review_data, tmp_path):
    manifest = review_data / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['views'][0]['original_resolution'] = False
    manifest.write_text(json.dumps(data))
    client = TestClient(create_app(review_data))
    assert post(client).status_code == 200
    with pytest.raises(ValueError, match='original imagery'):
        ReviewStore(review_data).export(tmp_path / 'export')


def test_original_detail_preserves_labels_and_caches_projection(review_data):
    source = review_data / 'panorama.jpg'
    Image.new('RGB', (3840, 1920), (40, 90, 170)).save(source)
    manifest = review_data / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['analysis_version'] = ANALYSIS_VERSION
    data['views'][0]['source_file'] = str(source)
    manifest.write_text(json.dumps(data))
    store = ReviewStore(review_data)
    original_hash = store.hashes['0_front.jpg']
    detail = store.inspection('0_front.jpg')
    with Image.open(detail) as image:
        assert image.size == (1280, 1280)
    saved_time = detail.stat().st_mtime_ns
    assert store.inspection('0_front.jpg').stat().st_mtime_ns == saved_time
    assert ReviewStore(review_data).hashes['0_front.jpg'] == original_hash
    client = TestClient(create_app(review_data))
    assert client.get('/api/views').json()[0]['inspection_available']
    assert client.get('/inspection/.env.local').status_code == 404


def write_suggestions(root, threshold=.3, sha=None):
    import hashlib
    views = {}
    for i in range(3):
        name = f'{i}_front.jpg'
        digest = sha or hashlib.sha256((root / 'images' / name).read_bytes()).hexdigest()
        views[name] = {'sha256': digest, 'boxes': [[.1, .1, .2, .2, .9], [.5, .5, .6, .6, .1]]}
    data = {'detector': 'test-detector', 'threshold': threshold, 'views': views,
            'evaluation': {'chosen': {'precision': .4, 'recall': .5}}}
    path = root.parent / 'suggestions.json'
    path.write_text(json.dumps(data))
    return path


def test_suggestions_are_shown_only_on_unreviewed_photos_and_box_sources_are_kept(review_data):
    from training.import_suggestions import install
    install(write_suggestions(review_data), review_data)
    client = TestClient(create_app(review_data))
    assert post(client, '1_front.jpg', box_sources=['ai'], suggestions_shown=1).status_code == 200
    views = {v['file_name']: v for v in client.get('/api/views').json()}
    assert views['0_front.jpg']['suggestions'] == [[.1, .1, .2, .2, .9]]  # below-threshold box hidden
    assert views['1_front.jpg']['suggestions'] == []  # reviewed: no suggestions
    assert views['1_front.jpg']['review']['box_sources'] == ['ai']
    assert client.get('/api/assist').json()['precision'] == .4
    assert post(client, '2_front.jpg', box_sources=['ai', 'hand']).status_code == 422
    assert post(client, '2_front.jpg', box_sources=['robot']).status_code == 422


def test_suggestions_for_changed_images_are_not_installed(review_data):
    from training.import_suggestions import install
    result = install(write_suggestions(review_data, sha='0' * 64), review_data)
    assert result['views_with_suggestions'] == 0 and result['skipped_changed_images'] == 3
    assert all(v['suggestions'] == [] for v in TestClient(create_app(review_data)).get('/api/views').json())
