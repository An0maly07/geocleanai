import json
import zipfile

from PIL import Image

from backend.panorama import panorama_views
from training import package_suggestions


def test_bundle_has_every_view_and_only_reviewed_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(package_suggestions, 'SIZE', 32)
    batch = tmp_path/'batch'; (batch/'images').mkdir(parents=True); (batch/'reviews').mkdir()
    panorama = tmp_path/'panorama-1.jpg'; Image.new('RGB', (200, 100), (90, 90, 90)).save(panorama)
    views = []
    for view_id, _ in panorama_views():
        name = f'1_{view_id}.jpg'
        Image.new('RGB', (16, 16)).save(batch/'images'/name)
        views.append({'file_name': name, 'image_id': '1', 'view_id': view_id, 'source_file': str(panorama), 'original_resolution': True})
    (batch/'manifest.json').write_text(json.dumps({'views': views}))
    (batch/'reviews/1_front.jpg.json').write_text(json.dumps({'status': 'litter', 'boxes': [[.1, .1, .2, .2]]}))
    (batch/'reviews/1_back.jpg.json').write_text(json.dumps({'status': 'skip', 'boxes': []}))
    summary = package_suggestions.package(batch, tmp_path/'bundle.zip', workers=1)
    assert summary['views'] == 6 and summary['labelled_for_scoring'] == 1
    with zipfile.ZipFile(tmp_path/'bundle.zip') as archive:
        manifest = json.loads(archive.read('suggest-manifest.json'))
        assert sorted(n for n in archive.namelist() if n.startswith('views/')) == sorted(f'views/{v["file_name"]}' for v in views)
        with archive.open('views/1_front.jpg') as image:
            assert Image.open(image).size == (32, 32)
    assert manifest['labels'] == {'1_front.jpg': {'status': 'litter', 'boxes': [[.1, .1, .2, .2]]}}
    assert (tmp_path/'bundle.zip.sha256').read_text().strip() == summary['sha256']
