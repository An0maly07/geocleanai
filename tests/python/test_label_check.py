import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from training.label_check import build, compare, needs_fix, sample
from training.review_server import create_app

HEADERS = {'X-Review-Client': 'geocleanai'}


def make_source(tmp_path, n=20):
    source = tmp_path / 'src'; (source / 'images').mkdir(parents=True)
    views, labels = [], {}
    for i in range(n):
        name = f'p{i}_front.jpg'
        Image.new('RGB', (64, 64), (i * 10, 50, 50)).save(source / 'images' / name)
        views.append({'file_name': name, 'image_id': f'p{i}', 'view_id': 'front', 'capture_date': '2025-01-01', 'contributor': 'c'})
        labels[name[:-4]] = [[1, .1, .1, .3, .3]] if i % 4 == 0 else []
    (source / 'manifest.json').write_text(json.dumps({'analysis_version': 'v', 'views': views}))
    (source / 'claude_labels.json').write_text(json.dumps(labels))
    return source


def test_sample_takes_half_litter_half_clean():
    views = [{'file_name': f'{i}.jpg'} for i in range(40)]
    labels = {str(i): ([[0, 0, 0, 1, 1]] if i < 10 else []) for i in range(40)}
    picked = sample(views, labels, fraction=.2)
    assert len(picked) == 8
    assert sum(bool(labels[v['file_name'][:-4]]) for v in picked) == 4


@pytest.mark.parametrize('claude, review, reasons', [
    ([[.1, .1, .3, .3, 1]], {'status': 'litter', 'boxes': [[.1, .1, .3, .3]], 'box_sources': ['ai']}, []),
    ([], {'status': 'clean', 'boxes': []}, []),
    ([[.1, .1, .3, .3, 1]], {'status': 'clean', 'boxes': []}, ['reviewer marked clean']),
    ([], {'status': 'litter', 'boxes': [[.5, .5, .6, .6]], 'box_sources': ['hand']}, ['reviewer marked litter', '1 box(es) drawn by the reviewer']),
    ([[.1, .1, .3, .3, 1], [.5, .5, .6, .6, 1]], {'status': 'litter', 'boxes': [[.1, .1, .3, .3]], 'box_sources': ['ai']}, ['1 Claude box(es) rejected']),
])
def test_needs_fix(claude, review, reasons):
    assert needs_fix(claude, review) == reasons


def test_review_page_shows_claude_boxes_and_compare_scores_it(tmp_path):
    check = tmp_path / 'check'
    result = build(make_source(tmp_path), check, fraction=.4)
    assert result['views'] == 8 and result['claude_litter_views'] == 4
    with pytest.raises(ValueError, match='new folder'):
        build(tmp_path / 'src', check)
    client = TestClient(create_app(check))
    views = client.get('/api/views').json()
    litter = [v for v in views if v['suggestions']]
    assert len(litter) == 4 and litter[0]['suggestions'][0][:4] == [.1, .1, .3, .3]
    for i, v in enumerate(views):
        if v['suggestions']:
            body = {'status': 'litter', 'boxes': [v['suggestions'][0][:4]], 'box_sources': ['ai'], 'reviewer': 'T'}
        else:
            body = {'status': 'litter' if i == 0 else 'clean', 'boxes': [[.2, .2, .4, .4]] if i == 0 else [],
                    'box_sources': ['hand'] if i == 0 else [], 'reviewer': 'T'}
        assert client.post(f"/api/reviews/{v['file_name']}", json=body, headers=HEADERS).status_code == 200
    scored = compare(check)
    assert scored['checked'] == 8 and scored['unreviewed'] == 0
    expected = 1 if not views[0]['suggestions'] else 0
    assert scored['needed_fix'] == expected
    assert scored['passed'] is True
