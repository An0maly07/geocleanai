import json

import pytest
from PIL import Image

from training.prepare_plitter import category_counts, merge, prepare_from_files


def coco_file(tmp_path, name, city, start):
    images, annotations = [], []
    for i in range(start, start + 3):
        file_name = f'{city}/{i}.jpg'
        (tmp_path/'images'/city).mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (40, 20), (i * 4, 20, 30)).save(tmp_path/'images'/file_name)
        images.append({'id': i - start + 1, 'file_name': file_name, 'width': 40, 'height': 20})
        annotations.append({'id': len(annotations) + 1, 'image_id': i - start + 1, 'bbox': [2, 2, 8, 6], 'category_id': 1})
        annotations.append({'id': len(annotations) + 1, 'image_id': i - start + 1, 'bbox': [20, 4, 6, 6], 'category_id': 2})
    path = tmp_path/name
    path.write_text(json.dumps({'images': images, 'annotations': annotations,
                                'categories': [{'id': 1, 'name': 'plastic bag'}, {'id': 2, 'name': 'Garbage Bin'}]}))
    return path


def test_merge_renumbers_ids_and_keeps_bins_out_of_litter(tmp_path):
    files = [coco_file(tmp_path, f'{city}.json', city, 10 * n) for n, city in enumerate(['chiangrai', 'cantho'])]
    cocos = [json.loads(path.read_text()) for path in files]
    assert category_counts(cocos) == {'plastic bag': 6, 'Garbage Bin': 6}
    merged, groups, kept = merge(cocos, ['bin'])
    assert len({row['id'] for row in merged['images']}) == 6  # both files used ids 1-3
    assert kept == {'litter': 6, 'not-litter (skipped)': 6}
    assert all(ann['bbox'] == [2, 2, 8, 6] for ann in merged['annotations'])
    assert set(groups.values()) == {'chiangrai', 'cantho'}


def test_group_regex_and_ungrouped_files(tmp_path):
    coco = {'images': [{'id': 1, 'file_name': 'route7_000123.jpg', 'width': 4, 'height': 4}], 'annotations': []}
    assert merge([coco], ['bin'], r'^(route\d+)_')[1] == {'1': 'route7'}
    with pytest.raises(ValueError, match='group-regex'):
        merge([coco], ['bin'])


def test_cities_never_share_a_split(tmp_path):
    files = [coco_file(tmp_path, f'c{n}.json', f'city{n}', 10 * n) for n in range(6)]
    counts = prepare_from_files(files, tmp_path/'images', tmp_path/'out', ['bin'])
    assert sum(counts.values()) == 18
    rows = json.loads((tmp_path/'out/splits.json').read_text())['images']
    for group in {row['group'] for row in rows}:
        assert len({row['split'] for row in rows if row['group'] == group}) == 1
    for label in (tmp_path/'out/labels').rglob('*.txt'):
        assert label.read_text().count('\n') == 0 and label.read_text().startswith('0 ')  # bin box dropped
    assert 'CC BY 4.0' in (tmp_path/'out/credits.csv').read_text()
