import json

import pytest
from PIL import Image

from training.split_pune_labels import choose_validation, split


def test_validation_takes_whole_groups_with_enough_litter():
    stats = {'big': {'panoramas': 22, 'views': 132, 'litter_views': 10, 'regions': 30},
             'a': {'panoramas': 8, 'views': 48, 'litter_views': 2, 'regions': 4},
             'b': {'panoramas': 3, 'views': 18, 'litter_views': 3, 'regions': 12},
             'c': {'panoramas': 2, 'views': 12, 'litter_views': 0, 'regions': 0},
             'd': {'panoramas': 5, 'views': 30, 'litter_views': 1, 'regions': 2}}
    chosen, summary = choose_validation(stats)
    assert 'big' not in chosen and summary['met_region_constraint']
    assert .20 <= summary['validation_region_share'] <= .45
    # 40 panoramas: a third is ~13 -> {a, b, c} = 13 panoramas, 16/48 regions (33%).
    assert chosen == ['a', 'b', 'c']


def make_export(tmp_path, groups_regions):
    export = tmp_path/'export'
    (export/'images').mkdir(parents=True); (export/'labels').mkdir()
    views, candidates = [], []
    for index, (group, regions) in enumerate(groups_regions):
        image_id = str(1000 + index)
        candidates.append({'id': image_id, 'location_group': group})
        for view_id in ('front', 'back'):
            name = f'{image_id}_{view_id}.jpg'
            Image.new('RGB', (64, 64)).save(export/'images'/name)
            (export/'labels'/f'{image_id}_{view_id}.txt').write_text('0 .5 .5 .1 .1\n' * regions if view_id == 'front' else '')
            views.append({'file_name': name, 'image_id': image_id, 'view_id': view_id, 'sequence': f'seq-{group}'})
    (export/'manifest.json').write_text(json.dumps({'views': views}))
    (tmp_path/'candidates.json').write_text(json.dumps(candidates))
    (tmp_path/'test.json').write_text(json.dumps({'views': [{'sequence': 'locked'}]}))
    return export


def test_split_never_puts_a_group_on_both_sides(tmp_path):
    export = make_export(tmp_path, [('g1', 3), ('g1', 0), ('g2', 2), ('g3', 1), ('g4', 0), ('g4', 4)])
    result = split(export, tmp_path/'candidates.json', tmp_path/'train', tmp_path/'val', tmp_path/'test.json')
    train = json.loads((tmp_path/'train/evaluation-lock.json').read_text())['files']
    val = json.loads((tmp_path/'val/evaluation-lock.json').read_text())['files']
    assert {f['location_group'] for f in train}.isdisjoint({f['location_group'] for f in val})
    assert len(train) + len(val) == 12 and result['validation']['litter_views'] > 0
    with pytest.raises(ValueError, match='never overwritten'):
        split(export, tmp_path/'candidates.json', tmp_path/'train', tmp_path/'val2', tmp_path/'test.json')


def test_split_refuses_test_sequences(tmp_path):
    export = make_export(tmp_path, [('g1', 1), ('g2', 1)])
    (tmp_path/'test.json').write_text(json.dumps({'views': [{'sequence': 'seq-g2'}]}))
    with pytest.raises(ValueError, match='locked test sequences'):
        split(export, tmp_path/'candidates.json', tmp_path/'train', tmp_path/'val', tmp_path/'test.json')


def test_kept_validation_reuses_earlier_groups(tmp_path):
    export = make_export(tmp_path, [('g1', 3), ('g1', 0), ('g2', 2), ('g3', 1), ('g4', 0), ('g4', 4)])
    first = split(export, tmp_path/'candidates.json', tmp_path/'train1', tmp_path/'val1', tmp_path/'test.json')
    result = split(export, tmp_path/'candidates.json', tmp_path/'train2', tmp_path/'val2', tmp_path/'test.json',
                   keep_validation=tmp_path/'val1/evaluation-lock.json')
    assert result['validation_groups'] == sorted(first['validation_groups'])
    assert 'kept from val1' in result['rule']
    with pytest.raises(ValueError, match='not a Pune validation'):
        split(export, tmp_path/'candidates.json', tmp_path/'train3', tmp_path/'val3', tmp_path/'test.json',
              keep_validation=tmp_path/'train1/evaluation-lock.json')
