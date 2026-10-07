import json

import pytest
from PIL import Image

from training.build_pune_focus import build
from training.split_pune_labels import split
from test_split_pune import make_export


def make_taco(tmp_path):
    taco, tiles = tmp_path/'litter', tmp_path/'litter-v2'
    records = []
    for split_name, names in (('train', ['0.jpg', '1.jpg', '2.jpg']), ('test', ['9.jpg'])):
        for folder in (taco, tiles):
            (folder/'images'/split_name).mkdir(parents=True, exist_ok=True); (folder/'labels'/split_name).mkdir(parents=True, exist_ok=True)
        for name in names:
            Image.new('RGB', (32, 32)).save(taco/'images'/split_name/name)
            (taco/'labels'/split_name/name.replace('.jpg', '.txt')).write_text('0 .5 .5 .2 .2\n')
            records.append({'name': name, 'split': split_name, 'kind': 'original', 'groups': ['batch_1']})
    for name in ('0_r640_0.jpg', '1_r640_0.jpg'):
        Image.new('RGB', (32, 32)).save(tiles/'images/train'/name)
        (tiles/'labels/train'/name.replace('.jpg', '.txt')).write_text('0 .5 .5 .1 .1\n')
        records.append({'name': name, 'split': 'train', 'kind': 'derived', 'groups': ['batch_1']})
    (tiles/'splits.json').write_text(json.dumps({'images': records}))
    return taco, tiles


def test_pune_focus_dataset_validates_on_pune_and_keeps_taco_names(tmp_path):
    export = make_export(tmp_path, [('g1', 3), ('g1', 0), ('g2', 2), ('g3', 1), ('g4', 0), ('g4', 4)])
    split(export, tmp_path/'candidates.json', tmp_path/'train', tmp_path/'val', tmp_path/'test.json')
    taco, tiles = make_taco(tmp_path)
    report = build(tmp_path/'focus', tmp_path/'train', tmp_path/'val', taco, tiles, taco_originals=2, taco_tiles=1, pune_repeat=2)
    train_views = json.loads((tmp_path/'train/evaluation-lock.json').read_text())['views']
    val_views = json.loads((tmp_path/'val/evaluation-lock.json').read_text())['views']
    assert report['counts']['train'] == {'pune': 2 * train_views, 'original': 2, 'derived': 1}
    assert report['counts']['val'] == {'pune': val_views, 'original': 0, 'derived': 0}
    assert report['counts']['test']['original'] == 1
    assert (tmp_path/'focus/images/test/9.jpg').read_bytes() == (taco/'images/test/9.jpg').read_bytes()
    assert len(list((tmp_path/'focus/images/val').iterdir())) == val_views
    with pytest.raises(ValueError, match='exists'):
        build(tmp_path/'focus', tmp_path/'train', tmp_path/'val', taco, tiles)
    (tmp_path/'val/labels'/next((tmp_path/'val/labels').iterdir()).name).write_text('changed')
    with pytest.raises(ValueError, match='changed since the split'):
        build(tmp_path/'focus2', tmp_path/'train', tmp_path/'val', taco, tiles)
