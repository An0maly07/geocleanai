import json
from pathlib import Path
import pytest
from PIL import Image
from training.prepare_dataset import prepare

def test_capture_groups_never_cross_splits(tmp_path):
    images = tmp_path/'source'; images.mkdir()
    rows, groups, annotations = [], {}, []
    for i in range(12):
        name = f'{i}.jpg'
        Image.new('RGB',(20,10),(i*15,10,20)).save(images/name)
        rows.append({'id':i,'file_name':name,'width':20,'height':10})
        groups[str(i)] = f'sequence-{i//2}'
        annotations.append({'image_id':i,'bbox':[2,2,4,4],'category_id':9})
    ann = tmp_path/'annotations.json'; ann.write_text(json.dumps({'images':rows,'annotations':annotations}))
    group_file = tmp_path/'groups.json'; group_file.write_text(json.dumps(groups))
    output = tmp_path/'output'
    counts = prepare(ann,images,group_file,output)
    assert sum(counts.values()) == 12 and all(counts.values())
    split_data = json.loads((output/'splits.json').read_text())['images']
    for group in set(groups.values()):
        assert len({row['split'] for row in split_data if row['group']==group}) == 1
    for label in (output/'labels').rglob('*.txt'):
        assert label.read_text().startswith('0 ')

def test_groups_are_required(tmp_path):
    image = tmp_path/'image.jpg'; Image.new('RGB',(20,10)).save(image)
    ann = tmp_path/'annotations.json'; ann.write_text(json.dumps({'images':[{'id':1,'file_name':'image.jpg','width':20,'height':10}],'annotations':[]}))
    groups = tmp_path/'groups.json'; groups.write_text('{}')
    with pytest.raises(ValueError, match='capture/sequence'):
        prepare(ann,tmp_path,groups,tmp_path/'out')


def test_experiment_configs_are_valid_and_defaults_match_baseline(tmp_path):
    from training.train import DEFAULTS, load_config
    # With no config, settings equal the first baseline run.
    assert {k: DEFAULTS[k] for k in ('model', 'imgsz', 'batch', 'epochs', 'patience', 'seed')} == \
        {'model': 'yolo26n.pt', 'imgsz': 640, 'batch': 8, 'epochs': 50, 'patience': 10, 'seed': 42}
    root = Path(__file__).resolve().parents[2]
    configs = sorted((root/'training/experiments').glob('*.json'))
    assert len([p for p in configs if p.name.startswith('e')]) <= 3  # At most three first-stage experiments (E1-E3).
    for path in configs:
        config = load_config(path, {})
        assert config['ultralytics_version'] == '8.4.155' and config['seed'] == 42 and config['imgsz'] == 640
        assert config['batch_fallbacks'] == sorted(config['batch_fallbacks'], reverse=True)
        assert config['question']
        if path.name.startswith('s'):  # Second stages continue from a named, hash-checked checkpoint.
            assert config['model'].endswith('.pt') and len(config['model_sha256']) == 64
    typo = tmp_path/'typo.json'; typo.write_text(json.dumps({'epoch': 5}))
    with pytest.raises(ValueError, match='Unknown config keys'):
        load_config(typo, {})
