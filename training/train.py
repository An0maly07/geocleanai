"""Fine-tune a YOLO26 litter detector from a versioned experiment config and export it.

The config (JSON) fixes the model, data, resolution, batch, epochs, augmentation and
seed. Every run records its config, library versions and training logs next to the
weights. Pune accuracy is never computed here; it comes only from the locked
evaluation step after a candidate has been selected on validation data.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Settings the first baseline used; a config only needs to state what it changes.
DEFAULTS = {'model': 'yolo26n.pt', 'pretrained': None, 'imgsz': 640, 'batch': 8, 'epochs': 50, 'patience': 10, 'seed': 42,
            'deterministic': True, 'workers': 2, 'save_period': 5, 'batch_fallbacks': [], 'augment': {},
            'ultralytics_version': None, 'confidence_threshold': .35}
LOGS = ('results.csv', 'args.yaml', 'results.png', 'BoxPR_curve.png', 'BoxF1_curve.png', 'confusion_matrix.png',
        'labels.jpg', 'train_batch0.jpg', 'val_batch0_labels.jpg', 'val_batch0_pred.jpg')


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def check_version(expected):
    import ultralytics
    if expected and ultralytics.__version__ != expected:
        raise RuntimeError(f'Ultralytics {ultralytics.__version__} is loaded but the config pins {expected}. '
                           'Reinstall the pinned version and restart the runtime before training or export.')
    return ultralytics.__version__


def load_config(path: Path | None, overrides: dict) -> dict:
    config = {**DEFAULTS, **(json.loads(path.read_text()) if path else {}), **{k: v for k, v in overrides.items() if v is not None}}
    unknown = set(config) - set(DEFAULTS) - {'name', 'data', 'version', 'question', 'tiling'}
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    return config


def resolved_dataset(source_yaml: Path) -> Path:
    import yaml
    dataset = yaml.safe_load(source_yaml.read_text())
    if dataset.get('names') not in ({0: 'litter'}, ['litter']):
        raise ValueError('The training dataset must contain exactly one class: litter.')
    dataset['path'] = str(source_yaml.parent)
    resolved = source_yaml.with_name('dataset.resolved.yaml')
    resolved.write_text(yaml.safe_dump(dataset))
    return resolved


def fit(config: dict, data: Path, output: Path):
    """Train, halving through config['batch_fallbacks'] on CUDA out-of-memory. Returns (model, batch used)."""
    import torch
    from ultralytics import YOLO
    for batch in [config['batch'], *config['batch_fallbacks']]:
        name = config['name'] if batch == config['batch'] else f"{config['name']}-b{batch}"
        model = YOLO(config['model'])
        try:
            # A .yaml architecture (e.g. the P2 head) starts from the weights named in 'pretrained'.
            extra = {'pretrained': config['pretrained']} if config['pretrained'] else {}
            model.train(data=str(data), epochs=config['epochs'], imgsz=config['imgsz'], batch=batch,
                        patience=config['patience'], seed=config['seed'], deterministic=config['deterministic'],
                        workers=config['workers'], save_period=config['save_period'],
                        device=0 if torch.cuda.is_available() else 'cpu',
                        project=str(output/'runs'), name=name, exist_ok=False, **extra, **config['augment'])
            return model, batch
        except torch.cuda.OutOfMemoryError:
            print(f'Out of GPU memory at batch {batch}; trying the next fallback.')
            del model
            torch.cuda.empty_cache()
    raise RuntimeError('Out of GPU memory at every configured batch size.')


def train(args):
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    config = load_config(getattr(args, 'config', None), {'epochs': getattr(args, 'epochs', None)})
    config.setdefault('name', 'litter')
    config['version'] = getattr(args, 'version', None) or config.get('version') or 'pune-litter-candidate'
    started_version = check_version(config['ultralytics_version'])
    source_yaml = args.data.resolve()
    resolved = resolved_dataset(source_yaml)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    if args.resume:
        # Resume keeps the checkpoint's own arguments; change nothing else.
        model = YOLO(str(args.resume))
        model.train(resume=True)
        batch = None
    else:
        model, batch = fit(config, resolved, args.output)
    best_path = Path(model.trainer.best)
    run_dir = best_path.parent.parent
    logs = args.output/'logs'; logs.mkdir(exist_ok=True)
    for name in LOGS:
        if (run_dir/name).is_file():
            shutil.copy2(run_dir/name, logs/name)
    best = YOLO(str(best_path))
    # Supporting TACO metric at Ultralytics' own operating point; not comparable with Pune at a fixed threshold.
    metrics = best.val(data=str(resolved), split='test', imgsz=config['imgsz'], plots=True)
    export_version = check_version(config['ultralytics_version'])
    exported = Path(best.export(format='onnx', imgsz=640, dynamic=False, half=False, batch=1, simplify=False, opset=17, nms=False, max_det=100, device='cpu'))
    session = ort.InferenceSession(str(exported), providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape != [1,3,640,640] or session.get_outputs()[0].shape[-1] != 6:
        raise RuntimeError('Export is not the end-to-end format expected by the API. Do not deploy it.')
    target = args.output/'litter.onnx'; shutil.copy2(exported, target)
    shutil.copy2(best_path, args.output/'best.pt')
    versions = {'ultralytics': export_version, 'ultralytics_at_start': started_version, 'torch': torch.__version__, 'onnxruntime': ort.__version__}
    pending = {'status': 'not evaluated', 'note': 'Pune results are written only by the locked evaluation step (pune-evaluation.json).'}
    (args.output/'metrics.json').write_text(json.dumps({'split': 'taco-test', 'operating_point': 'ultralytics val default (not a fixed threshold)',
                                                        'results': {k: float(v) for k,v in metrics.results_dict.items()},
                                                        'versions': versions, 'batch_used': batch, 'pune_evaluation': pending}, indent=2))
    manifest = {'version': config['version'], 'trained': True, 'classes': ['litter'], 'format': 'yolo26-end2end', 'input_size': 640,
                'confidence_threshold': config['confidence_threshold'], 'confidence_threshold_source': 'config default; replace only via a validation selection record',
                'sha256': sha256(target), 'trained_at': datetime.now(timezone.utc).isoformat(), 'training_dataset': str(source_yaml.name),
                'split_manifest_sha256': sha256(source_yaml.parent/'splits.json') if (source_yaml.parent/'splits.json').is_file() else None,
                'config_sha256': sha256(args.output/'config.json'), 'versions': versions, 'pune_evaluation': pending}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Trained artifacts: {args.output}. Select on validation data before any Pune evaluation.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Experiment config JSON (see training/experiments/)')
    parser.add_argument('--output', type=Path, default=Path('training-output'))
    parser.add_argument('--version')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', type=Path)
    train(parser.parse_args())
