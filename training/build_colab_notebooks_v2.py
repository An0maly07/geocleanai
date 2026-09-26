"""Build the v2 Colab notebooks: A (diagnosis, no training) and B (experiments and locked evaluation)."""
import argparse
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells: list[tuple[str, str]], name: str) -> dict:
    built = []
    for index, (kind, source) in enumerate(cells):
        cell = {'cell_type': kind, 'id': f'geoclean-v2-{index}', 'metadata': {},
                'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code':
            compile(''.join(cell['source']), f'{name}-cell-{index}', 'exec')
            cell.update(execution_count=None, outputs=[])
        built.append(cell)
    return {'nbformat': 4, 'nbformat_minor': 5, 'cells': built, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': name}, 'accelerator': 'GPU'}}


def setup_cells(v1_sha: str, v2_name: str, v2_sha: str) -> list[tuple[str, str]]:
    return [
        ('code', f'''
            from google.colab import drive
            drive.mount('/content/drive')
            import hashlib, json, shutil, subprocess, sys, zipfile
            from pathlib import Path
            DRIVE = Path('/content/drive/MyDrive/GeoCleanAI')
            V1_ZIP, V2_ZIP = DRIVE/'geocleanai-colab-data-v1.zip', DRIVE/'{v2_name}'
            assert V1_ZIP.is_file(), 'geocleanai-colab-data-v1.zip must stay in My Drive/GeoCleanAI.'
            assert V2_ZIP.is_file(), 'Upload {v2_name} into My Drive/GeoCleanAI first.'
            V1, V2 = Path('/content/geocleanai-v1'), Path('/content/geocleanai-v2')

            def file_hash(path):
                with Path(path).open('rb') as stream:
                    return hashlib.file_digest(stream, 'sha256').hexdigest()

            def unpack(archive, target, expected):
                marker = target/'.bundle-sha256'
                if marker.is_file() and marker.read_text().strip() == expected:
                    return
                assert file_hash(archive) == expected, f'{{archive.name}} is incomplete or a different version.'
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive) as bundle:
                    for name in bundle.namelist():
                        assert (target/name).resolve().is_relative_to(target.resolve()), 'Unsafe archive path'
                    bundle.extractall(target)
                marker.write_text(expected)

            unpack(V1_ZIP, V1, '{v1_sha}')
            unpack(V2_ZIP, V2, '{v2_sha}')
            BUNDLE = json.loads((V2/'bundle-manifest.json').read_text())
            for name, digest in BUNDLE['files'].items():
                assert file_hash(V2/name) == digest, f'Changed file in bundle: {{name}}'
            # Derived datasets reuse TACO originals from v1; restore and verify each one.
            for dataset, rows in BUNDLE['originals_from_v1'].items():
                for row in rows:
                    target = V2/'datasets'/dataset/row['path']
                    if not target.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(V1/'dataset'/row['path'], target)
                    assert file_hash(target) == row['sha256'], f'Original differs from the local build: {{row["path"]}}'
            sys.path.insert(0, str(V2/'code'))
            print('Bundles verified:', V1, V2, '| derived datasets:', list(BUNDLE['originals_from_v1']))
        '''),
        ('markdown', 'Install the pinned training packages. If this cell asks for a restart, use **Runtime → Restart session** and run from the top again.'),
        ('code', '''
            from importlib.metadata import version, PackageNotFoundError
            PIN = 'ultralytics==8.4.155'
            def installed(name):
                try:
                    return version(name)
                except PackageNotFoundError:
                    return None
            if getattr(sys, '_geoclean_restart_required', False):
                raise RuntimeError('Restart the session (Runtime > Restart session), then run from the top.')
            watched = {'Pillow': 'PIL', 'numpy': 'numpy', 'torch': 'torch', 'ultralytics': 'ultralytics', 'onnxruntime': 'onnxruntime'}
            before, loaded = {n: installed(n) for n in watched}, set(sys.modules)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(V2/'code/training/requirements.txt')], check=True)
            if [n for n, m in watched.items() if before[n] != installed(n) and any(k == m or k.startswith(m + '.') for k in loaded)]:
                sys._geoclean_restart_required = True
                raise RuntimeError('Libraries changed after import. Restart the session, then run from the top.')
            import os
            # Stop Ultralytics installing or upgrading packages by itself mid-run (open question U2):
            # a missing package now fails loudly instead of silently changing versions.
            os.environ['YOLO_AUTOINSTALL'] = 'False'
            import ultralytics, torch
            assert ultralytics.__version__ == PIN.split('==')[1], f'Ultralytics {ultralytics.__version__} loaded; expected {PIN}.'
            FREEZE = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout
            print('Ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
        '''),
    ]


def diagnosis_cells(setup) -> list[tuple[str, str]]:
    return [
        ('markdown', '''
            # GeoCleanAI — A. Diagnose the baseline (no training)

            This notebook explains the 0% Pune result of **baseline-v1** before any retraining. It trains nothing and changes no data.

            **Disclosure:** the Pune test results have already been inspected. Everything below that looks at Pune test views is **diagnosis only**. It must not be used to choose thresholds or models, or to drop difficult examples.

            Before running: upload the new bundle into **My Drive/GeoCleanAI** next to the v1 data zip. Optionally set `RUN_FOLDER` below to the first run's Drive folder (`training-output/<date>`) so the real training logs are used.
        '''),
        *setup,
        ('code', '''
            RUN_FOLDER = None  # e.g. DRIVE/'training-output/20260925-170000' — the first training run, for results.csv
            BASE = V2/'baseline-v1/files'
            PUNE, PUNE_1024, TACO = V1/'pune-test', V2/'pune-test-v1-r1024', V1/'dataset'
            OUT = DRIVE/'diagnosis-baseline-v1'
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT/'pip-freeze.txt').write_text(FREEZE)
            REPORT = {'model': 'baseline-v1', 'disclosure': 'Pune test views inspected for diagnosis only.'}
        '''),
        ('markdown', '## 1. Integrity: files, checkpoint, classes, splits, label pairing'),
        ('code', '''
            from training.diagnose import checkpoint_metadata, pairing_report
            checks = {line.split()[1]: line.split()[0] for line in (V2/'baseline-v1/CHECKSUMS.txt').read_text().splitlines()}
            for name, digest in checks.items():
                if name.startswith('files/'):
                    assert file_hash(V2/'baseline-v1'/name) == digest, f'{name} changed'
            manifest = json.loads((BASE/'manifest.json').read_text())
            assert file_hash(BASE/'litter.onnx') == manifest['sha256'], 'ONNX does not match its manifest'
            assert file_hash(TACO/'splits.json') == manifest['split_manifest_sha256'], 'Training split differs from the one used'
            meta = checkpoint_metadata(BASE/'best.pt')
            assert meta['names'] in ({0: 'litter'}, ['litter']), meta['names']
            pairing = pairing_report(TACO)
            REPORT['integrity'] = {'checksums': 'match', 'onnx_matches_manifest': True, 'split_matches': True,
                                   'checkpoint': {k: v for k, v in meta.items() if k not in ('train_results', 'train_metrics')},
                                   'label_pairing': pairing}
            print(json.dumps(REPORT['integrity'], indent=1, default=str)[:3000])
            assert all(not p['images_without_label'] and not p['labels_without_image'] for p in pairing.values())
        '''),
        ('markdown', '## 2. Training curves: how many epochs ran and why training stopped (U1)'),
        ('code', '''
            from training.diagnose import read_results, curve_summary, plot_curves
            from IPython.display import Image as Show, display
            source = Path(RUN_FOLDER)/'runs/litter/results.csv' if RUN_FOLDER else None
            if source and source.is_file():
                results = read_results(source)
            elif meta.get('train_results'):
                results = read_results(meta['train_results'])
                source = 'checkpoint train_results'
            else:
                results = None
                print('No training log found. Set RUN_FOLDER to the first run folder on Drive.')
            if results:
                curves = curve_summary(results, epoch_limit=50, patience=10)
                plot_curves(results, OUT/'training-curves.png'); display(Show(str(OUT/'training-curves.png')))
                REPORT['training_curves'] = {'source': str(source), **curves}
                print(json.dumps(curves, indent=1))
        '''),
        ('markdown', '## 3. TACO test at the fixed 0.35 threshold (comparable with Pune)'),
        ('code', '''
            from argparse import Namespace
            from training.evaluate import evaluate
            taco = evaluate(Namespace(model=BASE/'litter.onnx', dataset=TACO, split='test', output=OUT/'taco-test-0.35.json', confidence=.35, pt=None))
            REPORT['taco_test_at_0.35'] = {k: taco[k] for k in ('true_positives', 'false_positives', 'false_negatives', 'precision', 'recall', 'recall_by_size')}
            print(json.dumps(REPORT['taco_test_at_0.35'], indent=1))
        '''),
        ('markdown', '## 4. Geometry and colour-order audit'),
        ('code', '''
            # Test-only packages at the project's pinned versions (the backend test setup imports them).
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'httpx==0.28.1', 'python-dotenv==1.2.3', 'pytest'], check=True)
            test_run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                       'tests/python/test_evaluate.py', 'tests/python/test_geometry.py', 'tests/python/test_diagnose.py'],
                                      cwd=V2/'code', capture_output=True, text=True)
            print(test_run.stdout[-1500:])
            REPORT['regression_tests_passed'] = test_run.returncode == 0
            # Same photo as PIL RGB and as OpenCV BGR must give the same boxes if colour order is handled.
            import cv2, numpy as np
            from PIL import Image as PILImage
            from ultralytics import YOLO
            model = YOLO(str(BASE/'best.pt'))
            sample = sorted((TACO/'images/val').iterdir())[:10]
            differences = []
            for path in sample:
                a = model.predict(PILImage.open(path).convert('RGB'), imgsz=640, conf=.25, verbose=False)[0].boxes.data.cpu().numpy()
                b = model.predict(cv2.imread(str(path)), imgsz=640, conf=.25, verbose=False)[0].boxes.data.cpu().numpy()
                differences.append(len(a) == len(b) and (len(a) == 0 or float(np.abs(a - b).max()) < 1.0))
            REPORT['rgb_bgr_consistent'] = all(differences)
            print('RGB/BGR consistent on', sum(differences), 'of', len(differences), 'images')
            if RUN_FOLDER:
                for name in ('train_batch0.jpg', 'val_batch0_labels.jpg'):
                    path = Path(RUN_FOLDER)/'runs/litter'/name
                    if path.is_file():
                        print(name, '(how Ultralytics read the labels)'); display(Show(str(path), width=900))
        '''),
        ('markdown', '## 5. Reproduce the official Pune result and check ONNX against PyTorch one-to-one'),
        ('code', '''
            official = evaluate(Namespace(model=BASE/'litter.onnx', dataset=PUNE, output=OUT/'pune-0.35.json', confidence=.35, pt=BASE/'best.pt'))
            reproduced = (official['true_positives'], official['false_positives'], official['false_negatives']) == (0, 21, 54)
            REPORT['pune_official_reproduced'] = reproduced
            failures = [p for p in official['parity'] if not p['passed']]
            REPORT['parity_at_0.35'] = {'passed': official['parity_passed'], 'failures': failures,
                                        'max_corner_px': max(p['max_corner_px'] for p in official['parity']),
                                        'max_confidence_diff': max(p['max_confidence_diff'] for p in official['parity'])}
            print('Reproduced 0/21/54:', reproduced)
            print(json.dumps(REPORT['parity_at_0.35'], indent=1))
        '''),
        ('markdown', '## 6. Diagnostic low-threshold run, near-miss categories and parity at 0.05 (diagnosis only)'),
        ('code', '''
            from training.diagnose import near_miss_table, category_counts, what_if
            low = evaluate(Namespace(model=BASE/'litter.onnx', dataset=PUNE, output=OUT/'pune-0.05-diagnostic-only.json', confidence=.05, pt=BASE/'best.pt'))
            table = near_miss_table(low)
            REPORT['miss_categories'] = category_counts(table)
            REPORT['what_if_iou_0.3'] = what_if(table, .3)
            REPORT['parity_at_0.05'] = {'passed': low['parity_passed'], 'failures': [p['image'] for p in low['parity'] if not p['passed']]}
            (OUT/'near-miss-table.json').write_text(json.dumps(table, indent=1))
            print(json.dumps(REPORT['miss_categories'], indent=1)); print(REPORT['what_if_iou_0.3'])
        '''),
        ('markdown', '## 7. Overlays: labels green, predictions ≥ 0.35 red, weaker predictions dashed orange; 1024px render alongside'),
        ('code', '''
            from training.diagnose import draw_overlay, views_to_inspect
            low_by_image = {r['image']: r for r in low['per_image']}
            chosen = views_to_inspect(official, table)
            for name in chosen:
                draw_overlay(PUNE/'images'/name, low_by_image[name], .35, OUT/'overlays'/name, companion=PUNE_1024/'images'/name)
            REPORT['overlays'] = chosen
            for name in chosen:
                print(name); display(Show(str(OUT/'overlays'/name), width=1100))
        '''),
        ('markdown', '## 8. How TACO differs from Pune: label size, sharpness and density'),
        ('code', '''
            from training.diagnose import box_statistics
            REPORT['domain'] = {f'taco_{s}': box_statistics(TACO/'images'/s, TACO/'labels'/s) for s in ('train', 'val', 'test')}
            REPORT['domain']['pune_v1_640'] = box_statistics(PUNE/'images', PUNE/'labels')
            REPORT['domain']['pune_v1_r1024_at_1024'] = box_statistics(PUNE_1024/'images', PUNE_1024/'labels', input_size=1024)
            print(json.dumps(REPORT['domain'], indent=1))
        '''),
        ('markdown', '## 9. Does the street-like validation proxy reproduce the collapse? (runs only if litter-v2 is in the bundle)'),
        ('code', '''
            from training.evaluate import sweep
            PROXY = V2/'datasets/litter-v2'
            if PROXY.is_dir():
                proxy = evaluate(Namespace(model=BASE/'best.pt', dataset=PROXY, split='val', output=OUT/'proxy-val-0.01.json', confidence=.01, pt=None))
                rows = sweep(proxy['per_image'], [.35])
                original_val = evaluate(Namespace(model=BASE/'best.pt', dataset=TACO, split='val', output=OUT/'taco-val-0.35.json', confidence=.35, pt=None))
                REPORT['proxy_check'] = {'proxy_val_at_0.35': {k: rows[0][k] for k in ('precision', 'recall', 'recall_by_size')},
                                         'taco_val_at_0.35': {k: original_val[k] for k in ('precision', 'recall')}}
                print(json.dumps(REPORT['proxy_check'], indent=1))
            else:
                print('litter-v2 not in this bundle yet; rerun after it is packaged.')
        '''),
        ('markdown', '## 10. Save the report and check Gate 1'),
        ('code', '''
            gate = {'integrity': True, 'regression_tests_passed': REPORT['regression_tests_passed'],
                    'official_result_reproduced': REPORT['pune_official_reproduced'],
                    'training_logs_read': 'training_curves' in REPORT,
                    'parity_explained': 'Review parity_at_0.35 failures: pixel and confidence differences are recorded.',
                    'misses_categorised': sum(REPORT['miss_categories']['by_category'].values()) == 54}
            REPORT['gate_1'] = gate
            (OUT/'diagnosis-report.json').write_text(json.dumps(REPORT, indent=2, default=str))
            print(json.dumps(gate, indent=1))
            print('Saved', OUT/'diagnosis-report.json', '— download the whole diagnosis-baseline-v1 folder for review.')
        '''),
    ]


def experiment_cells(setup) -> list[tuple[str, str]]:
    return [
        ('markdown', '''
            # GeoCleanAI — B. Experiments, selection and one locked Pune evaluation

            Rules:
            - At most three experiments (configs in `training/experiments/`). Run **E1**, then **E3** (tiled 1024), then **E2** (P2 head) only if its condition is met.
            - Every labelled Pune view trains (no Pune validation split, decided 26 September 2026), so decisions use the street-like TACO proxy (`litter-v2` / `litter-v2-r1024` val). A Pune validation folder is still used if one is ever added to the bundle. The Pune test is not opened until a candidate and its threshold are locked in `selection-record.json`.
            - Keep 0.35 results for comparison with the baseline. A different threshold must come from validation and be locked before the Pune run.
            - E3 (1024px views cut into 640px tiles) uses the same tiling code as the app (`backend/tiling.py`) and is scored on 1024px copies of the Pune views with unchanged labels.
        '''),
        *setup,
        ('code', '''
            from argparse import Namespace
            from datetime import datetime, timezone
            from training.evaluate import evaluate, sweep, choose_threshold
            BASE, PUNE = V2/'baseline-v1/files', V1/'pune-test'
            EXPERIMENT = 'e1-domain-matched'  # then 'e3-tiled-1024', then 'e2-p2-head' only if justified
            config = json.loads((V2/f'code/training/experiments/{EXPERIMENT}.json').read_text())
            DATA = V2/'datasets'/config['data']
            assert DATA.is_file(), f'{config["data"]} is not in this bundle.'
            diagnosis = DRIVE/'diagnosis-baseline-v1/diagnosis-report.json'
            if diagnosis.is_file() and 'training_curves' in json.loads(diagnosis.read_text()):
                config['epochs'] = json.loads(diagnosis.read_text())['training_curves']['next_epoch_cap']
            else:
                print('Diagnosis logs not found: keeping the epoch cap at', config['epochs'])
            OUTPUT = DRIVE/'experiments'/f"{EXPERIMENT}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            OUTPUT.mkdir(parents=True)
            (OUTPUT/'effective-config.json').write_text(json.dumps(config, indent=2))
            (OUTPUT/'pip-freeze.txt').write_text(FREEZE)
            print(EXPERIMENT, '->', OUTPUT); print(json.dumps(config, indent=1))
        '''),
        ('markdown', '## Train\nCheckpoints save to Drive every 5 epochs. After epoch 1, check the time per epoch: if the projected total is over ~3 hours, stop and rebuild with fewer derived tiles (not fewer epochs).'),
        ('code', '''
            from training.train import train
            train(Namespace(data=DATA, config=OUTPUT/'effective-config.json', output=OUTPUT, version=config['version'], epochs=None, resume=None))
        '''),
        ('markdown', '### Resume after a disconnect (only when needed)\nRun the setup cells, set `OUTPUT` to the interrupted folder, then run this cell instead of the one above.'),
        ('code', '''
            # OUTPUT = DRIVE/'experiments'/'REPLACE-WITH-INTERRUPTED-FOLDER'
            # run = next((OUTPUT/'runs').iterdir())
            # train(Namespace(data=DATA, config=OUTPUT/'effective-config.json', output=OUTPUT, version=config['version'], epochs=None, resume=run/'weights/last.pt'))
        '''),
        ('markdown', '## Checkpoint sweep on the validation proxy (baseline included for comparison)'),
        ('code', '''
            THRESHOLDS = [round(t, 2) for t in [x / 100 for x in range(5, 96, 5)]]
            TILING = config.get('tiling')
            # E3 is scored exactly as it would run in the app: 1024px views cut into 640px tiles.
            PUNE_VAL = V2/('datasets/pune-val-v1-r1024' if TILING else 'datasets/pune-val-v1')
            small = lambda row: sum(row['recall_by_size'][b]['matched'] for b in ('<8', '8-16')) / max(1, sum(row['recall_by_size'][b]['labels'] for b in ('<8', '8-16')))
            run_dir = next((OUTPUT/'runs').iterdir())
            checkpoints = sorted((run_dir/'weights').glob('*.pt')) + [BASE/'best.pt']
            table = []
            for checkpoint in checkpoints:
                label = 'baseline-v1' if checkpoint == BASE/'best.pt' else f'{EXPERIMENT}-{checkpoint.stem}'
                result = evaluate(Namespace(model=checkpoint, dataset=DATA.parent, split='val', output=OUTPUT/'sweeps'/f'{label}.json', confidence=.01, pt=None))
                rows = sweep(result['per_image'], THRESHOLDS)
                chosen = choose_threshold(rows)
                at_035 = next(r for r in rows if r['threshold'] == .35)
                entry = {'checkpoint': str(checkpoint), 'sha256': file_hash(checkpoint), 'experiment': EXPERIMENT, 'tiling': TILING,
                         'proxy_chosen': chosen, 'chosen': chosen,
                         'score_recall': chosen['recall'], 'small_recall_lt16': small(chosen), 'score_source': 'TACO proxy',
                         'at_0.35': {k: at_035[k] for k in ('precision', 'recall')}}
                if PUNE_VAL.is_dir():
                    pune = evaluate(Namespace(model=checkpoint, dataset=PUNE_VAL, output=OUTPUT/'sweeps'/f'{label}-pune-val.json', confidence=.01, pt=None, tiled=bool(TILING)))
                    pune_rows = sweep(pune['per_image'], THRESHOLDS)
                    regions = pune_rows[0]['true_positives'] + pune_rows[0]['false_negatives']
                    if regions >= 20:
                        entry['chosen'] = choose_threshold(pune_rows); entry['score_source'] = 'Pune validation'
                    else:
                        entry['chosen'] = next(r for r in pune_rows if r['threshold'] == chosen['threshold']) | {'rule': 'threshold from TACO proxy; Pune validation has < 20 regions'}
                        entry['score_source'] = 'Pune validation at proxy threshold'
                    entry.update(score_recall=entry['chosen']['recall'], small_recall_lt16=small(entry['chosen']), pune_val_regions=regions,
                                 pune_val_at_035={k: next(r for r in pune_rows if r['threshold'] == .35)[k] for k in ('precision', 'recall')})
                table.append(entry)
            table.sort(key=lambda r: (r['score_recall'] or 0, r['small_recall_lt16']), reverse=True)
            (OUTPUT/'checkpoint-sweep.json').write_text(json.dumps(table, indent=2))
            for row in table:
                print(f"{Path(row['checkpoint']).name:14} [{row['score_source']}] thr={row['chosen']['threshold']:.2f} P={row['chosen']['precision']} R={row['score_recall']} small={row['small_recall_lt16']:.3f} | proxy at 0.35: {row['at_0.35']}")
        '''),
        ('markdown', '''
            ## Stop conditions (validation only)
            - **E1:** if the best checkpoint's proxy recall (at the chosen threshold) beats the baseline by < 5 points → stop: the data, not the recipe, is the limit.
            - **E3:** run if E1's small-object recall (<16 px) stays low while ≥32px recall is good. Drop it if its small-object gain is < 5 points or the latency gate fails.
            - **E2 (P2 head):** run only after E3, if E3's <16 px recall is still well below its ≥32 px recall. Drop it if its gain over E3 is < 3 points.
        '''),
        ('code', '''
            baseline = next(r for r in table if r['checkpoint'].endswith('baseline-v1/files/best.pt'))
            best = next(r for r in table if r is not baseline)
            gain = (best['score_recall'] or 0) - (baseline['score_recall'] or 0)
            print(f"{EXPERIMENT}: proxy recall {best['score_recall']} vs baseline {baseline['score_recall']} (gain {gain*100:.1f} points)")
            print('Stop condition met (gain < 5 points):', gain < .05)
            (OUTPUT/'stop-check.json').write_text(json.dumps({'experiment': EXPERIMENT, 'gain': gain, 'best': best, 'baseline': baseline}, indent=2))
        '''),
        ('markdown', '''
            ## Select ONE candidate and lock it (after all experiments you intend to run)
            Pick the candidate with the highest proxy recall at its chosen threshold (tie-break: small-object recall). The record is hashed before the Pune test is opened.
        '''),
        ('code', '''
            sweeps = [json.loads(p.read_text()) for p in (DRIVE/'experiments').glob('*/checkpoint-sweep.json')]
            candidates = [row for rows in sweeps for row in rows if 'baseline-v1' not in row['checkpoint']]
            selected = max(candidates, key=lambda r: (r['score_recall'] or 0, r['small_recall_lt16']))
            SELECT_DIR = DRIVE/'selection'; SELECT_DIR.mkdir(exist_ok=True)
            assert not (SELECT_DIR/'selection-record.json').exists(), 'A selection is already locked; do not reselect after seeing Pune results.'
            record = {'selected_checkpoint': selected['checkpoint'], 'checkpoint_sha256': selected['sha256'],
                      'threshold': selected['chosen']['threshold'], 'threshold_rule': selected['chosen']['rule'], 'score_source': selected['score_source'],
                      'experiment': selected['experiment'], 'tiling': selected['tiling'],
                      'validation': {k: selected['chosen'][k] for k in ('precision', 'recall', 'recall_by_size')},
                      'compared': [{'checkpoint': r['checkpoint'], 'experiment': r['experiment'], 'score_recall': r['score_recall']} for r in candidates],
                      'locked_at': datetime.now(timezone.utc).isoformat()}
            (SELECT_DIR/'selection-record.json').write_text(json.dumps(record, indent=2))
            (SELECT_DIR/'selection-record.sha256').write_text(file_hash(SELECT_DIR/'selection-record.json'))
            print(json.dumps(record, indent=1))
        '''),
        ('markdown', '## Export the selected checkpoint and check ONNX against PyTorch on validation data'),
        ('code', '''
            import onnxruntime as ort
            from ultralytics import YOLO
            record = json.loads((SELECT_DIR/'selection-record.json').read_text())
            checkpoint = Path(record['selected_checkpoint'])
            assert file_hash(checkpoint) == record['checkpoint_sha256']
            exported = Path(YOLO(str(checkpoint)).export(format='onnx', imgsz=640, dynamic=False, half=False, batch=1, simplify=False, opset=17, nms=False, max_det=100, device='cpu'))
            session = ort.InferenceSession(str(exported), providers=['CPUExecutionProvider'])
            assert session.get_inputs()[0].shape == [1, 3, 640, 640] and session.get_outputs()[0].shape[-1] == 6
            FINAL = SELECT_DIR/'model'; FINAL.mkdir(exist_ok=True)
            shutil.copy2(exported, FINAL/'litter.onnx'); shutil.copy2(checkpoint, FINAL/'best.pt')
            parity = evaluate(Namespace(model=FINAL/'litter.onnx', dataset=DATA.parent, split='val', output=FINAL/'proxy-val-parity.json', confidence=record['threshold'], pt=FINAL/'best.pt'))
            print('ONNX/PyTorch parity on validation:', parity['parity_passed'], '| rule:', parity['parity_rule'])
            print('Old strict rule (unclipped, IoU only):', parity['parity_passed_strict_unclipped'])
            failures = [p for p in parity['parity'] if not p['passed']]
            print(len(failures), 'failures; worst corner difference', max((p['max_corner_px'] for p in parity['parity']), default=0), 'px')
            import ultralytics
            deploy = {'version': f"pune-litter-{record['experiment']}", 'trained': True, 'classes': ['litter'], 'format': 'yolo26-end2end',
                      'input_size': 640, 'confidence_threshold': record['threshold'], 'sha256': file_hash(FINAL/'litter.onnx'),
                      'selection_record_sha256': file_hash(SELECT_DIR/'selection-record.json'),
                      'versions': {'ultralytics': ultralytics.__version__, 'onnxruntime': ort.__version__}}
            if record['tiling']:
                deploy['tiling'] = record['tiling']  # The app renders views at face_size and detects on 640px tiles.
            (FINAL/'manifest.json').write_text(json.dumps(deploy, indent=2))
            print('Deploy manifest:', json.dumps(deploy, indent=1))
        '''),
        ('markdown', '''
            ## Locked Pune evaluation — run once
            Refuses to run unless the selection record is unchanged. Reports the locked threshold **and** 0.35, the baseline on identical inputs, size buckets, clean-view false alarms and intervals.
        '''),
        ('code', '''
            assert file_hash(SELECT_DIR/'selection-record.json') == (SELECT_DIR/'selection-record.sha256').read_text().strip(), 'Selection record changed after locking.'
            assert not (SELECT_DIR/'pune-final.json').exists(), 'The locked Pune evaluation has already been run.'
            record = json.loads((SELECT_DIR/'selection-record.json').read_text())
            runs = {}
            # A tiled candidate is scored on the same locked views re-rendered at 1024px (labels unchanged).
            CANDIDATE_TEST = V2/'pune-test-v1-r1024' if record['tiling'] else PUNE
            for label, model, threshold, dataset, tiled in (('candidate@locked', FINAL/'litter.onnx', record['threshold'], CANDIDATE_TEST, bool(record['tiling'])),
                                                            ('candidate@0.35', FINAL/'litter.onnx', .35, CANDIDATE_TEST, bool(record['tiling'])),
                                                            ('baseline@0.35', BASE/'litter.onnx', .35, PUNE, False)):
                result = evaluate(Namespace(model=model, dataset=dataset, output=SELECT_DIR/f'pune-{label.replace("@", "-")}.json', confidence=threshold, pt=None, tiled=tiled))
                runs[label] = {k: result[k] for k in ('true_positives', 'false_positives', 'false_negatives', 'precision', 'recall',
                                                      'recall_by_size', 'clean_views', 'clean_view_false_positives', 'clean_views_with_false_alarm',
                                                      'precision_wilson_95', 'recall_wilson_95', 'panorama_bootstrap', 'median_inference_ms', 'mode')}
            locked = runs['candidate@locked']
            final = {'selection_record_sha256': file_hash(SELECT_DIR/'selection-record.json'), 'runs': runs,
                     'target_met': all(locked[k] is not None and locked[k] >= .85 for k in ('precision', 'recall')),
                     'disclosure': 'Pune test views were inspected during baseline diagnosis; not used for selection.',
                     'limitations': ['48 views from 11 panoramas and 8 sequences; views are correlated.', 'Does not establish citywide accuracy.',
                                     '24 skipped views are excluded.', 'Panorama marker alignment and hosted six-view latency are validated separately.']}
            (SELECT_DIR/'pune-final.json').write_text(json.dumps(final, indent=2))
            for label, run in runs.items():
                print(f"{label:17} TP={run['true_positives']} FP={run['false_positives']} FN={run['false_negatives']} P={run['precision']} R={run['recall']}")
            print('Both >= 85%:', final['target_met'])
        '''),
        ('markdown', '## Download the results'),
        ('code', '''
            result_zip = SELECT_DIR/'geocleanai-v2-results.zip'
            with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as bundle:
                for path in SELECT_DIR.rglob('*'):
                    if path.is_file() and path != result_zip:
                        bundle.write(path, path.relative_to(SELECT_DIR))
                for path in (DRIVE/'experiments').rglob('*'):
                    if path.is_file() and (path.suffix in ('.json', '.csv', '.png', '.yaml', '.txt')):
                        bundle.write(path, Path('experiments')/path.relative_to(DRIVE/'experiments'))
            from google.colab import files
            files.download(str(result_zip))
        '''),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/colab/geocleanai-colab-v2.zip')
    args = parser.parse_args()
    v1_sha = (ROOT/'artifacts/colab/geocleanai-colab-data-v1.zip.sha256').read_text().split()[0]
    v2_sha = Path(str(args.bundle) + '.sha256').read_text().split()[0]
    setup = setup_cells(v1_sha, args.bundle.name, v2_sha)
    for name, cells in (('GeoCleanAI_A_Diagnosis.ipynb', diagnosis_cells(setup)), ('GeoCleanAI_B_Experiments.ipynb', experiment_cells(setup))):
        (args.bundle.parent/name).write_text(json.dumps(notebook(cells, name), indent=2, ensure_ascii=False), encoding='utf-8')
        print('Wrote', args.bundle.parent/name, f'({len(cells)} cells; all code cells compile)')


if __name__ == '__main__':
    main()
