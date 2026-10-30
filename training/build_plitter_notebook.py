"""Build the pLitter export notebook: inspect the pretrained street model and export it to ONNX.

No training. The notebook downloads pLitterStreet_YOLOv5l.pt from gicait/pLitter, records its
class list and settings, exports fixed 640-pixel ONNX files (FP32, FP16, INT8), and checks each
against PyTorch on Pune views with the same letterbox and parity rule the app uses.
"""
import argparse
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_URL = 'https://github.com/gicait/pLitter/releases/download/v0.0.0-street/pLitterStreet_YOLOv5l.pt'
# The YOLOv5 copy pLitter loads its weights with (its Yolov5_StrongSORT_OSNet submodule, Jan 2023).
YOLOV5_COMMIT = '43623729cc634d690cece1f1d015e4d59e0b9d98'
REPO_URL = 'https://github.com/siddhihingne12/GeoCleanAI_updated'
CHECK_VIEWS = 20


def notebook(cells: list[tuple[str, str]], name: str) -> dict:
    built = []
    for index, (kind, source) in enumerate(cells):
        cell = {'cell_type': kind, 'id': f'geoclean-plitter-{index}', 'metadata': {},
                'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code':
            compile(''.join(cell['source']), f'{name}-cell-{index}', 'exec')
            cell.update(execution_count=None, outputs=[])
        built.append(cell)
    return {'nbformat': 4, 'nbformat_minor': 5, 'cells': built, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': name}}}


def cells() -> list[tuple[str, str]]:
    return [
        ('markdown', '''
            # GeoCleanAI: pLitter model inspection and ONNX export

            Plan Phase 1 (tasks 1.1–1.2). **No training.** A CPU runtime is enough.

            1. Downloads `pLitterStreet_YOLOv5l.pt` from [gicait/pLitter](https://github.com/gicait/pLitter) (release `v0.0.0-street`).
            2. Records its class list, training image size and checksum.
            3. Exports fixed 640-pixel ONNX files: FP32, FP16 and INT8.
            4. Checks each file against PyTorch on Pune views, and times it on one CPU thread (like Vercel).

            Everything is saved to **My Drive/GeoCleanAI/plitter-v0**. Send the printed summary (or `model-info.json`) back to Claude.

            The weights have no stated licence. Do not redistribute them until the authors reply (plan task 0.5).
        '''),
        ('code', f'''
            from google.colab import drive
            drive.mount('/content/drive')
            import hashlib, json, os, subprocess, sys, time
            from pathlib import Path
            OUT = Path('/content/drive/MyDrive/GeoCleanAI/plitter-v0')
            OUT.mkdir(parents=True, exist_ok=True)
            WORK = Path('/content')

            def run(*cmd, cwd=None):
                print('$', ' '.join(map(str, cmd)))
                subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)

            def file_hash(path):
                with Path(path).open('rb') as stream:
                    return hashlib.file_digest(stream, 'sha256').hexdigest()

            # YOLOv5 at the exact commit pLitter uses.
            YOLOV5 = WORK/'yolov5'
            if not (YOLOV5/'.git').is_dir():
                run('git', 'clone', '--quiet', 'https://github.com/ultralytics/yolov5', YOLOV5)
            run('git', 'checkout', '--quiet', '{YOLOV5_COMMIT}', cwd=YOLOV5)

            # GeoCleanAI code (letterbox and parity rule) plus the Pune views, without the rest of data/.
            GEO = WORK/'geocleanai'
            if not (GEO/'.git').is_dir():
                run('git', 'clone', '--quiet', '--depth', '1', '--filter=blob:none', '--sparse', '{REPO_URL}', GEO)
            run('git', 'sparse-checkout', 'set', '--no-cone', '/backend/', '/training/', '/data/pune-label-v2/images/', cwd=GEO)
            GEO_COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=GEO, text=True).strip()
            print('GeoCleanAI commit', GEO_COMMIT)
        '''),
        ('markdown', 'Install the ONNX packages at the versions the app uses. Colab already has PyTorch and OpenCV.'),
        ('code', '''
            run(sys.executable, '-m', 'pip', 'install', '--quiet', 'onnx==1.23.0', 'onnxruntime==1.30.0')
            # Newer PyTorch refuses to unpickle full YOLOv5 checkpoints by default. These weights come from
            # a known release, but they still run pickle code, so only load them here in Colab.
            os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
            import numpy as np, onnx, onnxruntime as ort, torch
            from PIL import Image
            sys.path.insert(0, str(YOLOV5))
            sys.path.insert(1, str(GEO))
            from backend.geometry import letterbox
            from training.evaluate import parity
            print('torch', torch.__version__, '| onnx', onnx.__version__, '| onnxruntime', ort.__version__)
        '''),
        ('markdown', '## 1. Download and inspect the weights'),
        ('code', f'''
            PT = OUT/'pLitterStreet_YOLOv5l.pt'
            if not PT.is_file():
                torch.hub.download_url_to_file('{WEIGHTS_URL}', str(PT), progress=True)
            PT_SHA = file_hash(PT)

            ckpt = torch.load(PT, map_location='cpu', weights_only=False)
            net = (ckpt.get('ema') or ckpt['model'])
            names = net.names if isinstance(net.names, dict) else dict(enumerate(net.names))
            opt = ckpt.get('opt') or {{}}
            if not isinstance(opt, dict):
                opt = vars(opt)
            INFO = {{
                'source': '{WEIGHTS_URL}',
                'pt_sha256': PT_SHA, 'pt_bytes': PT.stat().st_size,
                'class_names': {{int(k): v for k, v in names.items()}}, 'num_classes': int(net.nc),
                'stride': [int(s) for s in net.stride.tolist()],
                'train_imgsz': opt.get('imgsz') or opt.get('img_size'), 'train_data': opt.get('data'),
                'epoch': ckpt.get('epoch'), 'best_fitness': None if ckpt.get('best_fitness') is None else float(ckpt['best_fitness']),
                'checkpoint_date': ckpt.get('date'), 'checkpoint_git': ckpt.get('git'),
                'yolov5_commit': '{YOLOV5_COMMIT}', 'geocleanai_commit': GEO_COMMIT,
                'versions': {{'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__}},
            }}
            del ckpt, net
            print(json.dumps(INFO, indent=2, default=str))
        '''),
        ('markdown', '''
            **Class map (plan task 1.3):** every class becomes `litter` except trash bins, which are dropped.
            Check the printed map. If it is wrong, edit `CLASS_MAP` and run on.
        '''),
        ('code', '''
            CLASS_MAP = {i: (None if 'bin' in name.lower() else 'litter') for i, name in INFO['class_names'].items()}
            INFO['class_map'] = CLASS_MAP
            for i, target in CLASS_MAP.items():
                print(f'{i}: {INFO["class_names"][i]!r:>14} -> {target or "dropped"}')
        '''),
        ('markdown', '## 2. Export ONNX (FP32, FP16, INT8)'),
        ('code', '''
            from models.experimental import attempt_load
            from models.yolo import Detect
            model = attempt_load(str(PT), device='cpu', inplace=True, fuse=True).float().eval()
            for module in model.modules():
                if isinstance(module, Detect):
                    module.inplace, module.dynamic, module.export = True, False, True
            dummy = torch.zeros(1, 3, 640, 640)
            with torch.no_grad():
                print('PyTorch output shape:', tuple(model(dummy)[0].shape))  # export mode returns a 1-tuple

            FP32 = OUT/'litter-fp32.onnx'
            kwargs = dict(opset_version=17, do_constant_folding=True, input_names=['images'], output_names=['output0'])
            with torch.no_grad():
                try:
                    torch.onnx.export(model, dummy, str(FP32), dynamo=False, **kwargs)
                except TypeError:  # older PyTorch has no dynamo argument
                    torch.onnx.export(model, dummy, str(FP32), **kwargs)
            onnx.checker.check_model(onnx.load(str(FP32)))

            from onnxruntime.transformers.float16 import convert_float_to_float16
            FP16 = OUT/'litter-fp16.onnx'
            onnx.save(convert_float_to_float16(onnx.load(str(FP32)), keep_io_types=True), str(FP16))

            from onnxruntime.quantization import QuantType, quantize_dynamic
            INT8 = OUT/'litter-int8.onnx'
            try:
                quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QUInt8)
            except Exception as error:
                print('INT8 quantisation failed:', error)
                INT8.unlink(missing_ok=True)

            VARIANTS = {p.stem.split('-')[1]: p for p in (FP32, FP16, INT8) if p.is_file()}
            for name, path in VARIANTS.items():
                print(f'{name}: {path.stat().st_size / 1e6:.1f} MB')
        '''),
        ('markdown', '''
            ## 3. Check against PyTorch on Pune views

            The decoder below is what the backend will use (plan task 1.4): objectness × class score, the class
            map, then NMS. PyTorch and each ONNX file get the identical letterboxed input from `backend/geometry.py`,
            and are compared with the app's parity rule from `training/evaluate.py`.
            FP32 must pass. FP16 and INT8 are expected to differ slightly; their numbers show how much.
        '''),
        ('code', f'''
            CONF, NMS_IOU = .25, .45

            def decode_yolov5(raw, conf=CONF, iou=NMS_IOU):
                """Raw [1, N, 5 + classes] output to [x1, y1, x2, y2, confidence, 0] rows in letterboxed pixels."""
                raw = np.asarray(raw, dtype=np.float32)[0]
                keep = np.array([CLASS_MAP[i] == 'litter' for i in range(raw.shape[1] - 5)])
                scores = raw[:, 4:5] * raw[:, 5:]
                scores[:, ~keep] = 0
                best = scores.max(axis=1)
                raw, best = raw[best >= conf], best[best >= conf]
                boxes = np.stack([raw[:, 0] - raw[:, 2] / 2, raw[:, 1] - raw[:, 3] / 2,
                                  raw[:, 0] + raw[:, 2] / 2, raw[:, 1] + raw[:, 3] / 2], axis=1)
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                order, kept = list(np.argsort(-best)), []
                while order:
                    i = order.pop(0)
                    kept.append(i)
                    if not order:
                        break
                    rest = np.array(order)
                    x1 = np.maximum(boxes[i, 0], boxes[rest, 0]); y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
                    x2 = np.minimum(boxes[i, 2], boxes[rest, 2]); y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
                    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                    overlap = inter / (area[i] + area[rest] - inter + 1e-9)
                    order = [j for j, o in zip(order, overlap) if o < iou]
                return [[*map(float, boxes[i]), float(best[i]), 0.] for i in kept[:300]]

            views = sorted(p for p in (GEO/'data/pune-label-v2/images').glob('*.jpg') if not p.stem.endswith('_up'))
            assert views, 'No Pune views found in the sparse checkout.'
            CHECK = views[::max(1, len(views) // {CHECK_VIEWS})][:{CHECK_VIEWS}]

            def prepared(path):
                image = Image.open(path).convert('RGB')
                tensor, scale, left, top = letterbox(image)
                bounds = (left, top, left + image.width * scale, top + image.height * scale)
                return image, tensor.astype(np.float32), bounds

            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            sessions = {{name: ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                        for name, path in VARIANTS.items()}}
            for name, session in sessions.items():
                print(name, 'input', session.get_inputs()[0].shape, 'output', session.get_outputs()[0].shape)

            results = {{name: [] for name in VARIANTS}}
            for path in CHECK:
                image, tensor, bounds = prepared(path)
                with torch.no_grad():
                    pt_rows = decode_yolov5(model(torch.from_numpy(tensor))[0].numpy())
                for name, session in sessions.items():
                    ort_rows = decode_yolov5(session.run(None, {{'images': tensor}})[0])
                    report = parity(pt_rows, ort_rows, bounds=bounds)
                    results[name].append({{'view': path.name, 'pt_count': len(pt_rows), 'onnx_count': len(ort_rows),
                                          'passed': report['passed'], 'max_confidence_diff': report['max_confidence_diff']}})

            INFO['parity'] = {{}}
            for name, rows in results.items():
                INFO['parity'][name] = {{'views': len(rows), 'passed': sum(r['passed'] for r in rows),
                                        'max_confidence_diff': max(r['max_confidence_diff'] for r in rows),
                                        'pt_detections': sum(r['pt_count'] for r in rows), 'onnx_detections': sum(r['onnx_count'] for r in rows)}}
                print(name, INFO['parity'][name])
            assert INFO['parity']['fp32']['passed'] == len(CHECK), 'FP32 ONNX does not match PyTorch. Stop here and send the output to Claude.'
        '''),
        ('markdown', '## 4. CPU speed (one thread, like Vercel)'),
        ('code', '''
            _, tensor, _ = prepared(CHECK[0])
            INFO['cpu_ms_per_view'] = {}
            for name, session in sessions.items():
                session.run(None, {'images': tensor})
                start = time.perf_counter()
                for _ in range(5):
                    session.run(None, {'images': tensor})
                INFO['cpu_ms_per_view'][name] = round((time.perf_counter() - start) / 5 * 1000)
            INFO['onnx'] = {name: {'sha256': file_hash(path), 'bytes': path.stat().st_size} for name, path in VARIANTS.items()}
            for name, ms in INFO['cpu_ms_per_view'].items():
                print(f'{name}: {ms} ms per 640 px view -> about {ms * 24 / 1000:.0f} s for 24 tiled views, {ms * 6 / 1000:.0f} s for 6 faces')
        '''),
        ('markdown', '## 5. Look at the detections\nBoxes from the FP32 model on six Pune views. Judge by eye whether it finds litter and what it wrongly marks.'),
        ('code', '''
            from PIL import ImageDraw
            from IPython.display import display
            PREVIEW = OUT/'preview'
            PREVIEW.mkdir(exist_ok=True)
            for path in CHECK[:6]:
                image, tensor, (left, top, _, _) = prepared(path)
                scale = min(640 / image.width, 640 / image.height)
                rows = decode_yolov5(sessions['fp32'].run(None, {'images': tensor})[0])
                draw = ImageDraw.Draw(image)
                for x1, y1, x2, y2, confidence, _ in rows:
                    box = [(x1 - left) / scale, (y1 - top) / scale, (x2 - left) / scale, (y2 - top) / scale]
                    draw.rectangle(box, outline=(255, 40, 40), width=3)
                    draw.text((box[0] + 3, box[1] + 3), f'{confidence:.2f}', fill=(255, 40, 40))
                image.save(PREVIEW/path.name)
                print(path.name, len(rows), 'detections')
                display(image.resize((image.width // 2, image.height // 2)))
        '''),
        ('markdown', '## 6. Save the summary\nSend the printed JSON to Claude. It decides the backend decoder, the manifest and which ONNX file to deploy.'),
        ('code', '''
            (OUT/'model-info.json').write_text(json.dumps(INFO, indent=2, default=str))
            print(json.dumps({k: INFO[k] for k in ('class_names', 'class_map', 'num_classes', 'train_imgsz', 'parity', 'cpu_ms_per_view', 'onnx')}, indent=2, default=str))
            print('Saved to', OUT)
        '''),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'training/plitter_export.ipynb')
    args = parser.parse_args()
    built = cells()
    args.output.write_text(json.dumps(notebook(built, args.output.name), indent=2, ensure_ascii=False), encoding='utf-8')
    print('Wrote', args.output, f'({len(built)} cells; all code cells compile)')


if __name__ == '__main__':
    main()
