import hashlib
import json
import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image
import pytest
from backend.detector import decode, detect, manifest, load_session
from backend.errors import ServiceError
from backend.schemas import ViewSpec

VIEW = ViewSpec(origin=(0,0,1), right=(2,0,0), down=(0,-1,0), aspect=2)

def test_decode_unpads_and_filters():
    crop = Image.new('RGB', (640,320))
    output = np.array([[[160,240,480,400,.8,0], [0,0,10,10,.9,0], [0,0,640,640,.01,0], [0,0,640,640,.8,1]]], dtype=np.float32)
    results = decode(output, crop, 1, 0,160, VIEW,.35)
    assert len(results) == 1
    assert results[0]['box'] == pytest.approx([.25,.25,.75,.75])
    assert results[0]['tag'] == pytest.approx([.5,.5])

def test_real_onnx_runtime_fixture_and_checksum(monkeypatch, tmp_path):
    # Deliberately synthetic transport test, not a litter model. Never deployed.
    output = helper.make_tensor('values', TensorProto.FLOAT, [1,1,6], [160,240,480,400,.8,0])
    graph = helper.make_graph([helper.make_node('Constant', [], ['output0'], value=output)], 'test-only', [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,640,640])], [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1,1,6])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',17)]); model.ir_version = 9
    path = tmp_path / 'fixture.onnx'; onnx.save(model, path)
    info = {'version': 'SYNTHETIC-TEST-ONLY', 'trained': True, 'classes': ['litter'], 'format': 'yolo26-end2end', 'input_size': 640, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    meta = tmp_path / 'manifest.json'; meta.write_text(json.dumps(info))
    monkeypatch.setenv('MODEL_PATH', str(path)); monkeypatch.setenv('MODEL_MANIFEST_PATH', str(meta))
    assert len(detect(Image.new('RGB', (640,320)), VIEW, manifest())) == 1
    with pytest.raises(ServiceError, match='does not match'):
        load_session(str(path), '0'*64)


def test_rejected_baseline_is_never_served(monkeypatch, tmp_path):
    from backend.detector import REJECTED_MODEL_SHA256
    path = tmp_path / 'litter.onnx'; path.write_bytes(b'placeholder')
    info = {'version': 'pune-litter-baseline-v1', 'trained': True, 'classes': ['litter'], 'format': 'yolo26-end2end',
            'input_size': 640, 'sha256': 'b7cb1fc9851b674467aeeb0f43224a74f9373434c6b6e46bcaec692463ac57ae'}
    assert info['sha256'] in REJECTED_MODEL_SHA256
    meta = tmp_path / 'manifest.json'; meta.write_text(json.dumps(info))
    monkeypatch.setenv('MODEL_PATH', str(path)); monkeypatch.setenv('MODEL_MANIFEST_PATH', str(meta))
    with pytest.raises(ServiceError, match='not been installed'):
        manifest()


def test_installed_model_is_not_the_rejected_baseline():
    from pathlib import Path
    from backend.detector import REJECTED_MODEL_SHA256
    root = Path(__file__).resolve().parents[2]
    installed = root / 'models/manifest.json'
    if installed.is_file():
        assert json.loads(installed.read_text())['sha256'] not in REJECTED_MODEL_SHA256
    model = root / 'models/litter.onnx'
    if model.is_file():
        assert hashlib.sha256(model.read_bytes()).hexdigest() not in REJECTED_MODEL_SHA256


def write_manifest(tmp_path, monkeypatch, extra):
    path = tmp_path / 'm.onnx'
    if not path.exists():
        path.write_bytes(b'placeholder')
    info = {'version': 'TEST', 'trained': True, 'classes': ['litter'], 'format': 'yolo26-end2end', 'input_size': 640,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **extra}
    meta = tmp_path / 'manifest.json'; meta.write_text(json.dumps(info))
    monkeypatch.setenv('MODEL_PATH', str(path)); monkeypatch.setenv('MODEL_MANIFEST_PATH', str(meta))


@pytest.mark.parametrize('tiling, valid', [
    ({'face_size': 1024, 'tile': 640, 'stride': 384}, True),
    ({'face_size': 1024, 'tile': 512, 'stride': 384}, False),
    ({'face_size': 600, 'tile': 640, 'stride': 384}, False),
    ({'face_size': 1024, 'tile': 640, 'stride': 0}, False),
    ({'face_size': '1024', 'tile': 640, 'stride': 384}, False),
])
def test_manifest_validates_tiling(tmp_path, monkeypatch, tiling, valid):
    write_manifest(tmp_path, monkeypatch, {'tiling': tiling})
    if valid:
        assert manifest()['tiling'] == tiling
    else:
        with pytest.raises(ServiceError):
            manifest()


def test_tiled_detection_maps_tile_boxes_to_the_view(monkeypatch, tmp_path):
    # Synthetic constant model: every 640px tile reports one box at tile pixels (100,100)-(140,140).
    output = helper.make_tensor('values', TensorProto.FLOAT, [1,1,6], [100,100,140,140,.8,0])
    graph = helper.make_graph([helper.make_node('Constant', [], ['output0'], value=output)], 'test-only', [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,640,640])], [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1,1,6])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',17)]); model.ir_version = 9
    onnx.save(model, tmp_path / 'm.onnx')
    write_manifest(tmp_path, monkeypatch, {'tiling': {'face_size': 1024, 'tile': 640, 'stride': 384}})
    square = ViewSpec(origin=(0,0,1), right=(1,0,0), down=(0,-1,0), aspect=1)
    found = detect(Image.new('RGB', (1024, 1024)), square, manifest())
    # Four tiles at (0,0), (384,0), (0,384), (384,384) -> four separate objects in view pixels.
    boxes = sorted(tuple(round(v * 1024) for v in d['box']) for d in found)
    assert boxes == [(100, 100, 140, 140), (100, 484, 140, 524), (484, 100, 524, 140), (484, 484, 524, 524)]
    assert all(d['confidence'] == pytest.approx(.8) for d in found)


PLITTER_MAP = {'0': 'litter', '1': 'litter', '2': 'litter', '3': None}


def yolov5_raw(*rows):
    """Raw pLitter-shaped rows: cx, cy, w, h, objectness, Plastic, Pile, Face mask, Trash bin."""
    return np.array([rows], dtype=np.float32)


def test_yolov5_rows_scores_maps_classes_and_suppresses_overlaps():
    from backend.detector import yolov5_rows
    raw = yolov5_raw([100,100,20,20, .9, .9,0,0,0],    # plastic, .81
                     [102,101,20,20, .8, .9,0,0,0],    # same object, lower score -> suppressed
                     [300,300,40,40, .9, 0,0,0,.95],   # trash bin -> not litter
                     [400,200,10,10, .8, 0,.6,0,0],    # pile, .48
                     [500,500,10,10, .5, .5,0,0,0],    # .25, below threshold
                     [600,600,10,10, np.nan, 1,0,0,0])  # non-finite -> ignored
    rows = yolov5_rows(raw, PLITTER_MAP, .35)
    assert rows.shape == (1, 2, 6)
    assert rows[0, :, :4].tolist() == [[90, 90, 110, 110], [395, 195, 405, 205]]
    assert rows[0, :, 4] == pytest.approx([.81, .48])
    assert (rows[0, :, 5] == 0).all()


def test_yolov5_rows_rejects_wrong_class_count():
    from backend.detector import yolov5_rows
    with pytest.raises(ServiceError, match='unsupported output shape'):
        yolov5_rows(np.zeros((1, 3, 8), np.float32), PLITTER_MAP, .35)


@pytest.mark.parametrize('class_map, valid', [
    (PLITTER_MAP, True),
    ({'0': 'litter'}, True),
    ({'0': None, '1': None}, False),        # nothing maps to litter
    ({'0': 'litter', '2': 'litter'}, False),  # missing index 1
    ({'0': 'bottle'}, False),
    (None, False),
])
def test_manifest_validates_yolov5_class_map(tmp_path, monkeypatch, class_map, valid):
    write_manifest(tmp_path, monkeypatch, {'format': 'yolov5', 'class_map': class_map})
    if valid:
        assert manifest()['class_map'] == class_map
    else:
        with pytest.raises(ServiceError):
            manifest()


def test_yolov5_onnx_runtime_fixture(monkeypatch, tmp_path):
    # Synthetic constant model with pLitter's raw output shape [1, N, 9]. Never deployed.
    raw = yolov5_raw([320,320,320,160, .9, .9,0,0,0], [322,321,320,160, .9, .8,0,0,0], [100,100,40,40, .9, 0,0,0,.9])
    output = helper.make_tensor('values', TensorProto.FLOAT, list(raw.shape), raw.flatten().tolist())
    graph = helper.make_graph([helper.make_node('Constant', [], ['output0'], value=output)], 'test-only', [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,640,640])], [helper.make_tensor_value_info('output0', TensorProto.FLOAT, list(raw.shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',17)]); model.ir_version = 9
    onnx.save(model, tmp_path / 'm.onnx')
    write_manifest(tmp_path, monkeypatch, {'format': 'yolov5', 'class_map': PLITTER_MAP})
    found = detect(Image.new('RGB', (640, 320)), VIEW, manifest())
    # One litter object after NMS; the trash bin is dropped. Letterbox pads 160px top and bottom.
    assert len(found) == 1
    assert found[0]['box'] == pytest.approx([.25, .25, .75, .75])
    assert found[0]['confidence'] == pytest.approx(.81)


def test_yolov5_model_needs_yolov5_manifest(monkeypatch, tmp_path):
    raw = yolov5_raw([320,320,320,160, .9, .9,0,0,0])
    output = helper.make_tensor('values', TensorProto.FLOAT, list(raw.shape), raw.flatten().tolist())
    graph = helper.make_graph([helper.make_node('Constant', [], ['output0'], value=output)], 'test-only', [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,640,640])], [helper.make_tensor_value_info('output0', TensorProto.FLOAT, list(raw.shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',17)]); model.ir_version = 9
    onnx.save(model, tmp_path / 'm.onnx')
    write_manifest(tmp_path, monkeypatch, {})  # yolo26-end2end expects 6 columns
    with pytest.raises(ServiceError, match='incompatible'):
        detect(Image.new('RGB', (640, 320)), VIEW, manifest())
