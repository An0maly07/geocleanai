import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from training.evaluate import evaluate


@pytest.mark.parametrize('with_detection', [True, False])
def test_evaluation_saves_numeric_parity_results(tmp_path, monkeypatch, with_detection):
    # Model outputs are float32, including the IoUs computed from these boxes.
    rows = np.array([[160, 160, 480, 480, .9, 0]], dtype=np.float32)
    if not with_detection:
        rows = np.empty((0, 6), dtype=np.float32)
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name='images')],
        run=lambda *_: [rows[None]],
    )
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=lambda *a, **k: session))
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: rows))
    model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=SimpleNamespace(data=tensor))])
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda _: model))

    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    Image.new('RGB', (640, 640)).save(tmp_path / 'images/view.jpg')
    (tmp_path / 'labels/view.txt').write_text('0 .5 .5 .5 .5\n' if with_detection else '')
    output = tmp_path / 'results/evaluation.json'
    evaluate(Namespace(model=tmp_path / 'model.onnx', pt=tmp_path / 'best.pt',
                       dataset=tmp_path, output=output, confidence=.35))

    result = json.loads(output.read_text())
    assert result['true_positives'] == int(with_detection)
    assert result['false_positives'] == result['false_negatives'] == 0
    assert result['precision'] == (1.0 if with_detection else None)
    assert result['recall'] == (1.0 if with_detection else None)
    assert result['parity_passed'] is True
    assert result['parity'][0]['min_matched_iou'] == 1.0
    assert isinstance(result['parity'][0]['min_matched_iou'], (int, float))


from training.evaluate import match, parity, read_labels, decode_rows, short_side_px, size_bucket, summarize
from backend.geometry import letterbox


def run_fake_evaluation(tmp_path, monkeypatch, rows, labels, size=(640, 640), pt_rows=None):
    """Evaluate one synthetic image with a stub ONNX session; no real model is loaded."""
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')], run=lambda *_: [rows[None]])
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=lambda *a, **k: session))
    if pt_rows is not None:
        pt = np.asarray(pt_rows, dtype=np.float32).reshape(-1, 6)
        tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: pt))
        model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=SimpleNamespace(data=tensor))])
        monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda _: model, __version__='test'))
    (tmp_path / 'images').mkdir(exist_ok=True)
    (tmp_path / 'labels').mkdir(exist_ok=True)
    Image.new('RGB', size).save(tmp_path / 'images/view.jpg')
    (tmp_path / 'labels/view.txt').write_text(labels)
    output = tmp_path / 'evaluation.json'
    evaluate(Namespace(model=tmp_path / 'model.onnx', pt=tmp_path / 'best.pt' if pt_rows is not None else None,
                       dataset=tmp_path, output=output, confidence=.35))
    return json.loads(output.read_text())


def test_yolo_labels_convert_to_corner_boxes(tmp_path):
    label = tmp_path / 'a.txt'
    label.write_text('0 0.5 0.25 0.2 0.1\n\n')
    assert read_labels(label) == [pytest.approx([.4, .2, .6, .3])]
    label.write_text('1 0.5 0.5 0.1 0.1\n')
    with pytest.raises(ValueError, match='class 0'):
        read_labels(label)


@pytest.mark.parametrize('size', [(1000, 500), (500, 1000), (640, 640)])
def test_letterbox_boxes_map_back_to_original_pixels(size):
    width, height = size
    tensor, scale, left, top = letterbox(Image.new('RGB', size))
    original = [100, 50, 300, 400]
    # Place the box in letterboxed pixels exactly as the model would report it.
    row = [original[0]*scale+left, original[1]*scale+top, original[2]*scale+left, original[3]*scale+top, .9, 0]
    (box, confidence), = decode_rows([row], width, height, scale, left, top, .35)
    assert [box[0]*width, box[1]*height, box[2]*width, box[3]*height] == pytest.approx(original, abs=.5)
    assert confidence == pytest.approx(.9)


def test_matching_is_one_to_one_and_threshold_inclusive():
    truth = [[0, 0, .5, .5]]
    # Two predictions on one label: one TP, the other becomes a FP.
    assert [(p, t) for p, t, _ in match([[0, 0, .5, .5], [0, 0, .5, .5]], truth)] == [(0, 0)]
    # IoU exactly 0.5 counts.
    assert len(match([[0, 0, .5, .25]], truth)) == 1
    assert match([[0, 0, .5, .24]], truth) == []
    # The higher-confidence (earlier) prediction claims the label even if a later one fits better.
    assert [(p, t) for p, t, _ in match([[0, 0, .5, .3], [0, 0, .5, .5]], truth)] == [(0, 0)]


def test_empty_predictions_count_every_label_as_missed(tmp_path, monkeypatch):
    result = run_fake_evaluation(tmp_path, monkeypatch, [], '0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n')
    assert (result['true_positives'], result['false_positives'], result['false_negatives']) == (0, 0, 2)
    assert result['precision'] is None and result['recall'] == 0


def test_empty_clean_view_has_undefined_metrics_without_crashing(tmp_path, monkeypatch):
    result = run_fake_evaluation(tmp_path, monkeypatch, [], '')
    assert result['precision'] is None and result['recall'] is None
    assert result['clean_views'] == 1 and result['clean_view_false_positives'] == 0


def test_missing_label_file_is_not_treated_as_clean(tmp_path, monkeypatch):
    (tmp_path / 'images').mkdir(); (tmp_path / 'labels').mkdir()
    Image.new('RGB', (640, 640)).save(tmp_path / 'images/view.jpg')
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=lambda *a, **k: None))
    with pytest.raises(ValueError, match='Missing reviewed label'):
        evaluate(Namespace(model=tmp_path / 'm.onnx', pt=None, dataset=tmp_path, output=tmp_path / 'o.json', confidence=.35))


def test_prediction_on_clean_view_is_a_reported_false_alarm(tmp_path, monkeypatch):
    result = run_fake_evaluation(tmp_path, monkeypatch, [[10, 10, 50, 50, .8, 0], [0, 0, 5, 5, .2, 0]], '')
    assert result['false_positives'] == 1  # The 0.2 row is below the 0.35 threshold.
    assert result['clean_view_false_positives'] == 1 and result['clean_views_with_false_alarm'] == 1


def test_size_buckets_use_detector_pixels():
    assert short_side_px([0, 0, .01, .5], 640, 640) == pytest.approx(6.4)
    # A 1280px-wide image is halved at 640px input.
    assert short_side_px([0, 0, .01, .5], 1280, 640) == pytest.approx(6.4)
    assert [size_bucket(v) for v in (0, 7.99, 8, 16, 31.9, 32, 500)] == ['<8', '<8', '8-16', '16-32', '16-32', '>=32', '>=32']
    record = {'tp': 1, 'fp': 0, 'fn': 1, 'matches': [{'truth': 1}],
              'truths': [{'size_bucket': '<8'}, {'size_bucket': '>=32'}]}
    buckets = summarize([record])['recall_by_size']
    assert buckets['<8'] == {'labels': 1, 'matched': 0, 'recall': 0.0}
    assert buckets['>=32']['recall'] == 1.0 and buckets['8-16']['recall'] is None


def test_report_json_accepts_numpy_values(tmp_path, monkeypatch):
    result = run_fake_evaluation(tmp_path, monkeypatch, [[160, 160, 480, 480, .9, 0]], '0 .5 .5 .5 .5\n',
                                 pt_rows=[[160, 160, 480, 480, .9, 0]])
    assert isinstance(result['per_image'][0]['predictions'][0]['confidence'], float)
    assert isinstance(result['per_image'][0]['matches'][0]['iou'], float)
    assert result['parity'][0]['pairs'][0]['confidence_diff'] == 0


def test_parity_rule_clips_to_image_and_allows_one_pixel():
    box = [100, 100, 110, 110, .9, 0]
    # Two PyTorch boxes cannot both pair with one ONNX box.
    report = parity(np.array([box, box]), np.array([box]))
    assert not report['passed'] and report['unpaired_pt'] == 1
    # One pixel on a 10px box: IoU < 0.99 but within 1 px -> passes; the strict rule is still recorded.
    shifted = parity(np.array([box]), np.array([[101, 100, 111, 110, .88, 0]]))
    assert shifted['passed'] and not shifted['passed_strict_unclipped']
    assert shifted['max_corner_px'] == pytest.approx(1) and shifted['max_confidence_diff'] == pytest.approx(.02)
    assert not parity(np.array([box]), np.array([[102, 100, 112, 110, .9, 0]]))['passed']
    # PyTorch clipped at the image edge; raw ONNX runs past it. Equal once both are clipped.
    edge = parity(np.array([[300, 0, 640, 330, .5, 0]]), np.array([[300, -20, 690, 330, .5, 0]]), bounds=(0, 0, 640, 640))
    assert edge['passed'] and not edge['passed_strict_unclipped']
    assert parity(np.empty((0, 6)), np.empty((0, 6)))['passed']


from training.evaluate import sweep, choose_threshold


def sweep_record(predictions, truths):
    return {'predictions': [{'box': b, 'confidence': c} for b, c in predictions],
            'truths': [{'box': b, 'size_bucket': '>=32'} for b in truths]}


def test_threshold_sweep_rescores_saved_predictions():
    truth = [0, 0, .5, .5]
    records = [sweep_record([(truth, .9), ([.6, .6, .9, .9], .5)], [truth]), sweep_record([([.1, .1, .2, .2], .4)], [])]
    rows = {r['threshold']: r for r in sweep(records, [.3, .45, .6, .95])}
    assert (rows[.3]['true_positives'], rows[.3]['false_positives']) == (1, 2)
    assert rows[.3]['clean_view_false_positives'] == 1
    assert (rows[.6]['precision'], rows[.6]['recall']) == (1.0, 1.0)
    assert rows[.95]['precision'] is None and rows[.95]['recall'] == 0


def test_threshold_choice_prefers_lowest_passing_then_max_f1():
    rows = [{'threshold': .2, 'precision': .5, 'recall': .9}, {'threshold': .4, 'precision': .86, 'recall': .6},
            {'threshold': .6, 'precision': .95, 'recall': .3}]
    assert choose_threshold(rows)['threshold'] == .4
    chosen = choose_threshold(rows, min_precision=.99)
    assert chosen['threshold'] == .4 and 'maximum F1' in chosen['rule']  # F1 .707 beats .643


from training.evaluate import wilson, panorama_bootstrap


def test_wilson_interval_matches_known_value():
    # 85% of 54 regions: roughly 73% to 92%, as stated in the plan.
    low, high = wilson(round(.85*54), 54)
    assert low == pytest.approx(.73, abs=.02) and high == pytest.approx(.92, abs=.02)
    assert wilson(0, 0) is None


def test_bootstrap_resamples_whole_panoramas():
    records = [{'image': 'a_front.jpg', 'tp': 2, 'fp': 0, 'fn': 0}, {'image': 'a_back.jpg', 'tp': 1, 'fp': 0, 'fn': 1},
               {'image': 'b_front.jpg', 'tp': 0, 'fp': 2, 'fn': 2}]
    result = panorama_bootstrap(records, samples=500)
    assert result['panoramas'] == 2
    low, high = result['recall_95']
    assert low == 0 and high == .75  # Two panoramas: all-b gives 0/2, all-a gives 3/4.
    assert panorama_bootstrap([{'image': 'a_x.jpg', 'tp': 0, 'fp': 0, 'fn': 0}], 10)['precision_95'] is None


def test_tiled_evaluation_uses_view_pixels_for_size(tmp_path, monkeypatch):
    # A stub model that reports one 20px box in the first tile only.
    counter = {'n': 0}
    def run_rows(*_):
        counter['n'] += 1
        rows = np.array([[100, 100, 120, 120, .9, 0]], dtype=np.float32) if counter['n'] == 1 else np.empty((0, 6), dtype=np.float32)
        return [rows[None]]
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')], run=run_rows)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=lambda *a, **k: session))
    (tmp_path/'images').mkdir(); (tmp_path/'labels').mkdir()
    Image.new('RGB', (1024, 1024)).save(tmp_path/'images/v.jpg')
    (tmp_path/'labels/v.txt').write_text(f'0 {110/1024} {110/1024} {20/1024} {20/1024}\n')
    result = evaluate(Namespace(model=tmp_path/'m.onnx', pt=None, dataset=tmp_path, output=tmp_path/'o.json', confidence=.35, tiled=True))
    assert counter['n'] == 4 and result['true_positives'] == 1 and result['false_positives'] == 0
    assert result['per_image'][0]['truths'][0]['short_side_px'] == pytest.approx(20)
    assert result['mode'].startswith('tiled 1024px')


def test_raw_yolov5_model_is_decoded_with_its_manifest(tmp_path, monkeypatch):
    import json
    from training.evaluate import load_runner
    raw = np.array([[[320,320,100,100, .9, .9,0,0,0], [100,100,40,40, .9, 0,0,0,.9]]], dtype=np.float32)
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='images')], run=lambda *_: [raw])
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(InferenceSession=lambda *a, **k: session))
    unread, _ = load_runner(tmp_path/'m.onnx', .1)
    with pytest.raises(ValueError, match='--manifest'):
        unread(np.zeros((1, 3, 640, 640), np.float32))
    meta = tmp_path/'manifest.json'
    meta.write_text(json.dumps({'format': 'yolov5', 'class_map': {'0': 'litter', '1': 'litter', '2': 'litter', '3': None}, 'confidence_threshold': .9}))
    run, _ = load_runner(tmp_path/'m.onnx', .1, meta)
    rows = run(np.zeros((1, 3, 640, 640), np.float32))
    # The evaluation threshold (0.1) is used, not the manifest's 0.9; the trash bin is dropped.
    assert rows.tolist() == [[270, 270, 370, 370, pytest.approx(.81), 0]]
