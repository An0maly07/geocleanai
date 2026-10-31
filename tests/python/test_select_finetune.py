import json

import pytest

from training.select_finetune_views import far_from, location_groups, newest_per_spot, select, test_points as read_test_points


def pano(id, lat, lng, captured_at=1):
    return {'id': id, 'lat': lat, 'lng': lng, 'captured_at': captured_at, 'creator': 'c', 'area': 'A'}


def test_newest_photo_wins_within_a_spot():
    # 'a' and 'b' are ~5 m apart (same spot); 'c' is ~200 m north.
    kept = newest_per_spot([pano('a', 18.6, 73.8, 1), pano('b', 18.60004, 73.8, 5), pano('c', 18.6018, 73.8, 2)])
    assert [p['id'] for p in kept] == ['b', 'c']


def test_panoramas_near_a_test_panorama_are_dropped():
    test = [(18.6, 73.8)]
    kept = far_from([pano('near', 18.6018, 73.8), pano('far', 18.6036, 73.8)], test)  # ~200 m and ~400 m
    assert [p['id'] for p in kept] == ['far']
    assert kept[0]['nearest_test_m'] == pytest.approx(400, abs=5)


def test_chained_panoramas_share_a_location_group():
    # a-b and b-c are ~78 m apart, so all three join; d is ~1 km away.
    rows = [pano('a', 18.6, 73.8), pano('b', 18.6007, 73.8), pano('c', 18.6014, 73.8), pano('d', 18.61, 73.8)]
    groups = location_groups(rows)
    assert groups['a'] == groups['b'] == groups['c'] != groups['d']


def test_selection_refuses_without_test_panoramas(tmp_path):
    empty = tmp_path / 'tests.json'; empty.write_text('[]')
    with pytest.raises(ValueError, match='exclusion zone'):
        read_test_points([empty])


def test_selection_refuses_a_folder_with_labels(tmp_path):
    (tmp_path / 'labels').mkdir(); (tmp_path / 'labels' / 'x.txt').write_text('')
    tests = tmp_path / 'tests.json'; tests.write_text(json.dumps([{'lat': 18.6, 'lng': 73.8}]))
    with pytest.raises(ValueError, match='already has labels'):
        select(tmp_path, [tests])
