import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training'))
from build_colab_notebooks_v2 import notebook, setup_cells  # noqa: E402
from build_round2_notebook import round2_cells  # noqa: E402


def test_round2_notebook_applies_every_change_and_writes_to_new_drive_folders():
    cells = round2_cells(setup_cells('0' * 64, 'geocleanai-colab-r2.zip', '1' * 64))
    code = '\n'.join(source for kind, source in cells if kind == 'code')
    assert "EXPERIMENT = 's1-pune-focus'" in code and "datasets/pune-val-v2" in code
    assert "DRIVE/'selection-r2'" in code and "DRIVE/'selection'" not in code
    assert "(DRIVE/'experiments')" not in code and "DRIVE/'experiments'/" not in code
    notebook(cells, 'round2-test')  # every code cell compiles


def test_setup_pins_ultralytics_from_requirements_and_disables_autoinstall():
    from build_colab_notebooks_v2 import ultralytics_pin
    install = next(source for kind, source in setup_cells('0' * 64, 'x.zip', '1' * 64) if 'pip' in source and 'PIN' in source)
    assert f"PIN = '{ultralytics_pin()}'" in install
    assert install.index("YOLO_AUTOINSTALL'] = 'False'") < install.index("'pip', 'install'")
