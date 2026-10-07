"""Build the round-2 notebook: Pune-focused second fine-tuning stage (s1-pune-focus).

It is notebook B with round-2 changes only, so training, the checkpoint sweep, export checks and
the locked evaluation run exactly the same code. Round 2 writes to new Drive folders
(experiments-r2, selection-r2), so round 1's results and locked selection stay untouched.
"""
import argparse
import json
import textwrap
from pathlib import Path

from build_colab_notebooks_v2 import ROOT, experiment_cells, notebook, setup_cells

TITLE = '''
# GeoCleanAI — Round 2: Pune-focused fine-tuning

Continues from round 1's selected checkpoint (E1, epoch 5) on **all 946 labelled Pune views**:
- Training: the Pune training split shown twice per epoch, plus 300 TACO photos and 300 street-like tiles so general litter is not forgotten.
- Validation: the **Pune validation split** (same location groups as round 1, so no validation street was ever trained on). Early stopping, `best.pt`, the checkpoint sweep and the threshold all use it.
- Round 1's E1 checkpoint and the v1 baseline are scored on the same validation views for comparison. If round 2 does not beat E1, the selection picks E1.
- The locked Pune test is then run once for the selected candidate. It is the second candidate evaluated on that test set (after E1 on 7 October 2026); neither was chosen using test results.

Needs in **My Drive/GeoCleanAI**: `geocleanai-colab-data-v1.zip`, the round-2 bundle, and round 1's `experiments/e1-domain-matched-20261007-140223/` folder (the starting checkpoint). Use a T4 GPU and **Runtime → Run all**.
'''

CHANGES = [
    ("EXPERIMENT = 'e1-domain-matched'  # then 'e2-capacity' / 'e3-tiled-1024' only if justified", "EXPERIMENT = 's1-pune-focus'"),
    ("""diagnosis = DRIVE/'diagnosis-baseline-v1/diagnosis-report.json'
if diagnosis.is_file() and 'training_curves' in json.loads(diagnosis.read_text()):
    config['epochs'] = json.loads(diagnosis.read_text())['training_curves']['next_epoch_cap']
else:
    print('Diagnosis logs not found: keeping the epoch cap at', config['epochs'])
""", """START = Path(config['model'])
assert START.is_file(), f'Round 1 starting checkpoint not found in Drive: {START}'
assert file_hash(START) == config['model_sha256'], 'The starting checkpoint differs from round 1 selection.'
"""),
    ("OUTPUT = DRIVE/'experiments'/f", "OUTPUT = DRIVE/'experiments-r2'/f"),
    ("# OUTPUT = DRIVE/'experiments'/'REPLACE", "# OUTPUT = DRIVE/'experiments-r2'/'REPLACE"),
    ("PUNE_VAL = V2/('datasets/pune-val-v1-r1024' if TILING else 'datasets/pune-val-v1')", "PUNE_VAL = V2/'datasets/pune-val-v2'"),
    ("checkpoints = sorted((run_dir/'weights').glob('*.pt')) + [BASE/'best.pt']",
     "checkpoints = sorted((run_dir/'weights').glob('*.pt')) + [START, BASE/'best.pt']  # round 1's E1 start, scored on the same views"),
    ("label = 'baseline-v1' if checkpoint == BASE/'best.pt' else f'{EXPERIMENT}-{checkpoint.stem}'",
     "label = 'baseline-v1' if checkpoint == BASE/'best.pt' else 'e1-start' if checkpoint == START else f'{EXPERIMENT}-{checkpoint.stem}'"),
    ("'sha256': file_hash(checkpoint), 'experiment': EXPERIMENT,",
     "'sha256': file_hash(checkpoint), 'experiment': 'e1-domain-matched' if checkpoint == START else EXPERIMENT,"),
    ("""- **E1:** if the best checkpoint's proxy recall (at the chosen threshold) beats the baseline by < 5 points → stop: the data, not the recipe, is the limit.
- **E2:** run only if E1 improved but underfits (training loss still falling, validation still rising at the end). Drop E2 if its gain over E1 is < 3 points.
- **E3:** run only if small-object recall stays low while ≥32px recall is good.""",
     """Compares round 2's best checkpoint with its starting point (round 1's E1) and the v1 baseline, all on the same Pune validation views.
If round 2 does not beat its start, the selection below picks the E1 checkpoint: more Pune data alone did not help, and the next lever is resolution (E3) or a larger model (E2)."""),
    ("""baseline = next(r for r in table if r['checkpoint'].endswith('baseline-v1/files/best.pt'))
best = next(r for r in table if r is not baseline)
gain = (best['score_recall'] or 0) - (baseline['score_recall'] or 0)
print(f"{EXPERIMENT}: proxy recall {best['score_recall']} vs baseline {baseline['score_recall']} (gain {gain*100:.1f} points)")
print('Stop condition met (gain < 5 points):', gain < .05)
(OUTPUT/'stop-check.json').write_text(json.dumps({'experiment': EXPERIMENT, 'gain': gain, 'best': best, 'baseline': baseline}, indent=2))""",
     """baseline = next(r for r in table if r['checkpoint'].endswith('baseline-v1/files/best.pt'))
start = next(r for r in table if r['checkpoint'] == str(START))
best = next(r for r in table if r is not baseline and r is not start)
for name, row in (('round 2 best', best), ('E1 start (round 1)', start), ('baseline v1', baseline)):
    print(f"{name:19} {Path(row['checkpoint']).name:12} Pune val recall {row['score_recall']} at threshold {row['chosen']['threshold']} (precision {row['chosen']['precision']})")
gain = (best['score_recall'] or 0) - (start['score_recall'] or 0)
print(f'Gain over the E1 start: {gain * 100:.1f} points.' + ('' if gain > 0 else ' Round 2 did not beat its start; the selection will pick the E1 checkpoint.'))
(OUTPUT/'stop-check.json').write_text(json.dumps({'experiment': EXPERIMENT, 'gain_over_start': gain, 'best': best, 'start': start, 'baseline': baseline}, indent=2))"""),
    ("sweeps = [json.loads(p.read_text()) for p in (DRIVE/'experiments').glob('*/checkpoint-sweep.json')]",
     "sweeps = [json.loads(p.read_text()) for p in (DRIVE/'experiments-r2').glob('*/checkpoint-sweep.json')]"),
    ("SELECT_DIR = DRIVE/'selection'; SELECT_DIR.mkdir(exist_ok=True)", "SELECT_DIR = DRIVE/'selection-r2'; SELECT_DIR.mkdir(exist_ok=True)"),
    ("'disclosure': 'Pune test views were inspected during baseline diagnosis; not used for selection.',",
     "'disclosure': 'Pune test views were inspected during baseline diagnosis; not used for selection. Second candidate evaluated on this test set (first: E1, 7 October 2026).',"),
    ("result_zip = SELECT_DIR/'geocleanai-v2-results.zip'", "result_zip = SELECT_DIR/'geocleanai-r2-results.zip'"),
    ("for path in (DRIVE/'experiments').rglob('*'):", "for path in (DRIVE/'experiments-r2').rglob('*'):"),
    ("bundle.write(path, Path('experiments')/path.relative_to(DRIVE/'experiments'))",
     "bundle.write(path, Path('experiments-r2')/path.relative_to(DRIVE/'experiments-r2'))"),
]


def round2_cells(setup) -> list[tuple[str, str]]:
    cells = [(kind, textwrap.dedent(source).strip()) for kind, source in experiment_cells(setup)]
    cells[0] = ('markdown', textwrap.dedent(TITLE).strip())
    for old, new in CHANGES:
        hits = [i for i, (_, source) in enumerate(cells) if old in source]
        if len(hits) != 1 or cells[hits[0]][1].count(old) != 1:
            raise ValueError(f'Expected exactly one match in notebook B for: {old[:70]!r}')
        kind, source = cells[hits[0]]
        cells[hits[0]] = (kind, source.replace(old, new))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/colab/geocleanai-colab-r2.zip')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/colab/GeoCleanAI_R2_PuneFocus.ipynb')
    args = parser.parse_args()
    v1_sha = (ROOT/'artifacts/colab/geocleanai-colab-data-v1.zip.sha256').read_text().split()[0]
    bundle_sha = Path(str(args.bundle) + '.sha256').read_text().split()[0]
    cells = round2_cells(setup_cells(v1_sha, args.bundle.name, bundle_sha))
    args.output.write_text(json.dumps(notebook(cells, args.output.name), indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {args.output} ({len(cells)} cells; all code cells compile)')


if __name__ == '__main__':
    main()
