# Pune labelling batch (pune-label-v2)

160 Pune panoramas (960 views) for **training and validation**, labelled by a team of four on one review page. The locked test set (`data/pune-test-v1`) is not part of this and is never changed. This batch replaces the earlier 40-panorama batch (`pune-label-v1`, removed unlabelled); it keeps nothing from it except the method.

## Where it came from (26 September 2026)

- `scripts/find_training_panoramas.py --grid-km 2.5 --radius-m 1800` searched Mapillary metadata over a grid covering the whole Pune demo area (182 searches). It found 13,393 panoramas. 29 searches failed at Mapillary and 4 hit the result cap, so this is still not a complete city inventory.
- Excluded: panoramas in the 8 test capture sequences, panoramas within 300 m of any panorama reviewed for the test (kept or skipped), and those reviewed panoramas themselves. That left **12,768 safe panoramas in 163 sequences**.
- `scripts/select_label_batch.py --count 160` (seed 42) took one panorama from each of 160 sequences, at least 100 m apart within a sequence. They fall into 24 location groups (sequences passing within 100 m of each other), the largest holding 34 panoramas.
- Limitation: 155 of 160 panoramas come from one contributor's 2024–2026 captures, mostly wider roads. Expect many clean views. Clean views are still useful: they are real Pune street scenes, which TACO lacks, and they teach the model where not to fire.
- Images: Mapillary, CC BY-SA 4.0. Contributors, dates and source links are in `data/pune-label-v2/manifest.json`; credit them wherever images are shown.

## Label

One page for the whole team: see `docs/TEAM_LABELLING.md`.

```powershell
.\.venv\Scripts\python.exe -m training.review_server --data data/pune-label-v2
```

## Export

```powershell
.\.venv\Scripts\python.exe -m training.review_server --data data/pune-label-v2 --export data/pune-label-v2-export
.\.venv\Scripts\python.exe training\split_pune_labels.py --all-train --train-output data/pune-train-all
```

No train/val split (decided 26 September 2026): every labelled view trains, and model selection uses the street-like TACO proxy. The script refuses views from the locked test's capture sequences and any image byte-identical to a locked test image, then writes `data/pune-train-all/` with a hash lock. The original group-based split (without `--all-train`) is still available if a Pune validation set is wanted later.
