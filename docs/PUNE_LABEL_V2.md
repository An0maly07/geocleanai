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

**Progress (7 October 2026): 958 / 960 views reviewed** (242 litter, 702 clean, 14 skipped); photos 443 and 720 remain. Vineet labelled 1–240, Jayesh 241–720 (481–719 merged from his zip), Siddhi 721–960 (merged from `siddhihingne12/GeoCleanAI_updated`, review files only). Siddhi also labelled photos 1–50; Vineet's labels were kept and hers are in `data/pune-label-v2/second-opinions/siddhi/` for an agreement check (27 of 50 views differ). The first training split (`pune-train-v1`/`pune-val-v1`) used only the 527 views available at the time; the next run needs a new split version.

```powershell
.\.venv\Scripts\python.exe -m training.review_server --data data/pune-label-v2
```

## Export and split

```powershell
.\.venv\Scripts\python.exe -m training.review_server --data data/pune-label-v2 --export data/pune-label-v2-export
.\.venv\Scripts\python.exe training\split_pune_labels.py
```

The split is fixed in advance and uses label counts only (never model results). Whole location groups go to one side. Validation takes about a third of the panoramas and 20–45% of the litter regions, and the largest group stays in training. Output: `data/pune-train-v1/` and `data/pune-val-v1/`, each with a hash lock.

## Show the labels in the app

```powershell
.\.venv\Scripts\python.exe scripts\build_team_labels.py
```

This writes `public/team-labels/` (an index plus the labelled views that contain litter). The website then lists **Team-labelled street views** and offers a **Team labels** switch next to **AI detections** for those panoramas. Only panoramas with all six views reviewed are included. The markers are the team's hand-drawn boxes, placed with the same geometry as model detections; the app labels them as human labels, never as AI output. Re-run it after merging more reviews.
