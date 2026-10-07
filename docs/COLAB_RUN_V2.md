# Colab v2: diagnose, experiment, evaluate once

All training and model evaluation run in Google Colab (free T4). Nothing in this guide runs a model on your computer.

## Files

In `artifacts/colab/`:

- `geocleanai-colab-v2.zip` — code, regression tests, the preserved baseline (read-only copy) and the 1024px Pune test copy. Small; derived training data is added in a later bundle.
- `GeoCleanAI_A_Diagnosis.ipynb` — explains the baseline's 0% result. Trains nothing.
- `GeoCleanAI_B_Experiments.ipynb` — up to three experiments, selection on validation data, one locked Pune evaluation.

Keep `geocleanai-colab-data-v1.zip` in **My Drive/GeoCleanAI**; both notebooks reuse it. Upload the v2 zip next to it.

## Order

1. **Notebook A.** Set `RUN_FOLDER` to the first run's folder (`My Drive/GeoCleanAI/training-output/<date>`) so the real `results.csv` is used. Run top to bottom. Download `My Drive/GeoCleanAI/diagnosis-baseline-v1/`.
2. **Gate 1** (printed at the end of A) must pass before training: integrity, regression tests, the official 0/21/54 reproduced, logs read, every miss categorised. If a pipeline bug is found, it is fixed and the *same* baseline is re-scored once at 0.35.
3. **Label the new Pune batch** (160 panoramas, 960 views; see `docs/PUNE_LABEL_V2.md` and `docs/TEAM_LABELLING.md`), export it, and split it with `python training/split_pune_labels.py` into `data/pune-train-v1` and `data/pune-val-v1`.
4. **Manual TACO review** of `data/litter-review/review-queue.csv` (see `docs/DATA_QUALITY.md`), then build and package:
   ```
   python training/build_derivatives.py --output data/litter-v2 --profile r640 --pune-train data/pune-train-v1
   # E3 (tiled 1024px): the same Pune views re-rendered at 1024 from the originals, labels unchanged
   python training/derive_pune_resolution.py --source data/pune-train-v1 --output data/pune-train-v1-r1024
   python training/derive_pune_resolution.py --source data/pune-val-v1 --output data/pune-val-v1-r1024
   python training/build_derivatives.py --output data/litter-v2-r1024 --profile r1024 --pune-train data/pune-train-v1-r1024
   python training/package_colab_v2.py --output artifacts/colab/geocleanai-colab-v3.zip --datasets litter-v2 pune-val-v1 litter-v2-r1024 pune-val-v1-r1024
   python training/build_colab_notebooks_v2.py --bundle artifacts/colab/geocleanai-colab-v3.zip
   ```
5. **Notebook B**: E1, then E3 (set `EXPERIMENT` in the first cell). E2 only if its condition below is met.
6. **Select and lock** one candidate (cell in B). Then run the locked Pune evaluation **once**.

## Experiments (at most three)

Training data for every experiment: TACO originals + street-like TACO tiles + the Pune training views (repeated 3×). Model choice uses the Pune validation set (whole location groups never seen in training); the TACO proxy is the fallback for the threshold when Pune validation has fewer than 20 litter regions.

Common: Ultralytics 8.4.155 (checked at install, start of training and before export; automatic package installs are switched off), seed 42, deterministic, patience 10, checkpoints every 5 epochs, batch 16 with 8 → 4 fallback on GPU memory errors (Ultralytics accumulates gradients to an effective batch of 64, so this does not change the learning rate schedule). Validation = street-like proxy. Epoch cap = 50 unless the baseline logs show it was still improving at epoch 50 (then 100).

| | E1 Domain-matched | E2 Capacity | E3 Real detail (tiled 1024) |
|---|---|---|---|
| Question | Does street-like TACO transfer better? | Is the nano model the limit? | Does real panorama detail recover small litter? |
| Run only if | Gate 1 passed | E1 improves on the proxy but underfits | **Justified by the diagnosis**: TACO test recall 5% under 16 px vs 68% at ≥32 px; 61% of Pune regions under 16 px |
| Start | `yolo26n.pt` | `yolo26s.pt` | better of E1/E2 |
| Data | litter-v2 (640 scale) | litter-v2 | litter-v2-r1024 |
| Input | 640 | 640 | 1024px views as 2×2 overlapping 640 tiles |
| Deploy change | none | ≈3–4× larger/slower model; re-measure | 24 inferences per panorama; `backend/tiling.py` is used by the app only when the model manifest has a `tiling` block; runtime must be re-measured |
| Stop if | proxy recall gain < 5 points over baseline | gain over E1 < 3 points | small-object gain < 5 points or latency gate fails |

ONNX/PyTorch agreement rule (decided 26 September 2026 after the diagnosis): box counts must be equal; boxes are clipped to the image first (PyTorch clips, raw ONNX does not, the app does); each pair passes at IoU ≥ 0.99 or with every corner within 1 px. The old strict result is still recorded next to it.

Selection rule: for each checkpoint, precision/recall on the proxy at thresholds 0.05–0.95; the operating threshold is the lowest with precision ≥ 0.85 (otherwise maximum F1); score = recall there, tie-break small-object recall. Every candidate is also reported at 0.35.

## What the final report contains

Precision, recall, TP/FP/FN at the locked threshold and at 0.35; recall by label size (<8, 8–16, 16–32, ≥32 px); false alarms on the 30 clean views; the baseline on identical inputs; a 95% Wilson interval and a bootstrap over the 11 panoramas. For scale: 85% measured on 54 regions has a Wilson interval of about 73–92%, before accounting for correlated views. This test set cannot establish citywide accuracy.

## Reading the result honestly

- Misses mostly under 8 px, even after E3 → resolution/data ceiling; more TACO tuning is unlikely to help.
- Misses at ≥ 16 px while the proxy looks good → the proxy does not represent Pune; progress needs labelled Pune-like training data (outside current constraints).
- Training mAP far above validation → not enough data variety.
- False alarms concentrated in a few scenes → look-alike confusion; TACO has no clean scenes, so negatives are limited.

## Final acceptance checklist

- [ ] Pune precision ≥ 0.85 **and** recall ≥ 0.85 at IoU 0.50, at the threshold locked from validation; TP/FP/FN, size buckets, clean-view false alarms and intervals reported. Below target is reported as a failure.
- [ ] Selection record hashed before the Pune run; test-inspection disclosure included; no test examples dropped.
- [ ] ONNX/PyTorch one-to-one agreement passes; Ultralytics version matches the pin.
- [ ] Real panorama marker alignment on ≥ 5 Pune panoramas: markers on the objects, near seams and view boundaries, after rotate/zoom/fullscreen, desktop and mobile; duplicates across views suppressed.
- [ ] Uncached six-view (or 24-tile) analysis within the 45 s processing check and the 60 s hosting limit on the Vercel preview; cold and warm measured separately over ≥ 5 panoramas.
- [ ] The rejected baseline is not installed (the backend refuses its hash; the guard test passes).
- [ ] Hitting the numbers alone does not finish the work: panorama validation above must also pass.

## Round 2: Pune-focused second stage (7 October 2026)

After E1 (Pune test: 9 of 54 found, precision 8%) and with all 960 views labelled, round 2 continues from E1's selected checkpoint instead of running E2/E3 first. The three-experiment limit above applies to the first-stage experiments E1–E3; a second stage (`training/experiments/s1-*.json`) must name its starting checkpoint and its SHA-256.

```
python training/split_pune_labels.py --export data/pune-label-v2-export-full --train-output data/pune-train-v2 --val-output data/pune-val-v2 --keep-validation data/pune-val-v1/evaluation-lock.json
python training/build_pune_focus.py
python training/package_colab_v2.py --output artifacts/colab/geocleanai-colab-r2.zip --datasets pune-focus-v1 pune-val-v2
python training/build_round2_notebook.py
```

- **Split:** validation keeps E1's three location groups (now 57 panoramas, 342 views, 253 litter regions; 36% of panoramas and 29% of regions), so no validation street was ever trained on. Training: 101 panoramas, 604 views, 607 regions.
- **Data (`pune-focus-v1`):** Pune training views ×2 (1,208), 300 TACO photos and 300 street-like tiles (67% Pune); validation is Pune only, so early stopping and `best.pt` follow Pune.
- **Config (`s1-pune-focus`):** start from E1 `epoch5.pt` (hash-checked), AdamW lr0 0.0005, 30 epochs, patience 8, augmentation scale 0.25.
- **Notebook (`GeoCleanAI_R2_PuneFocus.ipynb`):** notebook B with new Drive folders (`experiments-r2`, `selection-r2`); E1's checkpoint and the baseline are scored on the same validation views, and E1 is selected if round 2 does not beat it. The locked test is run once for the selected candidate and the result discloses that it is the second candidate on that test set.
