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
3. **Label the new Pune batch** (160 panoramas, 960 views; see `docs/PUNE_LABEL_V2.md` and `docs/TEAM_LABELLING.md`), merge the team's labels into `data/pune-label-v2` and export them. There is **no Pune train/val split** (decided 26 September 2026): every labelled view trains. The locked 48-view test set never does; the script refuses its capture sequences and any byte-identical image:
   ```
   python -m training.review_server --data data/pune-label-v2 --export data/pune-label-v2-export
   python training/split_pune_labels.py --all-train --train-output data/pune-train-all
   ```
4. **Manual TACO review** of `data/litter-review/review-queue.csv` (see `docs/DATA_QUALITY.md`), then build and package:
   ```
   python training/build_derivatives.py --output data/litter-v2 --profile r640 --pune-train data/pune-train-all
   # E3/E2 (tiled 1024px): the same Pune views re-rendered at 1024 from the originals, labels unchanged
   python training/derive_pune_resolution.py --source data/pune-train-all --output data/pune-train-all-r1024
   python training/build_derivatives.py --output data/litter-v2-r1024 --profile r1024 --pune-train data/pune-train-all-r1024
   python training/package_colab_v2.py --output artifacts/colab/geocleanai-colab-v3.zip --datasets litter-v2 litter-v2-r1024
   python training/build_colab_notebooks_v2.py --bundle artifacts/colab/geocleanai-colab-v3.zip
   ```
   Pune views or tiles with litter are repeated 3× (`--pune-repeat`); litter-free Pune tiles are used once and capped at 15% of the training set (`--pune-empty-share`). Why: `docs/SMALL_OBJECT_TRAINING.md`.
5. **Notebook B**: E1, then E3 (set `EXPERIMENT` in the first cell). E2 (P2 head) only if its condition below is met.
6. **Select and lock** one candidate (cell in B). Then run the locked Pune evaluation **once**.

## Experiments (at most three)

Training data for every experiment: TACO originals + street-like TACO tiles + every labelled Pune view (litter pieces repeated 3×, litter-free pieces capped at 15%). With no Pune validation split, model choice and the threshold use the street-like TACO proxy.

Common: Ultralytics 8.4.155 (checked at install, start of training and before export; automatic package installs are switched off), seed 42, deterministic, patience 10, checkpoints every 5 epochs, batch 16 with 8 → 4 fallback on GPU memory errors (Ultralytics accumulates gradients to an effective batch of 64, so this does not change the learning rate schedule). Validation = street-like proxy. Epoch cap = 50 unless the baseline logs show it was still improving at epoch 50 (then 100). Small-object augmentation: zoom range 0.75–1.5 instead of 0.5–1.5, no rotation, shear, perspective, vertical flip, MixUp or CutMix (see `docs/SMALL_OBJECT_TRAINING.md`).

| | E1 Domain-matched | E3 Real detail (tiled 1024) | E2 Small-object head (P2) |
|---|---|---|---|
| Question | Does street-like TACO plus Pune transfer better? | Does real panorama detail recover small litter? | Does a stride-4 head find litter under 16 px that stride 8 misses? |
| Run only if | Gate 1 passed | **Justified by the diagnosis**: TACO test recall 9% at 8–16 px vs 68% at ≥32 px; 61% of Pune regions under 16 px | After E3, if its <16 px proxy recall is still well below its ≥32 px recall |
| Start | `yolo26n.pt` | `yolo26n.pt` | `yolo26n-p2.yaml` initialised from `yolo26n.pt` |
| Data | litter-v2 (640 scale) | litter-v2-r1024 | litter-v2-r1024 |
| Input | 640 | 1024px views as 2×2 overlapping 640 tiles | as E3 |
| Deploy change | none | 24 inferences per panorama; `backend/tiling.py` is used by the app only when the model manifest has a `tiling` block; runtime must be re-measured | as E3, ≈1.6× compute per tile (9.5 vs 6.1 GFLOPs); re-measure |
| Stop if | proxy recall gain < 5 points over baseline | small-object gain < 5 points or latency gate fails | gain over E3 < 3 points or latency gate fails |

The earlier E2 (a larger `yolo26s` model) was dropped: the diagnosis found the baseline was not undertrained, and the limit is object size, not capacity.

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
