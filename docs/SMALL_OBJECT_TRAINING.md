# Training for small litter: choices (26 September 2026)

Why this exists: the baseline found 0 of 54 Pune regions. The diagnosis (`docs/BASELINE_V1.md`) showed two causes: litter is tiny (61% of Pune regions under 16 px at 640, where TACO test recall is 9% at 8–16 px and under 1% below 8 px) and the model had never seen Pune streets. Every choice below targets one of those two causes. None was chosen by looking at Pune test results.

## Resolution: tiles, not a bigger input

- The deployed model keeps a fixed 640px input (the API refuses anything else). Detail comes from rendering each view at 1024px and cutting it into 2×2 overlapping 640px tiles (E3, E2). This cuts Pune regions under 16 px from 61% to 41%, and under 8 px from 30% to 7%.
- Training tiles are cut with the same grid as the app (`backend/tiling.py`, 384px stride), so the model sees litter at exactly the pixel size it will see in deployment. Training at `imgsz` above 640 was rejected: the model would learn object sizes it never meets at inference.
- 1280px views were not added: they would need 9 tiles per view (54 per panorama) against a 45-second processing limit. Revisit only if E3 passes and latency has room.

## Augmentation

- **Zoom range 0.75–1.5** (Ultralytics' default is 0.5–1.5). Ultralytics silently drops any box that ends up 2 px or smaller after augmentation but leaves its pixels in the image, which teaches the model that small litter is background. Our smallest kept label is 3 px (`MIN_LABEL_PX`), and 3 × 0.75 stays above 2 px. Zooming in up to 1.5× also gives tiny items more pixels during learning.
- **Off:** rotation, perspective, vertical flip (street views have a fixed horizon), MixUp and CutMix (both blend or paste whole images, which blurs pieces of litter only a few pixels wide).
- **Kept:** mosaic (it keeps each tile at its native scale and adds scene variety; switched off for the last 10 epochs as usual), horizontal flip, colour jitter. Blur, noise, shadows and JPEG loss are already baked into the street-like TACO tiles.
- Copy-paste was not used: it needs segmentation masks and our labels are boxes.

## Anchors → a P2 head (E2)

YOLO26 is anchor-free, so there are no anchors to tune. The equivalent lever is where the model predicts: its finest head is stride 8, so an 8 px item is one grid cell. E2 adds a stride-4 (P2) head (`yolo26n-p2.yaml`, started from `yolo26n.pt` weights). Only 360 of its 902 weight tensors come from `yolo26n.pt` (the backbone; the head is new), so it starts less pretrained than E3. It costs about 1.6× compute per tile, so it runs only after E3, and only if E3 still misses much more below 16 px than at 32 px and over. It replaces the old E2 (the larger `yolo26s`), because the diagnosis found the baseline was not undertrained.

## Mixing Pune and TACO

- There is no Pune train/val split: every labelled Pune view trains (`split_pune_labels.py --all-train`). The locked 48-view test set never trains; the script refuses its capture sequences and any byte-identical image, and the labelling batch already excluded panoramas within 300 m of it.
- TACO provides volume: originals, plus "street-like" collages shrunk to street distances (`build_derivatives.py`).
- Pune views or tiles **with litter** are repeated 3×, so about 160 panoramas are not swamped by about 3,000 TACO images.
- Pune tiles **without litter** are real street negatives (flags, people, road clutter: the baseline's false alarms). With all 960 views training they would be most of the data, which pushes recall down, so each is used once and they are capped at 15% of the training set, picked by seed. Tiles where litter was only partly visible (greyed out) or under 3 px are neither clean nor usable positives and are left out.

## What decides

With no Pune validation set, the checkpoint and threshold are chosen on the street-like TACO proxy, and the only Pune number is the single locked test run. That is weaker than validating on Pune: a proxy win may not transfer. If the proxy and the final Pune result disagree badly, a small held-out Pune validation set is the next thing to add (`split_pune_labels.py` without `--all-train` still does this).

## Checked so far

On CPU with Ultralytics 8.4.155, one epoch on a toy dataset: the E1 and E2 configs train, the zoom range reaches the trainer as `[0.75, 1.5]`, the P2 model loads `yolo26n.pt` weights, and both export to the `[1, 3, 640, 640]` → `[1, 100, 6]` ONNX format the API requires. The 101 Python tests pass. No real training has run; the effect on recall is unmeasured.
