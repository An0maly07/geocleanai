# AI-assisted labelling

A ready-made open-vocabulary detector (OWLv2, prompted with descriptions of litter) draws **suggested** boxes on photos nobody has reviewed yet. A person accepts or rejects every suggestion; nothing becomes a label on its own. Suggestions are never shown on photos that already have a review.

Why not let the detector label by itself: labels are both what the model learns from and what it is scored against. Unchecked machine boxes would teach the model the detector's mistakes and make the validation score measure agreement with the detector instead of real litter.

## 1. Build the Colab bundle (on this computer, no model runs)

```powershell
.\.venv\Scripts\python.exe training\package_suggestions.py
.\.venv\Scripts\python.exe training\build_suggest_notebook.py
```

The first command renders every view of `data/pune-label-v2` at 1280 px from its original panorama and adds the team's existing labels (used only to score the suggestions). It writes `artifacts/colab/geocleanai-suggest-v1.zip`. The second writes `artifacts/colab/GeoCleanAI_C_Suggestions.ipynb`, which checks that exact zip by its SHA-256.

## 2. Run notebook C in Colab

Upload the zip into **My Drive/GeoCleanAI**, open the notebook, choose a T4 GPU runtime and use **Runtime → Run all** (about 30–45 minutes). If Colab disconnects, run all again; finished views are kept in Drive.

The notebook scores the suggestions against the views the team already labelled: precision, recall, recall by litter size, and how many clean views would get suggestions, at IoU 0.5 and 0.3. It picks the threshold with the best F1 at IoU 0.5 and warns when fewer than 3 in 10 suggestions are right. It then downloads `suggestions.json`.

## 3. Install the suggestions

```powershell
.\.venv\Scripts\python.exe training\import_suggestions.py $HOME\Downloads\suggestions.json
```

Each suggestion is matched to its image by checksum; views whose image changed get none. An earlier `suggestions.json` is kept with a timestamp. Restart the labelling page afterwards:

```powershell
.\.venv\Scripts\python.exe -m training.review_server --data data/pune-label-v2
```

## 4. Label with suggestions

Dashed blue boxes are suggestions. The side panel shows how well they scored.

| Key | Action |
|---|---|
| Click a dashed box | Accept that suggestion |
| **A** | Accept all suggestions, save, next photo |
| **X** | Reject all suggestions |
| **S** | Save the boxes, next photo |
| **C** | No visible litter (asks to confirm), next photo |
| **Z** | Undo last box |
| **N** / **→**, **P** / **←** | Next / previous photo |

Accept only boxes that sit tightly on real litter; reject loose or wrong ones and draw your own. **Always scan the whole photo**: the detector misses many small items, and a missed item saved as "no litter" teaches the model to ignore it.

Every saved review records `box_sources` (`ai` for an accepted suggestion, `hand` for a drawn box) and `suggestions_shown`, so assisted and hand-drawn labels can be compared later. Export and training use the boxes exactly as before.
