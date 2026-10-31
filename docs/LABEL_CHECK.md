# Checking Claude's fine-tune labels (plan task 2B.3)

Claude (an AI) labelled 756 Pune street views for fine-tuning the pLitter model. Before training, a person checks a random 76 of them (10%): 38 that Claude marked as litter and 38 it marked clean. About an hour.

## Start the page

```bash
.venv/bin/python -m training.review_server --data data/pune-finetune-v1-check
```

Open http://127.0.0.1:8765 and type your name or initials in the box on the right.

## For each photo

**Dashed blue boxes are Claude's labels.** They are not a detector's guesses, so the percentage next to them means nothing.

1. Look at the whole photo first, including the road edges.
2. **A box is right** (it sits tightly around real litter): click it, or press **A** to accept every box on the photo and move on.
3. **A box is wrong** (kerb paint, a sign, a car, building material in use, shadow, or not tight): click **Reject** for that box. If there is litter there but the box is badly placed, reject it and drag a new box.
4. **Claude missed litter:** drag a new box around it.
5. **No boxes and no litter:** press **C** ("I checked: no visible litter").
6. **You can't tell** (too blurry or dark): **Skip** and write why in Notes.

What counts as litter: bags, wrappers, bottles, cups, paper, cloth, loose or torn sacks, and piles of dumped rubbish. A long strip of dumped rubbish can be one box. **Not litter:** painted kerbs and road markings, bins, vehicles, people, plants, the camera car, and building material that is in use (sand or gravel heaps, bricks, site sheeting, stacked or filled sacks at a work site).

Ignore the yellow note "This view uses a smaller preview": it is about a different export and does not matter here. The page does not record whether a box is a single item or a pile; this check only asks whether each box is litter.

## When you finish

Tell Claude, or run:

```bash
.venv/bin/python -m training.label_check compare
```

It counts how many checked photos needed a fix: a litter/clean decision changed, a Claude box rejected, or a box you drew. **If more than 1 in 5 needed a fix, the labels fail**, and Claude re-labels using what you corrected before training starts. The result is saved to `data/pune-finetune-v1-check/check-result.json`.

Your reviews are saved as you go, so you can stop and continue later.
