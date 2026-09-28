# M1: card detector, v0

**Date:** 2026-09-28. **Status:** trained once, on synthetic boards only, and run on the three M0 broadcasts. It passes the M1 recall bar on real footage it never trained on.

The question from [ARCHITECTURE §3.8](../ARCHITECTURE.md): can an amodal detector, trained only on the synthetic boards of [M1 synth v0](m1-synth-v0.md), find the cards on real broadcasts, including stacked and covered ones, well enough to replace the bootstrap mat finder? Numbers only. No frames are in this folder (D-006, D-015).

## Summary

- **On real footage it finds 97.6% of the 5,465 cards reviewed in M0**, zero-shot: 98.1% at Los Angeles, 99.5% at Barcelona, 96.2% at Shenyang (red mat, the smallest cards). The M1 exit bar is 95% ([ROADMAP](../ROADMAP.md)). It tells a face-up card from a sleeve back 98.1% of the time.
- **Those reviewed cards are the easy half.** They are the isolated cards the bootstrap finder could see, so this measures recall on cards lying on their own. Stacks and precision are measured on synthetic boards only, below. Real stack labels come next, from the detector's own proposals.
- **On held-out synthetic boards, fully visible cards are all found** (99.97%, corners within 1.3% of the card's length, about 2 px on the broadcast), and covered cards mostly: 88% of those half or more visible, 82% of strips, 40% of slivers under 15%.
- **Watching the grand final live**, it finds nearly the whole board: rune columns, sleeved units, battlefields and legends in magnetic cases under dice, where the mat finder found 3 to 6 cards a frame. Its slips (a box between two cards, a card in a case outlined two or three times) are filtered by the live tracker ([ml/README](../../ml/README.md#live-a-recording-or-a-stream-named-as-it-plays)).

## The model and the run

RF-DETR keypoint preview (Apache-2.0 code and weights), fine-tuned on four corners per card with per-corner visibility, classes `card` and `card_back` ([ml/README](../../ml/README.md#card-detector-m1)). 2,000 synthetic boards, every tenth held out, cut into 576 px tiles at two scales. 10 epochs, batch 8 × 2, cosine learning rate to 5%, on one NVIDIA L4: 280 minutes for the whole job, training about 27 minutes an epoch. Weights 82 MB in float16, private (they learned from card art).

RF-DETR's own validation on the held-out tiles, after the last epoch ([training CSV](m1-detector-v0-training.csv)):

| Box mAP 50 | Box mAP 50:95 | Keypoint mAP 50:95 | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 98.1% | 92.1% | 87.4% | 97.4% | 95.3% | 96.3% |

It still rose a little every epoch, most in the strict box fit (mAP 50:95 from 90.3% at epoch 5).

## Held-out synthetic boards, strict

200 held-out frames, 6,564 cards showing at least 8% of themselves. A card counts as found only when the detector's full quad (the card's whole outline, covered part included) overlaps it at IoU ≥ 0.75. Precision counts every other box as a miss, the boxes of cards under 8% visible included, so it reads low ([CSV](m1-detector-v0-synth.csv)):

| Cut | All | Whole | Half or more | Strip (15–50%) | Sliver (8–15%) | Precision | Corner error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 92.8% | 99.97% | 88.5% | 85.4% | 68.2% | 52.9% | 1.33% |
| **0.5** | **91.1%** | **99.97%** | **88.3%** | **81.5%** | **40.1%** | **79.0%** | **1.32%** |

Card backs: 91.9% at both cuts. The live runner keeps boxes at 0.4 by default, and `--det-score 0.5` drops most of the extra ones.

## Real broadcasts

The detector ran on the frames M0 sampled from each broadcast (1,681 frames), inside each layout's table window. A reviewed card counts as found when a box overlaps its M0 crop at IoU ≥ 0.5 ([CSV](m1-detector-v0-real.csv)):

| Broadcast | Frames | Reviewed cards | Found | Card backs found | Class right |
|---|---:|---:|---:|---:|---:|
| Los Angeles RQ, official stream (cards about 163 px) | 574 | 1,729 | **98.1%** | 100% (62) | 98.1% |
| Barcelona, PlusRB restream (about 142 px) | 444 | 1,299 | **99.5%** | 99.6% (237) | 98.1% |
| Shenyang (red mat, about 131 px) | 663 | 2,437 | **96.2%** | 97.7% (43) | 98.3% |
| **All** | **1,681** | **5,465** | **97.6%** | | **98.1%** |

On a 4-core cloud CPU it takes 0.6 s for a 1080p frame whose table window fits one tile (Los Angeles) and about 1.3 s for two (Barcelona, Shenyang).

## What it does not show yet

- **Real stacks and real precision.** No real frame has every card outlined yet. Real stack labels were the plan from the start ([M0 §7](m0-spike.md#7-stacked-cards-identity-from-a-strip)): the detector proposes the covered cards, a review pack confirms them, and those reviews are the first real stack labels.
- **Phone-size and picture-in-picture cards.** The smallest real cards here are about 130 px long; the training boards went down to 45 px.
- **Speed on the viewer's machine.** The live runner uses it on a Mac's GPU (MPS), where its frame rate is not measured yet. ONNX export and a small browser model are M2 work (D-013).

## Next

1. **Covered-card review packs** from these detections, for the first real stack labels.
2. **Real frames in training:** those labels plus the reviewed isolated cards, mixed into the synthetic set, and a second round.
3. **The embedder** (training now) to name what the detector finds, including strips of covered cards.

## Reproduce

```bash
cd ml && . .venv/bin/activate
bash scripts/m1-detector.sh                     # on a GPU machine; SMOKE=1 first for a short check
python -m rifteye_ml.detect run --frames DIR --only la-frames.txt --table 0.157,0,0.843,1 --card-px 163.3 \
  --checkpoint detector-v0.pth --out dets-la.jsonl
python -m rifteye_ml.detect score-real --dets dets-la.jsonl --crops la-v1/crops/crops.json --labels la-v1/labels.csv
```
