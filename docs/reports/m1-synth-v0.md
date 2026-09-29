# M1: synthetic boards, v0

> Written when the project was called RiftEye. It is now Wardeye ([D-020](../decisions.md#d-020-the-product-is-called-wardeye)).

**Date:** 2026-09-27. **Status:** the generator is built and calibrated once against the M0 broadcasts, and a first set of 400 boards is rendered (private). Training the detector and the embedder on it needs a GPU.

The question from [04 §4.3](../research/04-data-and-evaluation.md#43-synthetic-board-generator-mlsynth): can RiftEye make whole broadcast frames of Riftbound boards, with exact labels, that look enough like real streams to train the detector and the embedder before real frames are labelled? Numbers only. No frames are in this folder (D-006, D-015).

## Summary

- **The generator works end to end.** `python -m rifteye_ml.synth` samples a plausible 1v1 board on the tournament table, films it through a camera model into a broadcast layout, and pushes every frame through a real H.264 encode and decode ([ml/README](../../ml/README.md#synthetic-boards-m1)). It writes each card's full quad, visible fraction, zone, pile and identity, and a 16-bit id map. Face-down cards never carry an identity. Runs are bit-exact for a seed.
- **The stacks M1 needs are dense.** Of the face-up cards inside the frame, 59% are partly covered, in rune columns and fans, fanned bases, tucked gear and piles. Real footage cannot give that: labelling real covered cards cheaply failed ([M0 §7](m0-spike.md#7-stacked-cards-identity-from-a-strip)).
- **The first version was far too clean.** Its fully visible cards were named 99% of the time by every M0 baseline, legends under dice included. On the real broadcasts the colour grid named 70–95%, and 16–51% of the legends at Los Angeles and Barcelona.
- **After one calibration pass, ordinary cards are in the real range.** Colour grid 95.7%, dHash 96.3%, the two together 98.3%, close to the Shenyang set (94.7%, 91.3%, 96.6%). Legends are still easier than at Los Angeles and Barcelona.

## The v0 set

400 boards (seed 2026) in 19 minutes on a 4-core cloud CPU, 348 MB:

| | |
|---|---|
| Layouts | 200 between RQ-style side panels, 125 full screen, 75 picture-in-picture |
| Resolutions and bitrates | 256 at 1080p, 112 at 720p, 32 at 480p; 2.5, 4, 6 or 8 Mbps per clip |
| Card size (long side) | median 110 px full screen, 114 px between panels, 56 px picture-in-picture; 22–172 px overall |
| Cards in view | 24,266 (61 per board): 16,037 face up, 8,229 face down (34%) |
| Face-up cards inside the frame | 11,654: 41% fully visible, 25% half or more, 15% a strip (15–50%), 19% buried in a pile |
| Piles | 1,110 rune columns, 243 rune fans, 270 fanned bases, 363 tucked gear, 528 battlefield groups, 800 decks, 380 rune decks, 778 trash piles |
| On top | 620 dice, 302 hands, 111 counters, 50 markers |
| Printings | 1,178 of the gallery's 1,189 |

The face-down share matches the one real random sample: 34% of Barcelona's reviewed tracks were face-down sleeves ([M0 §5.6](m0-spike.md#56-the-held-out-test-barcelona)).

## Realism check

Each fully visible face-up card is cut out through its quad, with the box up to 2% off in scale and 1.5% in position, as a detector's would be, and named against the gallery. Card-level top-1 ([CSV](m1-synth-v0-check.csv), first version [CSV](m1-synth-v0-check-first.csv)):

| | Colour grid | dHash | Colour grid + dHash | Legends, colour grid |
|---|---:|---:|---:|---:|
| Synthetic, first version | 99.1% | 98.3% | 99.3% | 99.7% |
| **Synthetic, calibrated (4,788 cards)** | **95.7%** | **96.3%** | **98.3%** | **84.7%** |
| Real: Shenyang ([M0 §5.2](m0-spike.md#52-reviewed-set-2381-crops)) | 94.7% | 91.3% | 96.6% | 98.8% |
| Real: Los Angeles ([§5.4](m0-spike.md#54-a-second-camera-los-angeles)) | 69.5% | 84.1% | 81.0% | 16.0% |
| Real: Barcelona ([§5.6](m0-spike.md#56-the-held-out-test-barcelona)) | 80.5% | 93.0% | 95.9% | 51.3% |

The real rows are the reviewed crops of the bootstrap detector, which finds only isolated cards, so they compare with fully visible synthetic cards. What changed in calibration, each from something seen on the M0 broadcasts:

- **A physical print is not the digital art:** every card gets its own contrast, saturation, brightness and colour cast.
- **Foil:** 60% of legends (competitive legends are often foil or showcase printings), 10% of runes and 15% of other cards. The rainbow sheen is twice as strong as in the M0 simulator, and the card's hue drifts.
- **Glossy sleeves:** 35% of cards catch a highlight, on sleeved cards mostly a long bright streak.
- **Dice** are 18–25 mm seen from above, a third of the card's width as on the broadcasts, with their sides showing. They sit over the art, and a quarter of legends with a die get a second one.

**Still easier than reality:** the colour grid names 85% of synthetic legends against 16–51% at Los Angeles and Barcelona. dHash and the pair are in Barcelona's range (94% against 92% and 96%). The real legends' printings seem to differ from the gallery more than a hue drift reproduces. Not simulated yet: player-camera video in the panels, lower thirds, motion between frames, and printings the gallery lacks (the tokens of [04 §4.2](../research/04-data-and-evaluation.md#42-catalogue)).

## What it is for

- **The amodal detector** (ARCHITECTURE §3.8): full quads of covered cards, with the visible fraction, and `card_back` for sleeve backs, with 16-bit id maps for masks.
- **The embedder** (§3.4): synthetic crops of every printing under the looks above, and strips of covered cards whose visible part the id map gives exactly.
- **Model selection stays on real data.** The synthetic set is for training. Every model is judged on the reviewed real sets of M0, which is where the calibration came from.

## Next

1. **Train the detector v0** on this set (GPU), and run it on the three M0 broadcasts. Its proposals for covered cards become a review pack, and those reviews are the first real stack labels.
2. **Fine-tune DINOv2-S** on synthetic crops with random covering, text scrambling and the foil above, scored on the real sets.
3. **Calibrate again** as those models show what the synthetic set still gets wrong, starting with the legends.

## Reproduce

```bash
cd ml && . .venv/bin/activate
python -m rifteye_ml.synth --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --boards 400 --previews 12 --seed 2026 --out ~/rifteye-data/synth/v0
python -m rifteye_ml.synth.check --run ~/rifteye-data/synth/v0 --catalog ~/rifteye-data/catalog/catalog.jsonl \
  --cache ~/rifteye-data/art --out m1-synth-v0-check.csv
```
