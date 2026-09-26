# M0 feasibility spike

**Date:** 2026-09-26. **Status:** real curve from one broadcast; the stacked-card strips and the quick fine-tune are still running and will be added here.

The question from [04 §4.8](../research/04-data-and-evaluation.md#48-the-m0-feasibility-spike): can RiftEye name Riftbound cards from their art at the size they appear on stream? Numbers only. No images of cards or streams are in this folder (D-006, D-015).

## Summary

- **Go for full-screen 1080p table cameras.** On 174 real crops from a tournament broadcast (47 cards, 5 matches, cards about 130 px long), a 16×16 colour grid with the sleeve edge trimmed names **97%** of isolated face-up cards (98% top-5).
- **Frozen general-purpose ViTs are the wrong tool.** On the same real crops, DINOv2-S names 34% and PE Core S16 12%. Under simulated camera conditions, no frozen ViT passes 55% even at 160 px. A trained embedder is still needed, for everything the real set does not yet cover.
- **The codec is not the bottleneck.** With clean crops, identity survives H.264 at 2–6 Mbps down to 40 px: the colour grid and dHash name 99–100% of cards.
- **What hurts is the camera and the table.** With tilt, lighting, defocus, occluders and detector error added, the best encoder drops to 51–71% (40–160 px). Stacks and occlusion are the next data problem ([ARCHITECTURE §3.8](../ARCHITECTURE.md#38-stacks-and-covered-cards)).
- **Language matters only to structure-based encoders.** Chinese printings searched against the English gallery cost dHash and the ViTs 3–8 points and the colour grid nothing.
- **Decisions:**
  1. The first identifier is the colour grid with a gallery pyramid, behind the change gate ([D-016](../decisions.md#d-016-a-change-gate-decides-when-and-where-the-heavy-stages-run), [D-017](../decisions.md#d-017-the-embedding-index-holds-a-gallery-pyramid)). It needs no neural model in the browser.
  2. The M1 embedder is **DINOv2-S**, fine-tuned. It beats PE Core S16 frozen at every size from 80 px up, in every setting.
  3. Training adds random covering (stacks) and text-box scrambling (languages).

## Setup

| | |
|---|---|
| Catalogue | Riot's public card gallery feed, 2026-09-26: 1,189 printings, 936 cards. Plus 371 Simplified Chinese printings of Origins with their own images |
| Simulator | Cards placed on a table at 40–160 px (long side) in a 1920×1080 frame, 36 per frame, sleeves and glare, sensor noise, then a real libx264 encode (veryfast, 2 s GOP, VBV) and decode. Single-threaded, so runs are bit-exact |
| Realism levels | **codec:** perfect crops, codec only. **camera:** adds tilt up to 15°, ±4° residual rotation, white balance ±8%, exposure and gamma drift, uneven light, defocus, occluders on 30% of cards and ±6% detector box error. The magnitudes are assumptions |
| Encoders | colour grid 16×16 and dHash 16 (baselines); DINOv2 ViT-S/14 and Perception Encoder Core S16, both frozen at 224 px |
| Gallery | A pyramid: every printing rendered at the query's on-screen size ([D-017](../decisions.md#d-017-the-embedding-index-holds-a-gallery-pyramid)) |
| Queries | 300 sampled printings per setting (seed 0), searched against the whole gallery; baselines also on all 1,189 |
| Metric | Card-level top-1, orientation unknown (all four 90° rotations searched). Printings roll up to cards |
| Hardware | A 4-core cloud CPU, shared with other jobs. Timings are not benchmarks |

## 1. Codec only: identity survives the stream

Card-level top-1, 4 Mbps ([chart](m0-codec-pyramid.svg), [CSV](m0-codec-pyramid.csv)):

| Encoder | 40 px | 60 px | 80 px | 120 px | 160 px |
|---|---:|---:|---:|---:|---:|
| colour grid 16×16 | 99% | 99% | 99% | 99% | 99% |
| dHash 16 | 100% | 99% | 100% | 100% | 100% |
| DINOv2 ViT-S/14, frozen | 57% | 74% | 88% | 94% | 96% |
| PE Core S16, frozen | 48% | 63% | 77% | 82% | 86% |

The information is there even at 40 px and 2 Mbps. Frozen ViTs, which encode what a picture shows rather than the exact picture, are worse than a hash at near-duplicate matching.

## 2. Camera realism: where it gets hard

Card-level top-1, 4 Mbps ([chart](m0-camera-pyramid.svg), [CSV](m0-camera-pyramid.csv)):

| Encoder | 40 px | 60 px | 80 px | 120 px | 160 px |
|---|---:|---:|---:|---:|---:|
| colour grid 16×16 | 51% | 61% | 64% | 68% | 71% |
| dHash 16 | 15% | 18% | 29% | 29% | 26% |
| DINOv2 ViT-S/14, frozen | 13% | 18% | 33% | 47% | 55% |
| PE Core S16, frozen | 14% | 23% | 25% | 35% | 39% |

Top-5 runs 10–20 points higher (colour grid: 66% at 40 px, 81% at 160 px), which is what the hover card's top-3 relies on while a card is uncertain ([ARCHITECTURE §3.5](../ARCHITECTURE.md#35-matcher-priors-and-fusion)). Not knowing a card's orientation costs 1–8 points against knowing it. Going from 2 to 6 Mbps moved top-1 by at most 1 point for the baselines and 3 for the ViTs (40–60 px, in an earlier run with a single sharp gallery); the full bitrate sweep for the baselines is still running.

## 3. The gallery pyramid

Camera level, the same 300 cards, card-level top-1 against one sharp gallery → against the pyramid ([CSV](m0-camera-sharp.csv)):

| Encoder | 40 px | 80 px | 160 px |
|---|---:|---:|---:|
| colour grid 16×16 | 49% → 51% | 65% → 64% | 71% → 71% |
| DINOv2 ViT-S/14, frozen | 2% → **13%** | 15% → **33%** | 45% → **55%** |
| PE Core S16, frozen | 2% → **14%** | 11% → **25%** | 31% → **39%** |

With clean crops (codec level, 60 cards) the effect is larger still: DINOv2-S goes from 13% to 62% at 40 px. A pretrained backbone sees a sharp 744 px card and a blurry 40 px crop as different images, and rendering the gallery at the size a card appears on screen fixes most of that. The detector knows the size ([D-017](../decisions.md#d-017-the-embedding-index-holds-a-gallery-pyramid)). The colour grid shrinks every image to 16×16 anyway, so it does not need the pyramid.

## 4. Other languages

The 371 Simplified Chinese printings of Origins, degraded at the camera level and searched against the English gallery (card-level top-1; the English column is the §2 run, a different card sample; [CSV](m0-camera-zh-pyramid.csv)):

| Encoder | 80 px, Chinese | 80 px, English | 160 px, Chinese | 160 px, English |
|---|---:|---:|---:|---:|
| colour grid 16×16 | 67% | 64% | 74% | 71% |
| dHash 16 | 21% | 29% | 23% | 26% |
| DINOv2 ViT-S/14, frozen | 27% | 33% | 47% | 55% |
| PE Core S16, frozen | 22% | 25% | 34% | 39% |

The text box is where printings of the same card differ. Colour ignores it; structure and semantics do not.

## 5. Real crops: the curve that decides

**Data.** VOD 2872785768, PlusRB's restream of the Riftbound Regional in Shenyang (2026-09-13). Full-screen overhead camera at 1080p60, cards 124–145 px long, mostly Simplified Chinese printings with English ones in the top cut. 38 overhead frames, about one every 3 minutes of table view, across 5 matches (Swiss R10 game 2, Swiss R11, quarterfinal, semifinal and final game 1).

- **Crops:** the bootstrap mat detector (`ml/rifteye_ml/matcrops.py`) finds isolated cards whose dark border stands out from the red mat, and straightens them. That gave 183 crops. 9 were card backs or sleeves and are excluded; 174 face-up crops remain, 118 physical cards, 47 distinct cards.
- **Labels:** candidates came from two encoders. The assistant confirmed each one by eye against the catalogue art, and the Maintainer labeled the 5 it could not identify (one was a sleeve). Then the Maintainer checked 15 labels (the 7 least certain plus 8 at random): 15 of 15 correct.
- **Frames and crops stay private.** Permission from the rights holders has not been requested yet, so these are internal feasibility numbers.

Card-level top-1 (top-5 in brackets). Orientation unknown, with the gallery pyramid at 120 and 140 px ([CSV](m0-real-en+zh-Hans.csv), [CSV](m0-real-en.csv)):

| Encoder | English + Chinese gallery | English gallery only |
|---|---:|---:|
| colour grid 16×16 | 93.7% (94.8%) | 93.7% (94.8%) |
| colour grid 16×16, 3% trimmed off every edge | **97.1% (97.7%)** | **97.1% (97.7%)** |
| dHash 16 | 91.4% (93.7%) | 91.4% (93.7%) |
| dHash 16, 3% trimmed | 93.1% (94.8%) | 93.1% (94.8%) |
| DINOv2 ViT-S/14, frozen | 33.9% (46.0%) | 35.1% (51.7%) |
| PE Core S16, frozen | 12.1% (23.6%) | 10.3% (22.4%) |

- **Trimming** the sleeve edge and the mat around the crop is worth 3 points for the colour grid.
- **The Chinese images add nothing** at card level here. At 130 px the text is not what separates cards. They still matter for what the hover card shows a Chinese viewer.
- **The frozen ViTs** are far behind on real crops, further than in simulation.

**What this set does not cover.** It is easier than a live table:
- **Isolated cards only:** the bootstrap detector skips stacked, overlapping, covered and light-bordered cards.
- **One camera and one production.**
- **Crops from a detector tuned to this mat.**
- **Label candidates from the same two encoders it scores.** The Maintainer's audit found no error, but hard cards were less likely to be crops at all.

Treat 97% as the ceiling for clean, isolated cards on a good camera, not as the product's accuracy.

## 6. Layer 1: the change gate

On 10 minutes of Swiss R11 game 1 (5 fps, a 320 px view), the gate reported 57 settled changes. A visual check of 40 found nearly all to be real board changes: cards played, removed, stacked or turned, rune changes, dice and counters. It ran faster than real time on the shared CPU, most of that decoding the 1080p60 video. Recall needs a logged timeline, which the Maintainer is recording ([ARCHITECTURE §3.1.1](../ARCHITECTURE.md#311-change-gate-layer-1)).

## What this means per broadcast framing

| Framing | Card size | Verdict |
|---|---|---|
| Full-screen overhead camera, 1080p | 110–140 px | **Go.** Colour grid plus pyramid for the first extension; the trained embedder for covered cards, other cameras and lighting |
| Picture-in-picture or 720p | 60–95 px | **Adjust.** Synthetic camera-level top-1 is 60–65% for the best encoder. Needs the trained embedder, priors and aggregation over frames (H2) |
| Smaller than 50 px | | **Priors first.** Identity has to come from priors, production graphics and close-ups (H3). Detection still works |

## Next

1. **A second broadcast** with a different camera and lighting, preferably English, and real crops of **stacked cards**.
2. **The M1 embedder.** Fine-tune DINOv2-S with random covering and text scrambling on a GPU, scored first on this real set as the first leaderboard row.
3. **The timeline** of Swiss R11 game 1, for change-gate recall and the first end-to-end test.

## Reproduce

```bash
cd ml && . .venv/bin/activate
python -m rifteye_ml.catalog fetch-feed --out ~/rifteye-data/catalog/feed
python -m rifteye_ml.catalog build --feed ~/rifteye-data/catalog/feed --out ~/rifteye-data/catalog/catalog.jsonl
python -m rifteye_ml.catalog download --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art
E="--encoder colorgrid --encoder dhash --encoder timm:vit_small_patch14_dinov2.lvd142m --encoder timm:vit_pe_core_small_patch16_384.fb@224"
python -m rifteye_ml.spike synthetic --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art $E \
  --realism camera --queries 300 --heights 40,60,80,120,160 --bitrates 4000 --gallery-scales 40,60,80,120,160 --out m0-camera-pyramid.csv
python -m rifteye_ml.spike synthetic ... --realism codec ... --out m0-codec-pyramid.csv
python -m rifteye_ml.spike real --catalog ~/rifteye-data/catalog/catalog-en+zh-Hans.jsonl --cache ~/rifteye-data/art \
  --crops ~/rifteye-data/real-crops/v1 --labels ~/rifteye-data/real-crops/v1/labels.csv $E --gallery-scales 120,140 --out m0-real.csv
```

Seeds are fixed and the codec pass is bit-exact, so synthetic runs reproduce exactly. The real set is private; its manifest is recorded with the leaderboard.
