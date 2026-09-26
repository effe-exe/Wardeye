# M0 feasibility spike

**Date:** 2026-09-26. **Status:** complete for one broadcast: synthetic curves, a reviewed real set of 2,381 crops, stacked-card strips, the bitrate sweep, a quick fine-tune and the change gate's precision. A second broadcast is next.

The question from [04 §4.8](../research/04-data-and-evaluation.md#48-the-m0-feasibility-spike): can RiftEye name Riftbound cards from their art at the size they appear on stream? Numbers only. No images of cards or streams are in this folder (D-006, D-015).

## Summary

- **Go for full-screen 1080p table cameras.** On **2,381 real crops** from a tournament broadcast (69 cards in 537 tracks, cards about 130 px long), a 16×16 colour grid with the sleeve edge trimmed names **94.7%** of isolated face-up cards (97.1% top-5); 93.7% counting each physical card once. The first, smaller set gave 97.1%; the reviewed set adds the hardest cards on purpose.
- **Labelling by review works.** The Maintainer answered 400 of the model's guesses in 13.6 minutes (0.64 s median), labelling 960 crops ([§5.2](#52-reviewed-set-2381-crops), [D-018](../decisions.md#d-018-labels-come-from-reviewing-model-proposals)). The misses cluster on special printings, probably foils.
- **Frozen general-purpose ViTs are the wrong tool.** On the reviewed real set, DINOv2-S names 32% and PE Core S16 13–15%. Under simulated camera conditions, no frozen ViT passes 55% even at 160 px.
- **Training on synthetic crops transfers to real ones.** A linear layer on frozen DINOv2-S, trained only on simulated crops of three sets, takes the real set from 36% to **72%**, and from 30% to 61% for cards of sets it never saw ([§8](#8-quick-fine-tune)). In simulation the same head lifts unseen cards to 47–95% (40–160 px). PE-S gains more in simulation at small sizes but reaches only 48% on the real crops.
- **The codec is not the bottleneck.** With clean crops, identity survives H.264 at 2–6 Mbps down to 40 px: the colour grid and dHash name 99–100% of cards. Across 2–6 Mbps, accuracy moves by 2 points at most, with or without camera effects.
- **What hurts is the camera and the table.** With tilt, lighting, defocus, occluders and detector error added, the best encoder drops to 51–71% (40–160 px).
- **Stacked cards need training.** From the top 40% of a card (what a stack leaves visible), the frozen models name 31–50% at 120 px; from a side strip, 11–33% ([§7](#7-stacked-cards-identity-from-a-strip)). Frozen DINOv2-S keeps as much from the top 40% as from the whole card.
- **Language matters only to structure-based encoders.** Chinese printings searched against the English gallery cost dHash and the ViTs 3–8 points and the colour grid nothing.
- **The change gate fires on real changes 77% of the time.** On 57 reviewed events, 44 were real card changes and 36 had the right kind ([§6](#6-layer-1-the-change-gate)).
- **Decisions:**
  1. The first identifier is the colour grid with a gallery pyramid, behind the change gate ([D-016](../decisions.md#d-016-a-change-gate-decides-when-and-where-the-heavy-stages-run), [D-017](../decisions.md#d-017-the-embedding-index-holds-a-gallery-pyramid)). It needs no neural model in the browser.
  2. **The M1 embedder is DINOv2-S, with PE Core S16 kept as the challenger.** DINOv2-S wins frozen everywhere and, with a trained head, on real crops (72% vs 48%). PE-S generalises better to unseen sets at 40–80 px in simulation only. M1 fine-tunes DINOv2-S first and re-runs PE-S once there are real small crops.
  3. Training adds random covering (stacks), text-box scrambling (languages) and foil-like colour shifts (special printings).
  4. Real labels come from reviewing model guesses ([D-018](../decisions.md#d-018-labels-come-from-reviewing-model-proposals)).

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

Top-5 runs 10–20 points higher (colour grid: 66% at 40 px, 81% at 160 px), which is what the hover card's top-3 relies on while a card is uncertain ([ARCHITECTURE §3.5](../ARCHITECTURE.md#35-matcher-priors-and-fusion)). Not knowing a card's orientation costs 1–8 points against knowing it.

**Bitrate.** All 1,189 printings, card-level top-1 for the baselines at 2, 4 and 6 Mbps ([CSV](m0-camera-baselines.csv), [CSV](m0-codec-baselines.csv)):

| Encoder, realism | 40 px | 60 px | 80 px | 120 px | 160 px |
|---|---:|---:|---:|---:|---:|
| colour grid, camera | 52 / 52 / 53% | 61 / 61 / 61% | 65 / 65 / 65% | 69 / 69 / 69% | 70 / 70 / 70% |
| dHash, camera | 12 / 14 / 14% | 19 / 18 / 19% | 21 / 21 / 22% | 23 / 23 / 22% | 23 / 24 / 23% |
| both, codec only | 100% at every size and bitrate |||||

Stream bitrates in the usual range barely matter. The camera and the card's size do.

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

### 5.1 First set: 174 crops

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

### 5.2 Reviewed set: 2,381 crops

**Data.** The same broadcast, now every overhead frame: 669 frames, one every 10 s. The bootstrap detector found 3,157 crops. Crops of one physical card in consecutive frames form a **track** (same place, similar look), which gives 862 tracks. None of the 159 tracks holding first-set labels mixed two cards.

**Review** ([D-018](../decisions.md#d-018-labels-come-from-reviewing-model-proposals), [apps/reviewer](../../apps/reviewer)). The colour grid proposed a card per track, with a confidence fitted on the first set. A pack of 400 tracks went to the Maintainer: the 360 least confident, plus 40 drawn at random from the confident rest as an audit. The Maintainer answered all 400 in 13.6 minutes, a median of 0.64 s per answer:

| | Tracks | Model right |
|---|---:|---:|
| Least confident 360, where decided | 338 | 91.7% (95% CI 88.3–94.2%) |
| Random audit of the confident rest | 40 | 100% (95% CI 91.2–100%) |
| Can't tell | 22 | 7 Mech tokens (missing from Riot's gallery), 9 face-down cards, 4 deck or sleeve art, 2 unclear |

- **Every miss was a different card**, never the right card in another printing. For 15 of the 28, the right card was among the next three guesses.
- **The misses cluster on special printings:** Morgana, Vindictive (overnumbered) 7 times, Irelia, Fervent (overnumbered) 4, Master Yi (alt art) 4, Chaos Rune (alt art) 2. These are likely foils, whose colours shift under stage light. That is the colour grid's weak spot and a case for the simulator.
- **Low confidence also flags "not a card I know":** all 7 tokens missing from the catalogue and all 9 face-down cards were in the least-confident pack.

**Labels.** The review answers label every crop of their tracks, and the first-set labels spread the same way through their tracks. Face-down cards, the missing token and unclear crops are set aside. That leaves **2,381 crops, 537 tracks and 69 cards.** Long tracks weigh more per crop, so the table also counts each track once.

Card-level top-1 (top-5 in brackets), orientation unknown, gallery pyramid at 120 and 140 px ([CSV](m0-real-v2-en+zh-Hans.csv), [CSV](m0-real-v2-en.csv)):

| Encoder | English + Chinese gallery | English gallery only |
|---|---:|---:|
| colour grid 16×16 | 90.6% (94.3%) | 90.6% (94.3%) |
| colour grid 16×16, 3% trimmed | **94.7% (97.0%)** | **94.7% (97.1%)** |
| same, each track once (first crop) | | 93.7% |
| dHash 16 | 88.2% (92.9%) | 88.1% (93.0%) |
| dHash 16, 3% trimmed | 91.4% (93.9%) | 91.3% (93.9%) |
| DINOv2 ViT-S/14, frozen | 31.8% (48.2%) | 31.7% (50.7%) |
| PE Core S16, frozen | 15.2% (25.7%) | 12.9% (24.1%) |

The review pack was chosen to be hard, so this set is harder than the broadcast as a whole. Among crops labelled in the first set, the trimmed colour grid is right on 96.3%; among crops labelled by review, 92.2%. The same limits as §5.1 apply: isolated cards only, one camera, one production.

## 6. Layer 1: the change gate

On 10 minutes of Swiss R11 game 1 (5 fps, a 320 px view), the gate reported 57 settled changes. It ran faster than real time on the shared CPU, most of that decoding the 1080p60 video. The Maintainer reviewed all 57 in the reviewer: the table before (when the region last matched the still table) and after, with the gate's guess of the kind ([CSV](m0-gate-r11g1.csv)):

| Gate said | Events | A real card change | Kind right |
|---|---:|---:|---:|
| a card was put here | 18 | 16 | 14 |
| a card was taken away | 14 | 13 | 11 |
| the card here changed | 25 | 15 | 11 |
| **all** | **57** | **44 (77%)** | **36 (63%)** |

- **Most false alarms are small "changed" regions:** 10 of the 13. Their median area is 0.64 of a card, against 1.9 for real changes.
- **A size threshold does not fix them.** Raising the minimum to half a card removes 4 false alarms and 6 real changes, because turned cards and counters also change small regions. That is work for a learned verifier on the before/after pair (the M2 experiment in [03 §3.9](../research/03-models-and-licensing.md#39-evaluated-typed-decision-models-laya)). These 57 events are its first labels.
- **For a trigger, 77% is enough.** A false alarm costs one detection on a small box, and the event engine only reports a change the detector and identifier confirm. **Recall** matters more, and needs a logged timeline, which the Maintainer is recording ([ARCHITECTURE §3.1.1](../ARCHITECTURE.md#311-change-gate-layer-1)).

## 7. Stacked cards: identity from a strip

A stack leaves part of each card visible. Card-level top-1 at 120 px, 4 Mbps, camera realism, orientation known, with the gallery rendered the same way (only the strip). 300 printings for all four encoders; all 1,189 for the baselines at 60–160 px gave the same picture ([CSV](m0-camera-strips-neural.csv), [CSV](m0-camera-strips-baselines.csv)):

| Encoder | whole card | top 40% | top 25% | left 30% |
|---|---:|---:|---:|---:|
| colour grid 16×16 | 70% | 50% | 32% | 33% |
| dHash 16 | 35% | 23% | 14% | 6% |
| DINOv2 ViT-S/14, frozen | 48% | **48%** | 24% | 20% |
| PE Core S16, frozen | 38% | 31% | 19% | 11% |

- **The top 40% carries the art**, and frozen DINOv2-S loses nothing to it. The colour grid loses 20 points.
- **Thin strips** (a quarter, or a side as in a fanned pile) cost every model half or more of its accuracy.
- The embedder has to be trained on covered cards (random covering), and the tracker has to keep an identity while a card is covered ([ARCHITECTURE §3.8](../ARCHITECTURE.md#38-stacks-and-covered-cards)).

## 8. Quick fine-tune

A proxy for the M1 fine-tune that fits a laptop CPU: a linear layer (dim × dim, starting as the identity) on the frozen backbone, trained with InfoNCE on synthetic camera-level crops of the Origins, Proving Grounds and Spiritforged sets (664 printings, 2 seeds, 5 sizes). It is scored on fresh crops of the **other** sets' 525 printings (Unleashed and Vendetta), which it never saw, searched against the whole gallery. Card-level top-1, frozen → with the head ([CSV](m0-adapter-dinov2.csv), [CSV](m0-adapter-pe.csv)):

| Backbone | 40 px | 60 px | 80 px | 120 px | 160 px |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-S/14 | 13 → 47% | 21 → 66% | 32 → 71% | 47 → **91%** | 55 → **95%** |
| PE Core S16 | 13 → **70%** | 19 → **82%** | 28 → **87%** | 37 → 89% | 47 → 93% |

150 printings per cell, so differences under about 10 points are noise. On the training sets' own printings the head reaches 69–98% (DINOv2-S) and 80–98% (PE-S). The unseen sets score 5–20 points lower, so most of what the head learns is the camera, and some is the cards it trained on. PE-S gains more at small sizes. A full fine-tune in M1 should do better than one linear layer.

**On real crops.** The same heads, still trained only on synthetic crops, scored on the 2,381 reviewed real crops (orientation unknown). Card-level top-1, frozen → with the head ([CSV](m0-adapter-real-dinov2.csv), [CSV](m0-adapter-real-pe.csv)):

| Backbone | all crops | each track once | cards from the training sets | cards from other sets |
|---|---:|---:|---:|---:|
| DINOv2 ViT-S/14 | 36 → **72%** | 38 → **72%** | 43 → **84%** | 30 → **61%** |
| PE Core S16 | 13 → 48% | 11 → 53% | 13 → 54% | 13 → 43% |

The frozen scores here are a few points above §5.2 because this run's gallery pyramid has different sizes: most crops meet the 120 px level here and the 140 px level there.

- **Synthetic training transfers to footage.** One linear layer trained only on simulated crops doubles DINOv2-S on real ones, including cards from sets it never trained on. That is the premise of the M1 fine-tune, now measured.
- **On real crops DINOv2-S is well ahead**, although PE-S generalised better at 40–80 px in simulation. The real crops are about 130 px, where the two tied in simulation. So the simulator still misses something that matters, and the small-size question needs real small crops.
- **Still below the colour grid** (94.7%) on this clean 1080p set. The trained embedder is for what the colour grid cannot do: covered cards, other cameras and lighting, small cards. It must beat the colour grid on those before it replaces it.


## What this means per broadcast framing

| Framing | Card size | Verdict |
|---|---|---|
| Full-screen overhead camera, 1080p | 110–140 px | **Go.** Colour grid plus pyramid for the first extension; the trained embedder for covered cards, other cameras and lighting |
| Picture-in-picture or 720p | 60–95 px | **Adjust.** Synthetic camera-level top-1 is 60–65% for the best untrained encoder; one trained linear layer reaches 66–87% on unseen sets. Needs the trained embedder, priors and aggregation over frames (H2) |
| Smaller than 50 px | | **Priors first.** Identity has to come from priors, production graphics and close-ups (H3). Detection still works |

## Next

1. **A second broadcast** with a different camera and lighting, preferably English, and real crops of **stacked cards**. Reviewed the same way, it takes a pack of about 400 answers.
2. **The M1 embedder.** Fine-tune DINOv2-S with random covering, text scrambling and foil-like colour shifts on a GPU, scored first on this real set as the first leaderboard row. PE-S gets a second look on real small crops.
3. **The timeline** of Swiss R11 game 1, for change-gate recall and the first end-to-end test.
4. **A gate verifier** trained on reviewed before/after pairs, once there are a few hundred.

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
# the reviewed set (ml/README.md, step 5): crops, a review pack, answers, labels
python -m rifteye_ml.reviewpack identity --crops ~/rifteye-data/real-crops/v2/crops ... --out identity-v2a.json
python -m rifteye_ml.reviewpack apply --pack identity-v2a.json --answers identity-v2a.answers.json --out labels-review.csv
python -m rifteye_ml.spike synthetic ... --strips top:0.25,top:0.4,left:0.3 --gallery-scales 120 --heights 120 --out m0-camera-strips-neural.csv
python -m rifteye_ml.adapter --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --encoder timm:vit_small_patch14_dinov2.lvd142m --train-sets OGN,OGS,SFD \
  --real-crops ~/rifteye-data/real-crops/v2/crops --real-labels ~/rifteye-data/real-crops/v2/labels.csv --out m0-adapter-real-dinov2.csv
```

Seeds are fixed and the codec pass is bit-exact, so synthetic runs reproduce exactly. The real set is private; its manifest is recorded with the leaderboard.
