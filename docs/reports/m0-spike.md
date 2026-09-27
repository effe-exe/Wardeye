# M0 feasibility spike

**Date:** 2026-09-26, updated 2026-09-27. **Status:** complete for two broadcasts: synthetic curves, reviewed real sets from Shenyang (2,381 crops) and Los Angeles (1,383), stacked-card strips simulated and cut from real cards, the bitrate sweep, a quick fine-tune and the change gate's precision. A third broadcast (Barcelona) is the held-out test, in review.

The question from [04 §4.8](../research/04-data-and-evaluation.md#48-the-m0-feasibility-spike): can RiftEye name Riftbound cards from their art at the size they appear on stream? Numbers only. No images of cards or streams are in this folder (D-006, D-015).

## Summary

- **Go for full-screen table cameras, scoring colour and structure together.** On **2,381 real crops** from the Shenyang Regional (69 cards in 537 tracks, cards about 130 px long), a 16×16 colour grid with the sleeve edge trimmed names **94.7%** of isolated face-up cards; scored together with a difference hash, **96.6%** ([§5.5](#55-colour-and-structure-together)). At 720p (about 87 px), 94.4% and 96.3% ([§5.3](#53-the-same-cards-at-720p)).
- **A second broadcast found what one camera could not.** On 1,383 reviewed crops from Riot's own stream of the Los Angeles RQ (cards about 166 px), the colour grid names only 69.5%. Players there keep a die on their legend as a counter, mostly over the middle of its art, and legends are 30% of the crops; the colour grid names 16% of them. dHash names 84.1% of all crops, and the pair 81.0% with the weight set on Shenyang. Without the legends and one token printed with other art than the gallery's, the colour grid names 98.8% ([§5.4](#54-a-second-camera-los-angeles)).
- **The gallery lacks tokens that are played.** Half of the model's misses in the Los Angeles review were one missing token, the Sand Soldier (67 tracks); Shenyang had the Mech (7 tracks). They need a supplement ([04 §4.2](../research/04-data-and-evaluation.md#42-catalogue)).
- **Labelling by review works.** The Maintainer answered 400 of the model's guesses in 13.6 minutes at Shenyang and 18.2 at Los Angeles ([§5.2](#52-reviewed-set-2381-crops), [D-018](../decisions.md#d-018-labels-come-from-reviewing-model-proposals)).
- **Frozen general-purpose ViTs are the wrong tool.** On the reviewed real sets, DINOv2-S names 32% (Shenyang) and 42% (Los Angeles), PE Core S16 13% and 37%. Under simulated camera conditions, no frozen ViT passes 55% even at 160 px.
- **Training on synthetic crops transfers to real ones.** A linear layer on frozen DINOv2-S, trained only on simulated crops of three sets, takes the real set from 36% to **72%**, and from 30% to 61% for cards of sets it never saw ([§8](#8-quick-fine-tune)). In simulation the same head lifts unseen cards to 47–95% (40–160 px). PE-S gains more in simulation at small sizes but reaches only 48% on the real crops.
- **The codec is not the bottleneck.** With clean crops, identity survives H.264 at 2–6 Mbps down to 40 px: the colour grid and dHash name 99–100% of cards. Across 2–6 Mbps, accuracy moves by 2 points at most, with or without camera effects.
- **What hurts is the camera and the table.** With tilt, lighting, defocus, occluders and detector error added, the best encoder drops to 51–71% (40–160 px).
- **A real strip keeps most of a card's identity.** Cut from real crops of cards that go in stacks, the top 40% alone gives colour grid + dHash 94.5% (Shenyang) and 99.5% (Los Angeles), the top quarter 88.0% and 97.5%, a side strip 79% and 52% ([§7](#7-stacked-cards-identity-from-a-strip)). These are upper bounds: exact cuts of isolated cards, orientation known. Simulation had suggested far less (50% from the top 40% for the colour grid). Real stacks from footage are next.
- **Language matters only to structure-based encoders.** Chinese printings searched against the English gallery cost dHash and the ViTs 3–8 points and the colour grid nothing.
- **The change gate fires on real changes 77% of the time.** On 57 reviewed events, 44 were real card changes and 36 had the right kind ([§6](#6-layer-1-the-change-gate)).
- **Decisions:**
  1. The first identifier scores the colour grid and dHash together, with a gallery pyramid, behind the change gate ([D-016](../decisions.md#d-016-a-change-gate-decides-when-and-where-the-heavy-stages-run), [D-017](../decisions.md#d-017-the-embedding-index-holds-a-gallery-pyramid), [D-019](../decisions.md#d-019-the-first-identifier-scores-colour-and-structure-together)). It needs no neural model in the browser. Legends come from context: one per player, all game, in a fixed zone.
  2. **The M1 embedder is DINOv2-S.** It wins frozen everywhere and, with a trained head, on real crops at 1080p and 720p alike (72% and 71%, against 48% and 49% for PE Core S16). PE-S generalises better at 40–80 px only in simulation.
  3. Training adds random covering (stacks), text-box scrambling (languages) and foil-like colour shifts (special printings).
  4. Real labels come from reviewing model guesses ([D-018](../decisions.md#d-018-labels-come-from-reviewing-model-proposals)).

## Setup

| | |
|---|---|
| Catalogue | Riot's public card gallery feed, 2026-09-26: 1,189 printings, 936 cards. Plus 371 Simplified Chinese printings of Origins with their own images |
| Simulator | Cards placed on a table at 40–160 px (long side) in a 1920×1080 frame, 36 per frame, sleeves and glare, sensor noise, then a real libx264 encode (veryfast, 2 s GOP, VBV) and decode. Single-threaded, so runs are bit-exact |
| Realism levels | **codec:** perfect crops, codec only. **camera:** adds tilt up to 15°, ±4° residual rotation, white balance ±8%, exposure and gamma drift, uneven light, defocus, occluders on 30% of cards and ±6% detector box error. The magnitudes are assumptions |
| Encoders | colour grid 16×16 and dHash 16 (baselines), and the two scored together; DINOv2 ViT-S/14 and Perception Encoder Core S16, both frozen at 224 px |
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

**Foil.** The `foil` level adds a holographic sheen, a rainbow drifting across the card with a bright band where the light catches it, because foil printings were the colour grid's main misses on real footage (§5.2). The same 300 printings at the camera level, with no foil and with every card foil, card-level top-1 ([CSV](m0-foil-effect.csv)):

| Encoder | 80 px | 120 px |
|---|---:|---:|
| colour grid 16×16, 3% trimmed | 69% → 60% | 69% → 63% |
| DINOv2 ViT-S/14, frozen | 33% → 16% | 47% → 23% |

Foil halves what frozen DINOv2-S gets right. The trained embedder has to see foils. The sheen's strength is an assumption, like the other camera effects.

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

### 5.3 The same cards at 720p

Viewers often watch the 720p rendition, and the M0 set lacked small real crops. The same labelled boxes were cut, scaled by 2/3, from Twitch's 720p60 rendition of the same moments. Frames come from segments fetched with the same tool and exact trim as the 1080p files; seeking the HLS playlist directly landed several seconds off, by a different amount on each seek. The 720p frames match the downscaled 1080p frames at 47 dB PSNR (median). Cards are about 87 px long instead of 131. Card-level top-1, gallery pyramid at 80 and 90 px ([CSV](m0-real-v2-720p.csv)):

| Encoder | 1080p, ~131 px | 720p, ~87 px |
|---|---:|---:|
| colour grid 16×16 | 90.6% | 89.9% |
| colour grid 16×16, 3% trimmed | **94.7%** | **94.4%** |
| dHash 16, 3% trimmed | 91.3% | 89.1% |
| DINOv2 ViT-S/14, frozen | 31.7% | 23.8% |
| PE Core S16, frozen | 12.9% | 14.1% |

On a clean full-screen table camera, stream resolution is not what limits identification down to about 90 px. That matches the codec-only simulation (§1). The trained heads keep their level too (§8).

### 5.4 A second camera: Los Angeles

**Data.** VOD 2884665030, Riot's official English stream of the Los Angeles Regional Qualifier, Day 1 (2026-09-26). A navy playmat, another camera and production, English printings, cards about 166 px long. 617 overhead frames from three windows of the day, one every 10 s. The mat detector, with the mat colour taken from each frame (`--mask notmat`), found 2,345 crops in 541 tracks.

**Review.** The same pack design as §5.2: the 360 least confident tracks, plus 40 drawn at random from the other 181 as an audit. The Maintainer answered all 400 in 18.2 minutes (1.15 s median). Card level:

| | Tracks | Model right |
|---|---:|---:|
| Least confident 360, where decided | 337 | 61.1% (95% CI 55.8–66.2%) |
| same, only cards the gallery has | 271 | 76.0% (95% CI 70.6–80.7%) |
| Random audit of the confident rest | 40 | 97.5% (95% CI 87.1–99.6%) |
| Can't tell | 23 | 17 face-down cards (15 in purple sleeves, 4 of them under a Hidden marker; 2 card backs), 6 unclear |

Weighting each group by the tracks it stands for (360 and 181), the guesses are right on an estimated 73.8% of the broadcast's tracks, and on 85.5% of those whose card the gallery has.

**Three causes make 128 of the 132 misses:**
- **A token the gallery lacks:** the Sand Soldier, which Azir's legend makes, on 67 tracks. The audit's one miss was one too.
- **Legends under a die:** 52 tracks. Players kept a white die on their legend as a counter, mostly over the middle of its art. Legends never leave the table, so they are 30% of the labelled crops.
- **A token printed with other art:** Brush (UNL-T03), 9 tracks. The printed token's art is not the gallery's, so no encoder can match it.

The other 4 are two Order Runes, a Retreat and an Ahri. Two more guesses were the right card in another printing. Two answers were corrected by eye, since the crops show the title or emblem clearly: an Emperor of the Sands answered as Emperor's Dais (a battlefield with a similar name), and an Order Rune answered as Body Rune.

**Labels.** 1,383 crops, 310 tracks, 54 cards. Set aside: 284 crops of the missing token, 62 face-down, 40 unclear. As in §5.2, the pack picked the hardest tracks, so this set is harder than the broadcast.

Card-level top-1, orientation unknown, gallery pyramid at 140 and 160 px ([CSV](m0-real-la.csv)):

| Encoder | Shenyang (§5.2) | Los Angeles | LA legends, 412 crops | LA other cards, 971 |
|---|---:|---:|---:|---:|
| colour grid 16×16 | 90.6% | 61.5% | | |
| colour grid 16×16, 3% trimmed | 94.7% | 69.5% | 16.0% | 92.2% |
| dHash 16, 3% trimmed | 91.3% | **84.1%** | **69.9%** | 90.1% |
| colour grid + dHash ([§5.5](#55-colour-and-structure-together)) | **96.6%** | 81.0% | 52.2% | **93.2%** |
| DINOv2 ViT-S/14, frozen | 31.7% | 41.9% | | |
| PE Core S16, frozen | 12.9% | 36.8% | | |

- **Legends are the colour grid's blind spot here.** Many of the legend crops it does name show no die. A white die over the middle of the art shifts most of the grid's average colours. dHash compares the direction of brightness steps, and most of those lie away from the die. Dropping the worst-matching grid cells did not help: +1.8 points at best.
- **The other cards are easy.** Without the legends and the Brush token, the colour grid names 98.8% (895 of 906 crops).
- **The frozen ViTs do better here** than at Shenyang, on these larger cards, but stay far behind.
- **For legends, context beats pixels.** A player has one legend all game, in a fixed zone, and this broadcast prints both legends' names in its side panels. One good read per game is enough ([ARCHITECTURE §3.5](../ARCHITECTURE.md#35-matcher-priors-and-fusion)).

### 5.5 Colour and structure together

The colour grid and dHash fail on different cards, so the matcher can use both. Each printing gets one vector: the two side by side, each scaled by the square root of its weight. One dot product then gives the weighted mean of the two cosines, and the pyramid and the four-turn search are unchanged ([D-019](../decisions.md#d-019-the-first-identifier-scores-colour-and-structure-together)). Card-level top-1, orientation unknown ([CSV](m0-real-v2-en.csv), [CSV](m0-real-v2-720p.csv), [CSV](m0-real-la.csv)):

| Weight on dHash | Shenyang 1080p | Shenyang 720p | Los Angeles |
|---|---:|---:|---:|
| 0 (colour grid alone) | 94.7% | 94.4% | 69.5% |
| 0.25 | 96.2% | 95.7% | 70.2% |
| 0.4 | 96.4% | 95.9% | 74.2% |
| **0.5** | **96.6%** | **96.3%** | **81.0%** |
| 0.6 | 96.4% | 96.2% | 84.4% |
| 0.75 | 96.0% | 95.2% | 86.4% |
| 1 (dHash alone) | 91.3% | 89.1% | 84.1% |

- **The weight was chosen on Shenyang:** 0.5 is the best there at 1080p and at 720p, and every weight from 0.25 to 0.75 beats both encoders alone.
- **Los Angeles was not used to choose it.** It prefers more structure (86.4% at 0.75) because of its legends. The Barcelona test decides that without tuning on it.
- **It costs almost nothing:** two small grids per crop, and 1,024 numbers per printing and pyramid level.

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

A stack leaves part of each card visible. On both broadcasts, runes lie in overlapping columns that show the top of each rune, and units overlap at battlefields.

**Simulated.** Card-level top-1 at 120 px, 4 Mbps, camera realism, orientation known, with the gallery rendered the same way (only the strip). 300 printings for all four encoders; all 1,189 for the baselines at 60–160 px gave the same picture ([CSV](m0-camera-strips-neural.csv), [CSV](m0-camera-strips-baselines.csv)):

| Encoder | whole card | top 40% | top 25% | left 30% |
|---|---:|---:|---:|---:|
| colour grid 16×16 | 70% | 50% | 32% | 33% |
| dHash 16 | 35% | 23% | 14% | 6% |
| DINOv2 ViT-S/14, frozen | 48% | **48%** | 24% | 20% |
| PE Core S16, frozen | 38% | 31% | 19% | 11% |

- **The top 40% carries the art**, and frozen DINOv2-S loses nothing to it. The colour grid loses 20 points.
- **Thin strips** (a quarter, or a side as in a fanned pile) cost every model half or more of its accuracy.

**Cut from real cards.** The same bands cut from the reviewed real crops (§5.2, §5.4). Each crop is turned upright by its label, and only the band is scored, against the same band of every gallery card at the crop's size. Only cards that go in stacks count: everything but legends and battlefields. Card-level top-1 ([CSV](m0-real-strips-v2.csv), [CSV](m0-real-strips-la.csv)):

| Shenyang, 1,263 crops | whole card | top 40% | top 25% | left 30% |
|---|---:|---:|---:|---:|
| colour grid 16×16, 3% trimmed | 94.9% | 90.3% | 76.6% | 78.6% |
| dHash 16, 3% trimmed | 93.2% | 88.0% | 74.7% | 54.1% |
| colour grid + dHash | **97.9%** | **94.5%** | **88.0%** | **79.4%** |

| Los Angeles, 638 crops | whole card | top 40% | top 25% | left 30% |
|---|---:|---:|---:|---:|
| colour grid 16×16, 3% trimmed | 98.6% | 98.7% | 82.4% | **52.5%** |
| dHash 16, 3% trimmed | 98.3% | 96.6% | 69.6% | 13.2% |
| colour grid + dHash | **100%** | **99.5%** | **97.5%** | 52.0% |

Runes alone, as a rune column shows them (180 Los Angeles crops): colour grid + dHash names 98.3% from the top quarter, the colour grid alone 76.1%.

- **A real strip keeps most of a card's identity**, far more than the simulator's camera level suggested: from the top 40%, 90% for the colour grid against 50%. The camera level is harsher than both real tables; its magnitudes were assumptions (Setup).
- **Colour and structure together** lose the least from the top bands. A side strip, as in a fanned pile, holds mostly frame and text, and stays hard.
- **These are upper bounds for a stack.** Each band is cut exactly from an isolated card: no covering card, no shadow, no detection error, and the label gives the orientation. In a real stack the detector must find the visible part and which edge it is, and the matcher must still try both ways up ([ARCHITECTURE §3.8](../ARCHITECTURE.md#38-stacks-and-covered-cards)).
- **What is still missing is real stacks from footage** (Next, 2). The embedder is still trained with random covering, and the tracker still keeps a covered card's identity: most covered cards were seen whole when they were played.

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

At 720p (§5.3), the same heads give DINOv2-S 28 → **71%** (all crops) and 32 → 66% (cards from other sets), and PE-S 14 → 49% and 16 → 45% ([CSV](m0-adapter-real-720p-dinov2.csv), [CSV](m0-adapter-real-720p-pe.csv)). So the head's gain holds on smaller real cards, and PE-S's advantage at small sizes in simulation does not appear on real ones.

The frozen scores here are a few points above §5.2 because this run's gallery pyramid has different sizes: most crops meet the 120 px level here and the 140 px level there.

- **Synthetic training transfers to footage.** One linear layer trained only on simulated crops doubles DINOv2-S on real ones, including cards from sets it never trained on. That is the premise of the M1 fine-tune, now measured.
- **On real crops DINOv2-S is well ahead**, at 1080p and 720p alike, although PE-S generalised better at 40–80 px in simulation. The simulator still misses something that matters for PE-S.
- **Still below the colour grid** (94.7%) on this clean 1080p set. The trained embedder is for what the colour grid cannot do: covered cards, other cameras and lighting, small cards. It must beat the colour grid on those before it replaces it.


## What this means per broadcast framing

| Framing | Card size | Verdict |
|---|---|---|
| Full-screen overhead camera, 1080p | 110–170 px | **Go.** Colour grid and dHash with a pyramid for the first extension, legends from context; the trained embedder for covered cards, other cameras and lighting |
| Full-screen table camera at 720p | 85–95 px | **Go.** The same real cards score 96.3% at 720p against 96.6% at 1080p with colour and structure together (§5.3, §5.5) |
| Picture-in-picture | 60–85 px | **Adjust.** Synthetic camera-level top-1 is 60–65% for the best untrained encoder; one trained linear layer reaches 66–87% on unseen sets. Needs the trained embedder, priors and aggregation over frames (H2). No real crops at this size yet |
| Smaller than 50 px | | **Priors first.** Identity has to come from priors, production graphics and close-ups (H3). Detection still works |

## Next

1. **The held-out test:** the Barcelona Regional (PlusRB restream, the same broadcast package as Los Angeles). 482 table frames, 1,935 crops, 450 tracks; a 300-track random sample is with the Maintainer. It checks the colour-and-structure weight and the Regional Qualifier format without tuning on either.
2. **Real stacks from footage.** Both broadcasts show rune columns and units overlapping at battlefields. A covered card was usually seen whole when it was played, so its label can come from its own track before the cover: the change gate reports a card put down over an identified one, and the covered card's visible part is its last box minus the new card. What is left goes to review.
3. **Tokens and legends.** A token supplement (Sand Soldier, Mech, and the Brush as printed; [04 §4.2](../research/04-data-and-evaluation.md#42-catalogue)), and the legend prior in the matcher.
4. **The M1 embedder.** Fine-tune DINOv2-S with random covering, text scrambling and foil-like colour shifts on a GPU, scored first on these real sets (1080p and 720p) as the first leaderboard rows. It has to beat colour and structure together.
5. **The timeline** of Swiss R11 game 1, for change-gate recall and the first end-to-end test. The Los Angeles gate events (51) are also waiting for review.
6. **A gate verifier** trained on reviewed before/after pairs, once there are a few hundred.

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
python -m rifteye_ml.spike real --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --crops ~/rifteye-data/real-crops/v2/crops --labels ~/rifteye-data/real-crops/v2/labels.csv \
  --encoder colorgrid/trim0.03+dhash/trim0.03 --encoder 'colorgrid/trim0.03*1+dhash/trim0.03*3' --gallery-scales 120,140 --out m0-real-v2-en.csv
# the second broadcast: navy mat, so the mat colour comes from each frame
python -m rifteye_ml.matcrops --frames ~/rifteye-data/vods/2884665030/frames/seg-*/ --only oh-frames.txt \
  --table 0.19,0.06,0.81,1.0 --long 155 --mask notmat --mat-tol 45 --out ~/rifteye-data/real-crops/la-v1/crops
python -m rifteye_ml.reviewpack identity --crops ~/rifteye-data/real-crops/la-v1/crops ... --gallery-scales 140,160 \
  --temperature 0.0212 --max-items 400 --out identity-la-a.json
python -m rifteye_ml.spike real ... --crops ~/rifteye-data/real-crops/la-v1/crops --labels ~/rifteye-data/real-crops/la-v1/labels.csv \
  $E --encoder colorgrid/trim0.03+dhash/trim0.03 --gallery-scales 140,160 [--only-types Legend | --skip-types Legend] --out m0-real-la.csv
# strips: simulated, then cut from real cards that go in stacks
python -m rifteye_ml.spike synthetic ... --strips top:0.25,top:0.4,left:0.3 --gallery-scales 120 --heights 120 --out m0-camera-strips-neural.csv
python -m rifteye_ml.spike real ... --crops ~/rifteye-data/real-crops/la-v1/crops --labels ~/rifteye-data/real-crops/la-v1/labels.csv \
  --encoder colorgrid/trim0.03 --encoder dhash/trim0.03 --encoder colorgrid/trim0.03+dhash/trim0.03 --gallery-scales 140,160 \
  --strips full,top:0.4,top:0.25,left:0.3 --skip-types Legend,Battlefield --out m0-real-strips-la.csv
python -m rifteye_ml.adapter --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --encoder timm:vit_small_patch14_dinov2.lvd142m --train-sets OGN,OGS,SFD \
  --real-crops ~/rifteye-data/real-crops/v2/crops --real-labels ~/rifteye-data/real-crops/v2/labels.csv --out m0-adapter-real-dinov2.csv
```

Seeds are fixed and the codec pass is bit-exact, so synthetic runs reproduce exactly. The real set is private; its manifest is recorded with the leaderboard.
