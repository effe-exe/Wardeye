# 02: The vision pipeline, designed from the constraints

This chapter explains *why* the pipeline in [ARCHITECTURE.md](../ARCHITECTURE.md) looks the way it does. Model choices and licences are in [03](03-models-and-licensing.md), and data and evaluation are in [04](04-data-and-evaluation.md).

## 2.1 Three problems of very different difficulty

| Sub-problem | Difficulty | Why |
|---|---|---|
| **Finding** cards | Low | High-contrast rectangles on a mat, a known aspect ratio, a mostly static camera. Detectors do this well even at small sizes. |
| **Naming** cards | High | Fine-grained, instance-level recognition across a few thousand printings, on small, compressed, glare-prone and often sleeved crops. |
| **Understanding** what happened | Medium | Needs zones, tracking, a model of the game, and care around camera cuts and replays. |

Most of the **accuracy risk** lives in naming. Most of the **product value** lives in understanding, the timeline. The plan spends effort accordingly: detection is kept simple and standard, identification gets the data engine, and the event engine starts as explicit rules that can be tested.

## 2.2 How big is a card on a stream?

This one number drives most design decisions. The assumptions: a Riftbound card is 63 × 88 mm, each player has a standard 610 × 356 mm (24 × 14 in) playmat, and the video is 16:9. The table gives the height of a card, in pixels, under common overhead framings.

| Framing | Overhead cam occupies | 1080p | 720p | 480p |
|---|---|---|---|---|
| Players top and bottom, both mats + 10% margin | full screen | 121 | 81 | 54 |
| | 60%-width window | 73 | 49 | 32 |
| Players top and bottom, tight on both mats | full screen | 133 | 89 | 59 |
| | 60%-width window | 80 | 53 | 36 |
| Players left and right, both mats + 10% margin | full screen | 142 | 94 | 63 |
| | 60%-width window | 85 | 57 | 38 |
| Wide table shot, 1.5 m across | full screen | 113 | 75 | 50 |
| | 60%-width window | 68 | 45 | 30 |

At those sizes the card's name is 4–7 px tall, and its collector number is smaller still. **Text is unreadable. Art is the signal.**

Working hypotheses, measured in milestone M0 ([04](04-data-and-evaluation.md#48-the-m0-feasibility-spike)):

- **H1.** At ≥ 100 px, a fine-tuned embedder identifies cards reliably from art alone.
- **H2.** At 60–100 px, it works with temporal aggregation plus game priors.
- **H3.** Below about 50 px, identity has to come mostly from priors, close-ups, production graphics and caster audio. Detection still works.

Design consequences:

1. The extension asks the viewer to switch to the source or 1080p rendition when `video.videoHeight < 1080`. That single setting can double the pixels per card.
2. **Identification crops come from the native-resolution frame.** The detector's downscaled input is only used to find boxes.
3. The embedder is trained for the 40–160 px range specifically, not for clean art.
4. Priors and non-visual evidence are first-class inputs from v1, not add-ons for later.

## 2.3 Why two stages, detect then identify

The alternative is a single detector with one class per card. It fails for four reasons:

- **The class count grows with every set.** A new set would mean relabeling and retraining a detector. With embeddings, a new set means embedding its catalogue art: new cards are recognised the day the catalogue updates, and accuracy improves after the next fine-tune.
- **Long tail.** Many cards are rarely played, so per-class detection data would be hopelessly imbalanced.
- **Different resolutions.** Detection needs spatial context at low resolution; identification needs every pixel of one card.
- **Prior art agrees.** Webcam and scanner card recognisers for other TCGs overwhelmingly use locate → rectify → match ([06](06-prior-art-and-starting-point.md)).

## 2.4 Rejected approaches, and where they still help

| Approach | Why it is not the main path | Where it still helps |
|---|---|---|
| **Vision-language model per frame** (Gemini or similar) | Cost and latency at 2–5 Hz per viewer; an open-source client cannot hold an API key; can invent cards that do not exist | Offline teacher in the VOD pipeline: pre-labels identities on clear close-ups for human review |
| **OCR of card names** | Names are 4–7 px tall on table cards | Production graphics, scoreboards, player name plates, close-ups |
| **Perceptual hashing** (pHash/dHash) | Brittle under glare, sleeves, perspective, codec noise and small scale | Exact matching of production graphics, which are clean digital renders; fingerprints for time sync ([ARCHITECTURE §7](../ARCHITECTURE.md#7-clocks-and-synchronisation)) |
| **Exact 4-corner keypoints from the detector, in the browser** | The permissive keypoint model (RF-DETR keypoint) is ViT-based and slow in browsers today ([03 §3.1](03-models-and-licensing.md#31-headline-findings)) | Server-side teacher and VOD pipeline. The browser uses oriented boxes plus an optional corner refiner |

## 2.5 Detector

- **Classes:** `card` (face-up) and `card_back` (face-down piles and cards). Exhausted versus ready, and whose side a card is on, come from geometry and layout. Fewer classes means cleaner labels.
- **Label convention:** boxes and quads follow the **printed card**, not the sleeve. A sleeve adds about 2 mm per side, roughly 3 px at 120 px card height. The rectifier learns to find the card inside it.
- **Hard negatives:** playmat art (it often contains illustrated rectangles), deck boxes, dice and counters, phones, sleeve packets, score trackers.
- **Overlap and stacks:** cards on battlefields and in the base overlap, runes lie in overlapping rows, and attached equipment is tucked under units by rule. Set-prediction detectors (the DETR family) handle crowded, overlapping objects without NMS tuning, which is one reason to prefer them. The detector is **amodal**: it predicts a covered card's full quad plus its visible fraction and edge, trained on synthetic stacks with full-quad ground truth ([ARCHITECTURE §3.8](../ARCHITECTURE.md#38-stacks-and-covered-cards)).
- **Model:** in the browser, RT-DETRv2-OBB-S (native oriented boxes) or D-FINE-S; on the server and as the labeling teacher, RF-DETR keypoint with four corners. All Apache-2.0 ([03 §3.2](03-models-and-licensing.md#32-detectors)).
- **Resolution:** detect on the overhead ROI resized to 640 px on the long side. At 1080p full-screen that keeps cards around 40 px, comfortably detectable. For windowed 720p layouts, tile the ROI in two.

## 2.6 Rectifier and orientation

- **An oriented box is usually enough.** Overhead table cameras have little perspective, so an affine warp from the detector's oriented box rectifies the card. The box's angle also encodes exhausted versus ready.
- **A corner heatmap network refines tilted views.** It has a small trunk, four heatmaps and soft-argmax, and predicts the four corners inside each box crop. The crop is taken from the full-resolution frame with a 15% margin. The network runs in single-digit milliseconds, and it is precise because heatmaps keep the spatial grid that global pooling throws away. It is retrained from a permissive initialisation ([03 §3.3](03-models-and-licensing.md#33-rectification)).
- The warp maps the crop to a canonical 176 × 246 portrait (63:88), or its transpose for landscape card types.
- **Orientation:** the opponent's cards are upside down to the camera, and exhausted cards lie sideways. The matcher batches all four 90° rotations of the crop through the embedder and keeps the best-scoring rotation. With a gallery of a few thousand vectors this is cheap. It also yields the card's rotation, which the event engine uses for exhausted and ready state.

## 2.7 Embedder

- **Input:** the rectified crop at low resolution (for example 128 × 176). Training at the target scale matches the domain and keeps inference fast.
- **Output:** a 256-d L2-normalised vector.
- **Backbone:** DINOv2 ViT-S/14 or Perception Encoder Core S16 (both Apache-2.0, about 40 MB at fp16) ([03 §3.4](03-models-and-licensing.md#34-embedders)).
- **Objective:** Sub-center ArcFace with one class per printing. It is strong on a closed gallery of a few thousand, and like face-recognition embeddings it still works open-set, so new cards match zero-shot from their catalogue art. The baseline to beat is InfoNCE/NT-Xent with a large negatives queue (MoCo-style), the recipe already proven on phone photos.
- **Positives** for a printing's catalogue art:
  1. Synthetic degradations of that art: downscale to 40–160 px, **real H.264 re-encode through ffmpeg** at stream bitrates with 4:2:0 chroma, sleeve border and glare, stage-light colour casts, motion blur, partial occlusion by hands and other cards, and small perspective.
  2. Real stream crops that the data engine has labeled ([04](04-data-and-evaluation.md)).
- **Partial views.** Covered cards show only a band along one edge. Training covers a random part of each crop, and the index stores strip views of every card (the band along each edge at a few visible fractions), so a covered card is matched on the part that shows.
- **Hard negatives.** Riftbound prints several cards around the same champion (legend, champion units, signature cards), with shared palettes and character designs. Mine them explicitly: they are the confusions that matter.
- **Model selection:** on the real held-out stream test set, every epoch. Never on loss.

## 2.8 From similarities to a committed identity

For a track with observations *i = 1..n*, each observation has a quality weight *wᵢ* (crop area × sharpness × (1 − glare fraction)) and cosine similarities *sᵢ(c)* to every gallery printing *c*:

```
S(c)        = Σᵢ wᵢ · sᵢ(c) / Σᵢ wᵢ                  # aggregated visual score
p_vis(c)    = softmax(S(c) / τ)                      # τ calibrated on real data
score(c)    = log p_vis(c) + λ₁·log p_zone(c) + λ₂·log p_domain(c)
            + λ₃·log p_decklist(c) + λ₄·log p_cost(c) + λ₅·log p_seen(c)
            + λ₆·log p_audio(c)
```

- `p_zone`: type consistency. Battlefields are the only landscape cards, and legends sit in the legend zone.
- `p_domain`: the legend's two domains. About a third of the pool survives ([01 §1.6](01-game-model.md#16-deck-rules-as-recognition-priors)).
- `p_decklist`: the published list, when there is one.
- `p_cost`: a card played right after *N* runes were exhausted most likely costs *N* energy ([01 §1.5](01-game-model.md#15-events-and-their-visual-signatures)).
- `p_seen`: already confirmed for this player in this match or event.
- `p_audio`: a caster mention within about ±10 s (server pipeline).

- **Commit rule:** the top card beats the runner-up by margin δ, over at least N observations, with at least one observation above a minimum crop size. τ, δ and N are tuned to reach **≥ 98% precision on committed identities** on the real test set. A wrong name is worse than "unknown card", so coverage is traded away for precision.
- **Priors are soft.** The domain prior multiplies out-of-domain cards by a small ε rather than zero. A misidentified legend then degrades gracefully instead of excluding the true card.
- **Printings roll up to cards.** p(card) = Σ p(printing). The timeline commits card-level identity even when the printing stays ambiguous (same art reprinted).
- Precision-coverage curves are reported for every model release, so the threshold choice is visible.

## 2.9 Tracking

- **Association:** two-pass IoU (high-score detections first, then low-score ones), ByteTrack-style. The overhead camera is static, so IoU alone handles almost all frames.
- **Re-ID for moves:** when a committed track vanishes and a new track appears elsewhere within a few seconds with the same identity (or a close embedding), stitch them together and emit `card_moved`.
- **Occlusion:** tracks persist through short occlusion by hands (a few seconds) without re-identification.
- **Covered cards:** a card covered by another card keeps its track, identity and last full quad for as long as its pile exists, and is re-confirmed whenever part of it shows. Zones hold ordered piles, so gear moves with its unit and a card vanishing from the middle of a pile is a removal ([ARCHITECTURE §3.8](../ARCHITECTURE.md#38-stacks-and-covered-cards)).
- **Camera cuts and zoom changes:** on a detected cut, all tracks freeze. When the overhead view returns, frozen tracks are re-associated by position and identity. If the framing changed (zoom or pan), the layout is re-estimated first.

## 2.10 Event engine

v1 is explicit rules over board-state diffs (the table in [ARCHITECTURE §3.7](../ARCHITECTURE.md#37-event-engine)). It is testable and explainable, and it can run on day one without training data. v2 can learn from the same diffs once enough human-logged timelines exist, for example a small sequence model over (board state, evidence) that predicts events. The v1 rules remain as the baseline.

Two broadcast behaviours need explicit handling:

- **Replays.** Production sometimes replays a key moment. The board then briefly shows a *past* state. The router flags replay transitions or graphics, and the engine suppresses events whose board state is inconsistent with the current one. The rule: board state may not regress.
- **Breaks, ads and side-by-side matches.** Some streams cut between two tables. Every event carries its `matchId`, layout presets can declare multiple tables, and tracks never cross tables.
- **Take-backs.** The rules let players reverse their most recent action. A card that appears and leaves again within seconds, without reaching the trash, produces a `retracted` event rather than two false ones.
- **Time-shifted broadcasts.** Official qualifier VODs air matches back-to-back after recording them. Match boundaries come from the scene router (a new pair of legends on the table) rather than from the clock.

**VOD mode is non-causal.** It sees the whole match. Each track's identity comes from its best frames, events can be back-filled once a card is identified later, and the timeline can be finalised after the match. **Live mode is causal**, with a 2–3 s confirmation buffer so tentative events do not flicker.

## 2.11 The other evidence channels

| Channel | Signal | Availability | Weight |
|---|---|---|---|
| **Production graphics** | The docked "featured card" render in official overlays, chosen by an operator when a card matters | Official and bigger broadcasts ([01 §1.8](01-game-model.md#18-the-broadcast-landscape)) | Near-certain identity, and a strong "this was played now" event. Also free training labels |
| **Close-up cams** | A card held to a camera, or a zoomed card cam | Some broadcasts | High-resolution identity |
| **Caster audio** | Casters name almost every card played | Nearly every broadcast | Strong prior within about ±10 s; also catches cards that never show clearly |
| **Published decklists** | The exact 40-card pool per player | Many competitive events, after the fact | Collapses the candidate set; best for VODs |
| **Chat** | Unreliable, spoilers | — | Not used |

Audio runs **server-side first** (VOD pipeline). ASR with vocabulary biasing toward card names, then fuzzy and phonetic matching of n-grams against the name list, gives time-stamped *mentions*. Mentions boost candidates in the fusion score and create low-confidence `spell_cast` candidates when nothing is visible. Casters work in many languages, so the name list must include localised names where they exist ([01](01-game-model.md)).

## 2.12 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Sleeve glare wipes out part of the art | Glare fraction lowers observation weight; aggregation over frames; glare in augmentation |
| Hands cover cards while playing | Track persistence; identity is committed after the hand leaves |
| Cards stacked or fanned on top of each other | Amodal detection, strip views in the index, covered-card persistence and piles in the board model ([ARCHITECTURE §3.8](../ARCHITECTURE.md#38-stacks-and-covered-cards)) |
| Low rendition (viewer on 480p or auto) | Prompt to switch; identity falls back to priors; overlay says "low quality" |
| Auto-exposure or focus hunting on the camera | Quality weight uses sharpness; skip frames below a floor |
| Picture-in-picture layout changes mid-event | Presets carry `validFrom`; auto-discovery re-runs after cuts |
| Replay or recap segments | Board state may not regress; replay detection in the router |
| Printing not yet in the catalogue (new promo, preview season) | "Unknown card" with a report button; catalogue refresh |
| Tokens, markers, proxies | Tokens in the catalogue as their own type; proxies are out of scope |
| Non-English cards | Localised printings grouped under one card; the art is shared |
| Two tables on one stream | Per-table layout regions; `matchId` per table |

## 2.13 What "good enough for v1" means

These are targets, to be confirmed or revised after M0. They are not claims.

| Metric (real stream test set) | v1 target |
|---|---|
| Detection recall, cards ≥ 50 px, IoU 0.5 | ≥ 97% |
| Committed-identity precision, card-level | ≥ 98% |
| Identity coverage, cards ≥ 80 px, after aggregation | ≥ 80% |
| `card_played` timeline F1, ±5 s tolerance | ≥ 0.85 |
| `spell_cast` timeline F1, ±5 s tolerance | ≥ 0.60 (audio and graphics make up the rest in VOD mode) |
