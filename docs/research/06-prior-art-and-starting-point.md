# 06: Prior art, competitors and starting point

## 6.1 Riftbound tools that already exist (September 2026)

| Tool | What it does | How | Relevance |
|---|---|---|---|
| **RiftSight** (riftsight.gg) | Hover any card on a YouTube or Twitch stream to see it. Chrome extension, plus a Twitch extension for RiftAtlas streams | The author reports tagging 2,000+ cards; self-reported accuracy about 80% on in-person streams and 90% on RiftAtlas; free, no ads | **Direct competitor for hover.** Shows the demand is real |
| **Riftbound Vision** (riftboundvision.com) | "Beta OBS plugin and Twitch extension for physical Riftbound TCG streams" | Not public | **Direct competitor for the broadcaster kit** |
| Rift Tap | NFC stickers in sleeves plus a Raspberry Pi reader drive an OBS overlay | Hardware; card art via the Riot API | Shows organisers want automated card graphics |
| Sideways Studio | Broadcast graphics package | Operator picks the featured card; "cards played" works only for games on the RiftAtlas simulator | Automated *physical* timelines are still missing |
| riftbound-obs-overlay, riftbound-live-overlay | Operator-driven OBS overlays | Manual | Same |
| **riftbound-scanner** (Nekoraru22) | In-browser card scanner | YOLO11s-pose (4 corners) trained on about 150 synthetic images per card, plus a 16×16 colour-grid fingerprint matched by cosine against about 950 cards; int8 ONNX about 2.5 MB | Good reference for the synthetic recipe. Its model is Ultralytics-trained (AGPL), so it is **not reusable** here ([03](03-models-and-licensing.md)) |
| Phone scanners (Rift TCG Scanner and others) | Collection scanning | Single card, close-up | Different problem |
| RiftAtlas, Pixelborn, Tabletop Simulator | Online play | Digital state, no vision | Out of scope, and Riot restricts simulators ([08](08-legal-and-policy.md)) |

**Where RiftEye differs:**

1. **The timeline.** Board state and events (played, cast, moved, turn, score), not just "what is this card". Nobody does this for physical play yet.
2. **Measured accuracy** on a real, event-split test set with a public protocol ([04](04-data-and-evaluation.md)), instead of self-reported numbers.
3. **Multi-signal fusion:** game priors, production graphics, caster audio ([02 §2.8–2.11](02-vision-pipeline.md#28-from-similarities-to-a-committed-identity)).
4. **Open code, models and tooling** under one coherent licence ([07](07-licensing-and-governance.md)).

The Riftbound community is small. Collaboration can beat competition here: shared layout presets, shared evaluation, even contributors. Worth reaching out to the authors of these tools early.

## 6.2 Other card games

| Game / tool | Surface | Technique | Notes |
|---|---|---|---|
| **SpellTable** (Magic, Wizards) | Web app for webcam play | Click a card in the webcam feed; matched against 17,000+ cards from a crop of a 720p feed | Needs a top-down camera and no glare. The closest analogue in the paper-TCG world, but for players rather than spectators |
| **Convoke** (Magic) | Twitch extension | Viewers click anywhere on the stream to identify a card | Click-to-identify, spectator side |
| **CardBoard Live** (Magic) | Broadcast platform | Viewers hover and expand cards in play and see decklists | Production-integrated |
| CardCast, InkwellOverlay (Lorcana, One Piece) | OBS overlays | Operator-driven | |
| Hearthstone Deck Tracker, Untapped.gg, LoR trackers | Twitch extensions | **Read the game client's state**, not pixels | The source of the sync pattern ([05 §5.5](05-delivery-surfaces.md#55-twitch-extensions)) |

## 6.3 Open-source card recognition

| Project | Approach | Lesson |
|---|---|---|
| magic_card_detector | Contours plus perceptual hash, still images | Works on clean, separated cards only |
| hj3yoo/mtg_card_detector | Tiny YOLO on about 40k synthetic layouts (88% on generated validation data), later replaced by contours + pHash; about 50 ms per card against 10k+ cards | **Failed on overlapping cards.** Synthetic validation numbers flatter |
| geaxgx/playing-card-detection | Synthetic playing-card dataset generator | Copy-paste recipe |
| mtgscan | Cloud OCR plus fuzzy matching against the card database | OCR needs readable text: not at stream scale |
| Pokémon and Yu-Gi-Oh scanners | Detection plus embedding plus similarity search | The same two-stage design RiftEye uses |

The synthetic-data recipe comes from a small canon: *Cut, Paste and Learn* (arXiv:1708.01642), *Domain Randomization* (arXiv:1703.06907) and *Simple Copy-Paste* (arXiv:2012.07177). **No peer-reviewed work on TCG recognition in broadcast video was found.** RiftEye's evaluation protocol could become one.

## 6.4 Starting point: the Maintainer's card-recognition work

RiftEye does not start from zero. The Maintainer's TCG pre-grading app, [Gradeon](https://gradeon.ai), already recognises single cards in phone photos across several games, Riftbound included. From that work, RiftEye inherits, re-published here under the AGPL as each piece is ported ([ROADMAP](../ROADMAP.md)):

- **Catalogue tooling:** bulk sync, image caching and incremental refresh. For RiftEye, card data and art must come from Riot's API ([08](08-legal-and-policy.md)).
- **A metric-learning trainer** for card-specific embeddings: contrastive loss with a negatives queue, augmentations modelled on the real domain gap, and model selection on real data every epoch.
- **A corner-heatmap rectifier** (soft-argmax) that exports to ONNX. It will be retrained from a permissive initialisation ([03 §3.3](03-models-and-licensing.md#33-rectification)).
- **Detector-training wrappers** around permissively licensed real-time DETRs.
- **Geometry utilities** (letterboxing, oriented boxes) in TypeScript.
- **An evaluation harness:** frozen test sets, sha256 dataset manifests, one scorer for every model, and a leaderboard that records git sha and dataset hash.
- **Labeling operations:** self-hosted CVAT, a quad-labeling tool with model prefill, adjudication and inter-annotator agreement.

**The lesson that shaped this plan.** On phone photos, an off-the-shelf embedder that looked excellent on synthetically distorted catalogue art lost tens of points of top-1 accuracy on real photos. Fine-tuning with real pairs recovered much of it. Stream frames are further from catalogue art than phone photos are, hence the insistence on real test sets and real training pairs ([04](04-data-and-evaluation.md)).

**What does not carry over:**

- Gradeon's users' photos, and anything trained on them.
- Model weights whose licence forbids redistribution or commercial use.
- Gradeon's services and keys.

## Sources

- riftsight.gg ; chromewebstore.google.com/detail/riftsight/ldnfcdenjcdhpefhggbbableckklgfmi ; riftboundvision.com
- github.com/nickorrnah/rift-tap ; github.com/sammor327/sideways-studio ; github.com/CaptainLoo/riftbound-obs-overlay ; github.com/atsadavut001/riftbound-live-overlay
- github.com/Nekoraru22/riftbound-scanner ; github.com/Teme1999/Riftbound-Card-Scanner ; riftatlas.com
- spelltable.wizards.com/faq ; magic.gg (SpellTable announcement) ; convoke.games/en/twitch-extension ; cardboard.live ; github.com/yzRobo/CardCast ; inkwelloverlay.com
- github.com/HearthSim/twitch-hdt-frontend ; Untapped.gg Twitch extension article
- github.com/tmikonen/magic_card_detector ; github.com/hj3yoo/mtg_card_detector ; github.com/geaxgx/playing-card-detection ; github.com/fortierq/mtgscan ; github.com/ArmanetPierre/pokemon-tcg-scanner ; github.com/dejaman/yugioh-ml-service
- arXiv:1708.01642, arXiv:1703.06907, arXiv:2012.07177
