# RiftEye research

> Written when the project was called RiftEye. It is now Wardeye ([D-020](../decisions.md#d-020-the-product-is-called-wardeye)).

How to build an open-source tool that watches Riftbound streams, logs every card played on a timeline, and lets viewers hover a card on the video to see it. What it takes, what already exists, and what constrains it. Researched in September 2026.

## Executive summary

1. **It is feasible, with the right expectations.**
   - Cards are about **70–140 px tall** in typical 1080p broadcasts, and 30–90 px at lower renditions ([02 §2.2](02-vision-pipeline.md#22-how-big-is-a-card-on-a-stream)).
   - Card text is unreadable at that size, so identification is **art-first**. Cards in every language are tournament-legal anyway.
   - Above about 100 px, art alone should identify reliably. Below that, context does the heavy lifting.
2. **Riftbound's rules are unusually helpful context** ([01](01-game-model.md)):
   - **Domains:** the legend's two domains cut the candidates to about **a third of the pool**.
   - **Decklists:** published lists cut them to **40 cards**.
   - **Rune taps:** the number of runes tapped reveals the cost of the card being played.
   - **Orientation:** at high-level events, ready cards face their controller and exhausted cards are all rotated the same way.
   - **Official broadcasts** already show a "featured card" graphic, which doubles as a free, exact label.
3. **The pipeline** ([ARCHITECTURE](../ARCHITECTURE.md), [02](02-vision-pipeline.md)):
   - scene router → oriented-box detector → rectification → embedding → gallery match with priors and per-track aggregation → tracker → event engine;
   - detect at low resolution, identify at native resolution, and identify each card once rather than every frame;
   - **every shipped model and library is Apache-2.0 or MIT** ([03](03-models-and-licensing.md)).
4. **Models** ([03](03-models-and-licensing.md)):
   - **Browser:** RT-DETRv2-OBB-S or D-FINE-S. ViT-based detectors are still slow in browsers.
   - **Server:** RF-DETR keypoint.
   - **Embedder:** DINOv2-S or Perception Encoder S with Sub-center ArcFace.
   - **Runtime:** ONNX Runtime Web on WebGPU with a WASM fallback.
   - **Excluded:** MobileCLIP (research-only weights), DEIMv2 (non-commercial since August 2026) and Ultralytics YOLO (AGPL).
5. **The first surface is a browser extension** doing local inference: no streamer cooperation, no per-viewer server cost, exact sync.
   - A Python VOD pipeline is built alongside it as the research and data engine.
   - Later: a web VOD library (timeline *beside* the player, as embed rules require), then a broadcaster sidecar plus Twitch Extension with an organiser partner ([05](05-delivery-surfaces.md)).
6. **Data** ([04](04-data-and-evaluation.md)):
   - Sources: synthetic boards with a **real H.264 codec pass**; footage obtained **directly from organisers, with permission**; and consented recordings of our own.
   - Measurement: test sets split by event; the product metric is timeline F1.
   - **A one-week feasibility spike comes first.** Earlier card work showed synthetic accuracy overstating real accuracy by tens of points.
7. **Licensing that keeps control** ([07](07-licensing-and-governance.md)):
   - AGPL-3.0-only;
   - a Harmony-based CLA that allows any outbound licence while keeping every contribution available under the AGPL;
   - a trademark policy;
   - permissive-only dependencies.
8. **Riot's policies are the biggest constraint** ([08](08-legal-and-policy.md)):
   - Riot's Riftbound policies ask apps for an app-specific Riot API key or a written licence, and allow only card assets that come from the API.
   - No cross-match "metagame" statistics.
   - Monetisation only with an approved key or a licence, a free tier and transformative paid content.
   - Riot also receives a broad licence to fan projects (LJJ §7).
   - **RiftEye does not use the Riot API** ([D-015](../decisions.md#d-015-no-riot-api-no-riot-assets-distributed)). It distributes no card images or text: the extension loads them from Riot's public card gallery in the viewer's browser. For Riftbound the project stays free and non-commercial.
9. **Competition exists**:
   - RiftSight already does hover, at a self-reported ~80% on in-person streams.
   - Riftbound Vision is in beta for OBS and Twitch.
   - RiftEye leads with the **timeline**, **measured accuracy** and **open tooling**, and should consider collaborating ([06](06-prior-art-and-starting-point.md)).
10. **The name "RiftEye" is a working name.** "Rift" leans on Riot's marks and sits close to RiftSight. Choose a neutral, game-agnostic name before any public launch ([D-010](../decisions.md#d-010-rifteye-is-a-working-name)).

## Chapters

| # | Chapter | Answers |
|---|---|---|
| 01 | [Riftbound as a CV problem](01-game-model.md) | Card pool, physical facts, zones, visible events, deck rules as priors, broadcast landscape |
| 02 | [Vision pipeline](02-vision-pipeline.md) | Scale analysis, why two stages, detector, rectifier, embedder, fusion maths, tracking, events, failure modes, targets |
| 03 | [Models and licensing](03-models-and-licensing.md) | Detector, embedder, runtime, tracker, OCR/ASR and labeling choices, each with code *and* weights licences |
| 04 | [Data and evaluation](04-data-and-evaluation.md) | Catalogue, synthetic generator, data engine, ground-truth timelines, splits, metrics, M0 spike, governance |
| 05 | [Delivery surfaces](05-delivery-surfaces.md) | Extension, web app, broadcaster kit, Twitch Extension, desktop app: capabilities, limits, costs, order |
| 06 | [Prior art and starting point](06-prior-art-and-starting-point.md) | Competitors, other TCGs, open-source recognisers, what the Maintainer's earlier work contributes |
| 07 | [Licensing and governance](07-licensing-and-governance.md) | AGPL + CLA + trademark, the alternatives, model and data licensing, how Riot's terms interact, changing licence later |
| 08 | [Legal and policy](08-legal-and-policy.md) | Riot's policies (quoted), platform terms, TDM law, personal data, AI Act, store policies, checklist |
| 09 | [Risks and open questions](09-risks-and-open-questions.md) | Risk register, open questions, kill criteria |

**How this was researched.** Four parallel research passes (game and policy, models and licences, delivery surfaces, licensing), cross-checked against primary sources: licence files, rules texts, specifications, source code and official pages. Some official pages were unreachable from the research environment. Those facts were taken from verbatim third-party copies or search extracts, and they are marked as such in the chapters. Re-verify them before relying on them.
