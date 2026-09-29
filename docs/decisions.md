# Decision log

Short records of settled decisions. The date is when a decision was taken, **Status** says whether it still holds, and the reasoning links to the research. To change a decision, open an issue and add a new entry that supersedes the old one. Never edit history.

---

### D-001: AGPL-3.0-only plus a CLA

- **Date:** 2026-09-26. **Status:** accepted. Its commercial part (the dual-licence option) is superseded by [D-022](#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon).
- **Decision:** All code is licensed under the GNU AGPL-3.0-only. Contributors sign a CLA that grants the Maintainer the right to relicense, including commercially. The name and logo are covered by a separate [trademark policy](../TRADEMARKS.md).
- **Why:** the project should be genuinely open source and welcoming to contributors, while nobody can take it closed and sell it as a service, and the Maintainer keeps the option of a commercial dual-licence. See [07](research/07-licensing-and-governance.md).

### D-002: Permissive dependencies only

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** Shipped dependencies (code, WASM and **model weights**) must be under permissive licences: MIT, BSD, Apache-2.0, ISC, zlib or equivalent. No AGPL/GPL/LGPL code, no Ultralytics, no research-only or non-commercial weights.
- **Why:** the Maintainer can only offer a commercial licence for code the project owns or may sublicense. One AGPL or research-only dependency would quietly remove that option. See [03](research/03-models-and-licensing.md).

### D-003: Two stages, detect then identify by embedding

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** A small detector finds cards. A rectifier and embedder identify them by nearest-neighbour search over the catalogue. There is no per-card detector class and no per-frame VLM.
- **Why:** new sets become recognisable from their catalogue art, the long tail is handled, detection and identification each get the resolution they need, and embeddings cannot invent cards. See [02 §2.3–2.4](research/02-vision-pipeline.md#23-why-two-stages-detect-then-identify).

### D-004: Browser extension with local inference is the first public surface

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** The first product is a Manifest V3 extension that runs inference locally with ONNX Runtime Web. A Python VOD runner is built alongside it as the research and data engine. The web app and the broadcaster kit come later.
- **Why:** it works on any stream without the streamer's cooperation, at no per-viewer server cost, and without video ever leaving the user's machine. See [05](research/05-delivery-surfaces.md).

### D-005: Public information only

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** Hand cams, face-down cards and all other hidden information are masked out and never processed.
- **Why:** tournament integrity and organiser trust. Wardeye must not become a stream-sniping aid.

### D-006: No third-party media in git

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** Frames, crops, VODs, audio, card images and datasets never enter the repository. Datasets are referenced by sha256 manifest and stored privately. Card art is loaded at runtime from allowed sources.
- **Why:** those files belong to broadcasters and to Riot Games. Git history is permanent. See [08](research/08-legal-and-policy.md).

### D-007: Card-level identity is the product metric

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** The timeline commits card identity (name and rules). A printing is shown only when the art differs, so it can actually be resolved. Headline metrics are card-level.
- **Why:** same-art reprints and language variants cannot be separated at stream resolution, and gameplay only needs the card.

### D-008: Game-agnostic core, Riftbound as the first game pack

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** Detection, rectification, embedding, tracking and fusion stay game-agnostic. Zones, event rules and priors live in a Riftbound game pack.
- **Why:** the same machinery applies to other paper TCGs later, and keeping game logic separate makes it testable.

### D-009: ONNX as the model interchange format

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** Models are trained in PyTorch and shipped as ONNX. They run on ONNX Runtime Web in the browser (fp16 or fp32 on WebGPU, uint8 on WASM), and on ONNX Runtime natively on servers and broadcaster PCs.
- **Why:** one artifact runs on every surface, and the runtime is MIT-licensed. WebGPU does not accelerate int8, hence the precision split ([03 §3.5](research/03-models-and-licensing.md#35-browser-runtimes)).

### D-010: RiftEye is a working name

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** "RiftEye" is used until a final, game-agnostic name has cleared a trademark search. The final name is chosen before any public launch.
- **Why:** "Rift" leans on Riot's marks, a competing tool is called RiftSight, and an unrelated "RiftEye" already exists ([08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)).

### D-011: Card data and art come only from the Riot API

- **Date:** 2026-09-26. **Status:** superseded by [D-015](#d-015-no-riot-api-no-riot-assets-distributed).
- **Decision:** The product catalogue (card data, text and art) is built from the Riot API. A small server holds the app-specific key, and clients fetch from that server. Scraped galleries and third-party image mirrors are never shipped.
- **Why:** Riot's Riftbound policies allow only "Riftbound assets (including cards) provided by the Riot API". They require an app-specific key, and the key may not be in distributed code ([08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)).

### D-012: No cross-match statistics

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** Wardeye publishes per-match timelines only. It does not compute, publish or retain play rates, win rates or matchup statistics of cards or decks.
- **Why:** Riot will not approve apps that "publish or retain metagame-defining data". Revisit only with Riot's written confirmation.

### D-013: Browser detector is a small CNN or OBB model; RF-DETR runs on the server

- **Date:** 2026-09-26. **Status:** accepted, to be confirmed by the M1 benchmarks.
- **Decision:** In the browser, start with RT-DETRv2-OBB-S, with D-FINE-S plus the corner refiner as fallback; both are Apache-2.0. On the server and as the labeling teacher, use RF-DETR keypoint (4 corners) or RF-DETR-Seg.
- **Why:** ViT-backbone DETRs are currently slow and fp16-fragile in browsers ([03 §3.2](research/03-models-and-licensing.md#32-detectors)).

### D-014: Broadcaster kit is a sidecar process, not a native OBS plugin

- **Date:** 2026-09-26. **Status:** accepted.
- **Decision:** The kit talks to OBS through obs-websocket and delivers graphics as a Browser Source.
- **Why:** OBS and its plugin template are GPL, and a native plugin would have to be GPL-compatible, which conflicts with D-001's dual licensing ([05 §5.4](research/05-delivery-surfaces.md#54-broadcaster-kit)).

### D-015: No Riot API; no Riot assets distributed

- **Date:** 2026-09-26. **Status:** accepted. Supersedes D-011.
- **Decision:** Wardeye does not use the Riot API.
  - **What Wardeye distributes:** its own code and models, plus an embedding index of vectors keyed by public collector codes (`OGN-001`, …).
  - **What it never distributes:** card images or card text.
  - **At display time:** the extension reads card names, text and images from Riot's public card gallery, in the viewer's browser.
  - **For Riftbound:** the project stays free and non-commercial, carries the Legal Jibber Jabber notice, and will act on any request from Riot.
- **Why:** the Maintainer's decision. Keeping every Riot asset out of Wardeye's distribution also keeps Riot's official gallery the single source of card data, which means official text and new sets on release day.

### D-016: A change gate decides when and where the heavy stages run

- **Date:** 2026-09-26. **Status:** accepted. The prototype is in `ml/rifteye_ml/changegate.py`; M2 ports it to the extension.
- **Decision:** A layer-1 change gate watches a small view of the table against a model of the still table and reports settled changes with a box and a kind ([ARCHITECTURE §3.1.1](ARCHITECTURE.md#311-change-gate-layer-1)). The detector and embedder run on those boxes; full detection passes run only at the start, after cuts and periodically as a safety net.
- **Why:** Proposed by the Maintainer. Most frames change nothing, and the gate costs a tiny fraction of a detection pass. It also times events to the moment a card settles. On 10 minutes of the M0 reference VOD it reported 57 events. The Maintainer reviewed all of them: 44 (77%) were real card changes, and the kind was right for 36 ([M0 report §6](reports/m0-spike.md#6-layer-1-the-change-gate)).

### D-017: The embedding index holds a gallery pyramid

- **Date:** 2026-09-26. **Status:** accepted for pretrained encoders. Revisit after the M1 fine-tune.
- **Decision:** The index stores each printing's embedding at several on-screen sizes (for example 40–160 px long side), and the matcher searches the level nearest each crop's size. The detector already knows the size.
- **Why:** In the M0 spike, frozen DINOv2-S found 13% of clean 40 px cards against a single sharp gallery and 62% against the gallery rendered at 40 px. A fine-tuned embedder may shrink the gap. The pyramid costs a few extra MB of float16.

### D-018: Labels come from reviewing model proposals

- **Date:** 2026-09-26. **Status:** accepted. Tools: `apps/reviewer` and `ml/rifteye_ml/reviewpack.py`.
- **Decision:** Identity and event labels are made by a person answering correct or wrong to the model's guess, with the next guesses and a catalogue name search for corrections. Nobody labels from scratch. Items are tracks of one physical card, so one answer labels many crops. Each pack puts the least confident items first and mixes in a random 10% audit of the confident rest. Bulk training data stays synthetic ([04 §4.3](research/04-data-and-evaluation.md#43-synthetic-board-generator-mlsynth)). Reviewed real labels are for evaluation, calibration and a small real share in fine-tuning.
- **Why:** Proposed by the Maintainer, who asked for thousands of labels without labelling thousands of crops. Confirming a guess takes about two seconds. On the M0 reference VOD the colour-grid matcher is right on 97.5% of already-labelled tracks, so most answers are a single key. The audit measures the accuracy of the guesses nobody checks, which is what allows skipping review for confident items later. Jev (closed, hosted) and Laya Vision (non-commercial weights) cannot be the proposer ([03 §3.9](research/03-models-and-licensing.md#39-evaluated-typed-decision-models-laya)). The proposer is Wardeye's own pipeline, and it improves as the answers come back.


### D-019: The first identifier scores colour and structure together

- **Date:** 2026-09-27. **Status:** accepted for the first extension build. Revisit when the fine-tuned embedder beats it on the real sets.
- **Decision:** The matcher scores each printing by the mean of two cosines: the 16×16 colour grid and the 16×16 difference hash (dHash), both with 3% trimmed off every edge. That is one vector per printing, the two side by side and each scaled by √½ (`colorgrid/trim0.03+dhash/trim0.03`), so the index, the gallery pyramid and the four-turn search stay as they are ([D-017](#d-017-the-embedding-index-holds-a-gallery-pyramid)).
- **Why:** The two fail on different cards. On the M0 reference VOD (Shenyang) the colour grid names 94.7% of the reviewed crops and dHash 91.3%; together they name 96.6%, and 96.3% at 720p. On a second broadcast (the Los Angeles RQ), where players keep a die on their legend over its art, the colour grid names 69.5%, dHash 84.1% and the pair 81.0% ([M0 §5.4–5.5](reports/m0-spike.md#54-a-second-camera-los-angeles)). Both are a few kilobytes of arithmetic per crop, with no model to download. The equal weight was chosen on Shenyang only. On the Barcelona Regional, which nothing was tuned on, the pair names 95.9% against 80.5% and 93.0% for its parts, and Barcelona's own best weight (0.6) is within half a point. Los Angeles, with the most dice, would prefer more structure (0.75); a fourth broadcast can settle 0.5 against 0.6.

### D-020: The product is called Wardeye

- **Date:** 2026-09-29. **Status:** accepted; the Maintainer chose it. A trademark search must clear it before the public launch. Supersedes the working name of [D-010](#d-010-rifteye-is-a-working-name). Its look (Gradeon's tokens and Space Mono) is superseded by [D-024](#d-024-wardeye-has-its-own-brand-book).
- **Decision:** "Wardeye" replaces "RiftEye" in everything a user sees: the extension, the README and the store listing.
  - The code's internal names (`rifteye_ml`, `@rifteye/*`, `RIFTEYE_*`) change in one mechanical pass before the repository goes public.
  - The extension takes Gradeon's look: the near-black palette, the brand purple `#6153CC` and Space Mono (SIL Open Font License).
  - The README credits the Maintainer, who also makes Gradeon.
- **Why:**
  - In MOBAs a ward is what gives you vision, so players get the nod. Yet "ward" is a plain English word (to watch over, to guard), not a Riot term. The name uses no Riot mark, as D-010 asked and as Riot's policy forbids ([08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)).
  - A web search on 2026-09-29 found nothing called Wardeye; the nearest are browser security extensions called Ward and Warden. That is not a trademark search.
  - Also considered: TableLoupe (a card grader's magnifier), and names from League such as "Summoner Eye", which Riot's policy rules out.

### D-021: Apply to Riot for a Riftbound app key

- **Date:** 2026-09-29. **Status:** proposed. The application is drafted in [riot-application.md](riot-application.md). Amended by [D-024](#d-024-wardeye-has-its-own-brand-book): Wardeye is never presented as Gradeon's, key or no key.
- **Decision:** Apply as a free spectator companion. Once a key is approved:
  - it supersedes [D-015](#d-015-no-riot-api-no-riot-assets-distributed): card data and art come from the Riot API, through a small server that holds the key (as [D-011](#d-011-card-data-and-art-come-only-from-the-riot-api) had it);
  - the extension may be presented as Gradeon's.
  Until then the project stays a free, non-commercial community project, and the README only credits the Maintainer.
- **Why:** Riot's policy counts "any Project that involves a business or legal entity" as commercial, even a free one. Its route for a commercial project without a written licence is an approved API key (LJJ §2; [08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)).

### D-022: Free for everyone, closed weights, a showcase for Gradeon

- **Date:** 2026-09-29. **Status:** accepted. Supersedes the commercial part of [D-001](#d-001-agpl-30-only-plus-a-cla) and the sustainability model in [07 §7.7](research/07-licensing-and-governance.md#77-sustainability-model). How it credits Gradeon is amended by [D-024](#d-024-wardeye-has-its-own-brand-book).
- **Decision:**
  - The project is free for everyone. It offers no commercial licences, no paid tiers and no paid services.
  - The code stays AGPL-3.0-only, and contributors still sign the CLA, which keeps the licensing in one hand. [D-002](#d-002-permissive-dependencies-only)'s permissive-only rule stays for the same reason.
  - The trained weights, and the data made from card art and broadcasts, are not published and are not open source. They ship only inside the extension.
  - Beyond its users, the project shows Gradeon's work on card recognition. The README credits the Maintainer and Gradeon; presenting the tool as Gradeon's waits for [D-021](#d-021-apply-to-riot-for-a-riftbound-app-key).
- **Why:** the Maintainer's decision. Free and non-commercial also fits Riot's rules: the Legal Jibber Jabber's non-commercial licence, and the developer policies' "Charge money for your app or provide exclusive access" on the list of things not to do.

### D-023: In the browser, the detector runs in float32 and the embedder in float16

- **Date:** 2026-09-29. **Status:** accepted.
- **Decision:** On WebGPU the extension runs the detector's float32 file, always, and the embedder's float16 file when the GPU has `shader-f16` (else float32, else WASM). The private package carries only those two files.
- **Why:** on an Apple-silicon Mac (Chrome 154, native WebGPU), the float16 detector decoded 34 of the check tile's 35 cards, with extra ones: its outputs drift enough to move cards. The float32 detector and the float16 embedder both match PyTorch (embedder cosine 0.99984). The float16 detector would have been faster (44.5 ms a 576 px tile); correct cards come first. The older WebGPU runtime (JSEP) is not used: its GridSample fails in float16.

### D-024: Wardeye has its own brand book

- **Date:** 2026-09-29. **Status:** accepted; the Maintainer's brand book (alpha, September 2026), kept as [assets/brand/README.md](../assets/brand/README.md). Supersedes the look in [D-020](#d-020-the-product-is-called-wardeye), and how [D-021](#d-021-apply-to-riot-for-a-riftbound-app-key) and [D-022](#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon) present Gradeon.
- **Decision:**
  - Wardeye has its own identity: the ward mark with a Space Grotesk wordmark; a violet primary (`#8B7CF6`) on near-black (`#0A0A0B`); Space Grotesk, Inter and JetBrains Mono; soft corners, hairline borders, restrained motion; a precise, calm voice. The one-liner is "Place the ward. See the table."
  - It inherits Gradeon's dark-first system, but it is a separate community project by the same maker, credited as "by Federico Vietti, who also makes Gradeon". It is never presented as a Gradeon product.
  - The code's internal names (`rifteye_ml`, `@rifteye/*`, `RIFTEYE_*`, `~/rifteye-data`) may stay, as the book allows; the mechanical pass that [D-020](#d-020-the-product-is-called-wardeye) planned before going public is no longer required. Public copy never says RiftEye.
  - Where the brand book's draft and the facts differ, the repository keeps the facts: the weights are not published (D-022), so no copy says "no closed models"; and public copy says "in MOBAs" where the book names League of Legends, which keeps Riot's names out of the brand.
- **Why:** one look across the extension, the tools, the README and the web page. And a community project by a person, not a company, fits Riot's Legal Jibber Jabber, which counts any project that involves a business as commercial.
