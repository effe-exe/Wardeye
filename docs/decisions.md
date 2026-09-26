# Decision log

Short records of settled decisions. The date is when a decision was taken, **Status** says whether it still holds, and the reasoning links to the research. To change a decision, open an issue and add a new entry that supersedes the old one. Never edit history.

---

### D-001: AGPL-3.0-only plus a CLA

- **Date:** 2026-09-26. **Status:** accepted.
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
- **Why:** tournament integrity and organiser trust. RiftEye must not become a stream-sniping aid.

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
- **Decision:** RiftEye publishes per-match timelines only. It does not compute, publish or retain play rates, win rates or matchup statistics of cards or decks.
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
- **Decision:** RiftEye does not use the Riot API.
  - **What RiftEye distributes:** its own code and models, plus an embedding index of vectors keyed by public collector codes (`OGN-001`, …).
  - **What it never distributes:** card images or card text.
  - **At display time:** the extension reads card names, text and images from Riot's public card gallery, in the viewer's browser.
  - **For Riftbound:** the project stays free and non-commercial, carries the Legal Jibber Jabber notice, and will act on any request from Riot.
- **Why:** the Maintainer's decision. Keeping every Riot asset out of RiftEye's distribution also keeps Riot's official gallery the single source of card data, which means official text and new sets on release day.
