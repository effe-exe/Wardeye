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

- **Date:** 2026-09-26. **Status:** superseded by the M2 benchmarks: the RF-DETR keypoint detector (v0) runs in the browser itself, in float32 on WebGPU ([D-023](#d-023-in-the-browser-the-detector-runs-in-float32-and-the-embedder-in-float16)). About 45 ms a 576 px tile on an Apple-silicon Mac was fast enough, and it was the model already trained. A smaller browser model stays an option for slow machines.
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

- **Date:** 2026-09-29. **Status:** accepted; the application was sent on 2026-09-30 and is in Riot's review ([riot-application.md](riot-application.md)). Amended by [D-024](#d-024-wardeye-has-its-own-brand-book): Wardeye is never presented as Gradeon's, key or no key. Amended by [D-025](#d-025-release-on-the-chrome-web-store-now-and-apply-to-riot-in-parallel): the release no longer waits for the answer.
- **Decision:** Apply as a free spectator companion. Once a key is approved:
  - it supersedes [D-015](#d-015-no-riot-api-no-riot-assets-distributed): card data and art come from the Riot API, through a small server that holds the key (as [D-011](#d-011-card-data-and-art-come-only-from-the-riot-api) had it);
  - the extension may be presented as Gradeon's.
  Until then the project stays a free, non-commercial community project, and the README only credits the Maintainer.
- **Why:** Riot's policy counts "any Project that involves a business or legal entity" as commercial, even a free one. Its route for a commercial project without a written licence is an approved API key (LJJ §2; [08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)).

### D-022: Free for everyone, closed weights, a showcase for Gradeon

- **Date:** 2026-09-29. **Status:** accepted. Supersedes the commercial part of [D-001](#d-001-agpl-30-only-plus-a-cla) and the sustainability model in [07 §7.7](research/07-licensing-and-governance.md#77-sustainability-model). How it credits Gradeon is amended by [D-024](#d-024-wardeye-has-its-own-brand-book). The extension's package is offered on GitHub too ([D-035](#d-035-each-releases-zip-is-on-github-too)).
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

### D-025: Release on the Chrome Web Store now, and apply to Riot in parallel

- **Date:** 2026-09-29. **Status:** accepted; the Maintainer's decision. Amends [D-021](#d-021-apply-to-riot-for-a-riftbound-app-key), whose release waited for Riot's answer.
- **Decision:** Riot's review can take months, so the in-browser build goes to the Chrome Web Store as a free alpha while the application is reviewed. The store build:
  - is standalone only: no companion mode and no access to `127.0.0.1`, which stays a developer tool built from the repository;
  - carries the models and the embedding index keyed by printing id, and no card image or card text ([D-015](#d-015-no-riot-api-no-riot-assets-distributed)): names, types and images come from Riot's public card gallery in the viewer's browser, as it watches;
  - keeps everything that makes it a fan project under the Legal Jibber Jabber: free for everyone, a community project by a person, no Riot logos, the notice, public information only.
  The application tells Riot the extension is live, and why.
- **Why:** a working release shows what Wardeye is better than a mock-up, and it can be withdrawn or changed if Riot asks. The risk, taken knowingly: Riot's Riftbound policy asks products that serve players to register, so launching first may bring a takedown request or a harder review ([08 §8.2](research/08-legal-and-policy.md#82-riot-games-policies)).

### D-026: Legends and published decklists narrow the search, and never reveal anything

- **Date:** 2026-09-29. **Status:** accepted; the Maintainer asked for the feature. Measured in the [decklist report](reports/m2-decklist-prior.md). The legend rule is in the live runner and the engine, on by default. Decklist import is built too: a box for each player in the extension's plays panel. Both ship with the extension's 0.2. Clarifies [D-005](#d-005-public-information-only).
- **Decision:**
  - **The legend rule, on by default.** Once a player's legend is pinned, the cards on that player's half compete only with the printings that fit the legend's domains, runes included. Battlefields and tokens are always allowed.
  - **Decklists, opt-in.** The viewer may paste a player's published list: a deck code, or a deckbuilder's text, tourney or JSON export. Wardeye fetches no list from any site.
  - **How a list is used.** A list is used only on the half whose pinned legend it names. Each listed card stands for all its printings, tokens are always allowed, and both lists' battlefields are allowed on both halves.
  - **What a list is for.** It only helps name cards already face up on the table. It is never shown, it is kept only for the match being watched, and it is never used to guess or reveal a hand, a face-down card or the rest of the list.
- **Why:** on Barcelona's hardest match, the legend rule names 97.8% of the cards on the table, against 91.8%. Players often play a printing other than the one listed (35% of the crops in the final). A wrong list used as a hard filter names 8.4%, and the legend guard removes that failure. A published list is public information. [D-005](#d-005-public-information-only) forbids hidden information: hands, face-down cards and what lies in a player's deck. The "deck contents" of the principles are those hidden cards, not a published list.

### D-027: Wardeye's own page at effe-exe.github.io

- **Date:** 2026-09-30. **Status:** accepted; the Maintainer's decision. The product URL of the Riot application ([D-021](#d-021-apply-to-riot-for-a-riftbound-app-key)).
- **Decision:** Wardeye's page is [effe-exe.github.io](https://effe-exe.github.io), a GitHub Pages site built from the repository [effe-exe/effe-exe.github.io](https://github.com/effe-exe/effe-exe.github.io). It is the product URL registered with Riot, and Riot's `riot.txt` check file lives at its root.
  - **What it is:** plain HTML in the brand book's look ([D-024](#d-024-wardeye-has-its-own-brand-book)), with no scripts and no trackers. Its illustration is drawn: no broadcast footage, card art or Riot logos ([D-006](#d-006-no-third-party-media-in-git)). Riot's notice is in the footer.
  - **The page at gradeon.ai/wardeye** stays the maker's other page about Wardeye ([brief](wardeye-page-brief.md)). Like everything else, it presents Wardeye as a community project, not a Gradeon product.
- **Why:** Riot checks that a product's site is the applicant's with a `riot.txt` file at the root of its domain. A GitHub Pages user site takes that file for free, where the Framer site needs a paid plan. It also keeps the registered URL off a company's site, as [D-024](#d-024-wardeye-has-its-own-brand-book) and the LJJ's business rule ask.

### D-028: The table's layout is a guide, not a rule

- **Date:** 2026-10-02. **Status:** accepted; the Maintainer's decision. Ships in 0.2.1, with the tracker fixes found on the Barcelona final.
- **Decision:** The official mat's layout guides a read and never throws one away. Each player has a base, a legend, a champion and a row of runes on their half, and the two battlefields lie on the midline between them. But cards go everywhere: a unit moves to a battlefield or changes hands (Akshan takes one), and mats vary a little from player to player.
  - **Battlefields belong to the strip.** The strip is the band along the table's midline, 12% of the table's width on either side of it (of its height, when the players sit top and bottom). A battlefield read in the strip is named and pinned as before. Off the strip, it is named only after three reads that agree, and never pinned: a rune or a unit turned sideways looks like a battlefield's landscape art. A pinned battlefield is moved, or merged with a second outline of itself, only in the strip.
  - **A named card stays itself.** A card's re-reads compare it with the cards its half allows and with its own name, so a unit that crosses to a battlefield, or is taken by the other player, keeps its name. The two players' cards are not pooled in the strip: a hand held over it would be named.
  - **Runes are counted, never named.** The Match tab shows how many runes each player has on the table, and how many are exhausted. From the player's seat, a ready rune points straight at them and an exhausted one lies across, along their edge. Battlefields are printed landscape and never exhaust, so a ready rune stands at right angles to the pinned battlefields; before one is pinned, the layout says where the players sit. Only runes are told ready or exhausted.
    - **Each frame** counts the boxes read as runes. A box counts as one when 30% of its reads say rune, whatever it is named: a foil rune is often read as another card.
    - **Stacks.** Runes lie in columns and fans, and the detector misses some of the strips a stack shows. Every card is the same size, so the count also takes the unread card-sized boxes beside a rune and turned its way. In a stack of three or more it finds the step from strip to strip (the gaps a fifth to a third of a card long), and a gap of 1.6 steps or more hides about gap ÷ step − 1 runes. This was the Maintainer's idea: the card's size tells how many strips a stack can hold.
    - **Over time.** Hands pass over the runes and the detector sometimes boxes two at once, so a player's count is the upper quartile of the last 12 s of frames, and the exhausted ones are the median of the last 3 s. The count holds while the camera is away.
    - **Measured** against the Maintainer's counts of 65 moments of the Barcelona and Los Angeles finals, two stretches of each: the count is right 43 times, against 30 for one per rune box seen, and off by 0.4 runes on average, against 1.3. On the stretches it was not tuned on, it is right 19 times of 31, against 10. The exhausted ones are right 30 times: the covered runes of a fan are counted but not told ready or exhausted, so on the Los Angeles final they come out short.
  - **Pasted lists are evidence too.** With both players' lists pasted, once one player's legend is a list's, the other player's cards are read against the other list, while their own legend is under a die or not read yet. With lists pasted, a legend on neither list is named only on a sure read. Lists for another match name neither legend on the table, so neither is used ([D-026](#d-026-legends-and-published-decklists-narrow-the-search-and-never-reveal-anything)).
- **Why:** The Barcelona final showed what a layout-blind tracker gets wrong: a rune turned sideways pinned as a battlefield for the rest of the game, and a battlefield drawn twice. The official mat's battlefield zone reaches 11% of the table's width either side of the midline. On the two finals measured, every battlefield's centre lay within 3% of the midline (Los Angeles 0.48–0.50 of the width, Barcelona 0.47–0.51), so the strip, at 12%, leaves room for a mat laid off centre. On the Los Angeles replay the extension is checked against, the cards, names and plays are the same as before. A strict model would fail the matches where players use the table their own way; a guide only adds evidence, and a card it does not expect is still named when the reads are sure.

### D-029: A card in a hand is not on the table

- **Date:** 2026-10-02. **Status:** accepted; the Maintainer's decision ("fix it first"). Ships in 0.2.1. Applies [D-005](#d-005-public-information-only). What counts as a hand is refined by [D-033](#d-033-a-hand-is-what-is-not-the-table) (0.2.2).
- **Decision:** A card is read only once it is put down: out of a hand for half a second, and still. A card never put down is neither drawn nor announced, and nothing of its face is looked at.
  - **A hand** is seen in a band around the card, outside its edges: 96 points at three distances, those inside the table window and off the other cards. A card with a tenth of them skin-coloured is in a hand, held or being put down. The card's own face is never looked at, since gold, faces and fire in the art are skin-coloured too. A card hemmed in by other cards shows no hand.
  - **Put down** means no hand around it for 0.5 s while it moves less than 4% of a card's length. Then it is read, and drawn from then on.
- **Why:** On the Barcelona final the players hold their hands low over the table, and the detector outlines the cards in them. Before the rule, the first 200 s listed 27 plays, 13 of them false: cards in a hand, a card turned in a hand, the same card twice. With it there are 17 plays, 3 of them false, and 13 of the 15 real plays are found (14 before). Of the three false plays left, one is a card held still in a hand for a moment, and two are cards announced a second time. On the Los Angeles final the rule removes two false plays and keeps every real one. A card in a hand is hidden information ([D-005](#d-005-public-information-only)): naming it would spoil the match.

### D-030: What stays put through the cuts is the broadcast's overlay

- **Date:** 2026-10-03. **Status:** accepted; the Maintainer's decision ("fix everything"), after testing 0.2.1 on a co-stream. Ships in 0.2.2. Applies [D-005](#d-005-public-information-only).
- **Decision:** What a broadcast lays over every shot (a co-streamer's webcam and chat, a scoreboard, a sponsor banner) is found from the cuts, and nothing on it is read.
  - **A cut** is a frame whose 96 x 54 thumbnail changed from the frame before by half. Play changes a quarter at most.
  - **The overlay** is the thumbnail's pixels that stayed through 90% of the cuts, once there have been four, in patches that reach the frame's edge, with their holes filled (a webcam's frame stays put, the face in it moves).
  - **What was overlay stays overlay** while it stays through 60% of the cuts: the face moving in a webcam changes its pixels at some cuts, and opens the webcam's frame to the table.
  - **Before four cuts,** what stayed through every cut so far is only suspect: nothing there is read until the cuts make it overlay, or for a minute after the last cut, since one cut alone can be the camera reframed. The frames the table was looked for in count too: a cut from a player cam to the table shows the webcam from the board's first frame.
  - **Once the overlay is known,** a box in it is not a card, the cards read off it before then are dropped (with a legend one of them gave its side), and their plays are withdrawn: the panel takes them off its list.
  - **The scene gate** leaves the overlay out of its score, and learns a frame only when the one before was the table camera too. How it scores is [D-034](#d-034-the-table-camera-is-told-by-the-table-window-block-by-block).
- **Why:** On a co-stream of a Riftbound tournament, recorded live, the co-streamer's webcam lay inside the table window: it was read as a legend (shown as one player's), a battlefield and a play, and the scoreboard's portraits and the chat were outlined. The scene gate, which scored the overlay with the table, took a close-up of a hand for the table 99 s in, learnt it, and from then on took every shot for the table. With these rules, on that game: four cards read on the webcam and the scoreboard before the fourth cut are withdrawn at it, 44 s into the video, and nothing on the webcam is read after; one graphic of the scoreboard is, at 165 s, a box wider than any card that the cuts never showed to stay put (the score track's numbers change). Before the rule that keeps what was overlay, the webcam left the overlay at the fifth cut, and the card on its wall was read and announced again 15 s later.

### D-031: A table camera shows both sides of a table

- **Date:** 2026-10-03. **Status:** accepted; the Maintainer's decision ("fix everything"). Ships in 0.2.2.
- **Decision:** The layout found from the footage (`autolayout`) is a table camera's only when its window holds five cards' lengths each way, cards lie on both sides of its middle (both players'), and they are not a graphic's grid: six or more cards all within 2 degrees of square and 3% of one size. Otherwise the look found nothing, and the table is looked for again a second later.
- **Why:** The layout is found once and kept for the whole video. Opened during a co-stream's pre-game close-up of a player shuffling, Wardeye took the close-up for the table (cards 304 px long, 3.3 cards high) and would have kept it all match; a sideboard screen's eight cards, in a grid, were read as eight plays. The table window holds 6.5 card lengths or more on every broadcast so far. On real tables the most crooked card lies 6 degrees off square or more and the sizes vary by 7% or more; on the sideboard screen every card was within 0.7 degrees and 1%.

### D-032: A card is a card's size, and not the mat's print

- **Date:** 2026-10-03. **Status:** accepted; the Maintainer's decision ("fix everything"). Ships in 0.2.2.
- **Decision:**
  - **A box longer than 1.45 cards is not one.** The detector outlines a co-stream's chat (two cards long) and, on a mat with printed card zones, a card and the empty zone beside it as one box.
  - **A zone printed on the mat is not a card.** A box of the mat's own colour inside as around it (their medians within 18 levels) and plain inside (the middle half of its points within 24) is a printed zone or the mat's logo. A card's face is never plain, and a face-down card is plain in its sleeve's colour.
  - **Four cards first named within a second are not four plays.** They are a graphic of cards or a view framed anew: nobody plays four cards a second. They are named, and not announced.
- **Why:** On the co-stream of [D-030](#d-030-what-stays-put-through-the-cuts-is-the-broadcasts-overlay) a box round a card and the empty zone above it was named as that card again and announced as a second play, and the zone a card was moved away from kept the card's name. Named cards on the two finals measure within 1.24 cards 99.5% of the time; the zone test flags none of 494 named cards checked on the Barcelona and Los Angeles finals, and 77 boxes on the co-stream's mats, all zones, the logo, or the spot a card left.

### D-033: A hand is what is not the table

- **Date:** 2026-10-03. **Status:** accepted; the Maintainer's decision ("fix everything"). Ships in 0.2.2. Refines [D-029](#d-029-a-card-in-a-hand-is-not-on-the-table).
- **Decision:** A point of the band around a card counts as skin only where the frame there is 40 levels unlike the still table: the median of the first five table frames half a second apart (hands move, the table does not), which then moves 2 levels nearer the frame every half second, so a hand passing over the table stays a hand and a card put down becomes part of the table within half a minute. The two are compared at a quarter of 1080p, so a mat's thin printed lines are the table too. Until the still table is known (2.5 s), the colour test alone decides, as before.
- **Why:** A wooden table is skin-coloured. On the co-stream of [D-030](#d-030-what-stays-put-through-the-cuts-is-the-broadcasts-overlay) every card beside the wood counted as held: both legends, a champion and the battlefields lying on the wood between the mats were never named. With the rule, both legends, both champions and a battlefield are named on that game.

### D-034: The table camera is told by the table window, block by block

- **Date:** 2026-10-03. **Status:** accepted; the Maintainer's decision ("fix everything"). Ships in 0.2.2. Applies [D-005](#d-005-public-information-only).
- **Decision:** The scene gate, which says whether a frame is the table camera, looks at the table window only, and in blocks.
  - **Only the table window is scored.** The panels beside it are laid over every shot, a close-up of a hand included.
  - **In blocks.** The score is the mean of the correlations in a 6 x 3 grid of blocks of the 96 x 54 thumbnail, each with 20 still pixels of the window or more: an arm or a banner over the table spoils a block or two, a cut to another shot spoils them all.
  - **Learnt again only with cards.** Away for 20 s, a view is learnt as the table camera moved only when it looks like the mat and holds five card-sized boxes or more, and half the most the table camera showed in its last minute on screen. A mat's colour alone is no proof. Only the boxes are counted: nothing on them is read.
- **Why:** Replayed on five stretches (Barcelona 0 to 200 s and 800 to 1000 s, Los Angeles 0 to 240 s, the game of the co-stream of [D-030](#d-030-what-stays-put-through-the-cuts-is-the-broadcasts-overlay)), 1,971 frames against shot labels checked by eye:
  - 0.2.1 took 488 frames of other shots for the table and lost none of it: Barcelona's close-ups of hands and of single cards, which the panels made look like the table, and on the co-stream nearly every other shot from 99 s on.
  - 0.2.2 before this rule took 434 and lost 87: on the co-stream it learnt a player cam as the table at 112 s, and on Barcelona an arm over the mat's printed score track took the table away for 9 s.
  - Now 9 and 8: in the first 6 s of a stretch, before the gate has learnt the table, and around cuts.
  - On the co-stream every shot passes for the mat by its colour (0.57 to 0.72 of the window, where 0.39 is enough), and the player cam the gate learnt at 112 s holds no card. Close-ups of one side of the table hold 13 card-sized boxes at most; the table holds 45 to 61.
  - Run on the stretches with the rest of 0.2.2: on Barcelona's first 200 s, 15 plays are announced, 12 of them real and 2 false, where 0.2.1 announced 17, 11 real and 3 false (checked against the frames). On the co-stream's game the gate is right on 687 frames of 688. On Los Angeles the plays are as before.
  - The rune count, which holds while the camera is away, is right 39 times of the 65 moments of [D-028](#d-028-the-tables-layout-is-a-guide-not-a-rule), against 43: on Barcelona's first 200 s it is one rune high at four moments more, most just after a close-up, whose empty frames had pulled 0.2.1's count down. The exhausted ones are right 30 times, as before.

### D-035: Each release's zip is on GitHub too

- **Date:** 2026-10-03. **Status:** accepted; the Maintainer's decision ("add the latest zip on GitHub for developers"). Refines [D-022](#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon).
- **Decision:** Every version's store zip is attached to its GitHub release. Unzipped and loaded in Developer mode, it is the extension the store gets, for developers and for anyone who wants a version before the store has approved it. It does not update itself.
  - **The models inside stay closed** ([D-022](#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon)): they are not under the AGPL, all rights are reserved, and they may be used only to run Wardeye. Each release's notes say so.
- **Why:** The store reviews every update, which takes days, and developers want the newest build, or a fixed copy of one. The models already reach everyone inside the store package, which anyone can download, so nothing new is exposed. Saying the terms where the zip is keeps the weights' status plain.

### D-036: A second outline of a card is no play

- **Date:** 2026-10-06. **Status:** accepted; the Maintainer's decision ("fix duplicates, measure both"). Ships in 0.2.3. Refines [D-032](#d-032-a-card-is-a-cards-size-and-not-the-mats-print).
- **Decision:** A hand passing over a card, or straightening it, can leave it outlined twice, and the second outline is named as the card too. The board already drew it once; now it is no play either:
  - **A second outline records no play.** A card named where an older track of the same card, in view, covers half of it or more is never announced. It leaves no play behind, and it does not make the next copy of the card put down nearby a play already made. How far apart the two outlines' centres lie does not matter: a Vex straightened by a hand was outlined again a quarter of a card off. A second copy put over half of the first is taken for the same card, as the board already drew it.
  - **A card does not move from where it still lies.** A track of a card that went out of sight is taken for that card moved, when the card is named elsewhere, only if no other track of the card has been seen since where it lay. Otherwise it was a second outline, and the card named elsewhere is a new one.
- **Why:** Replayed at five frames a second (Full, the extension's pace) and at two (Balanced, and a slow computer's), on the Barcelona final's first 200 s and the Los Angeles final's first 4 minutes. The rules add no play, and every play they withdraw was checked against the frames: each is a card already on the table, announced again after a hand passed over it, straightened it or spread its fan.
  - On Los Angeles they withdraw 4 plays at each pace: a Grim Apothecary lying there from 26 s (twice at each pace), the Vex, a Pit Rookie and a First Mate.
  - On Barcelona they withdraw a Tideturner at two frames a second, and nothing at five.
  - On the replay the engine is checked against (Los Angeles' first 2 minutes at two frames a second), a Charm and the Grim Apothecary are gone, and nothing else changes.
  - A first version also asked the two centres to lie within a quarter of a card. It rested on the Vex, labelled by eye as a second copy, and let two of these false plays through. Five of the plays labelled by eye on Los Angeles were such cards announced again.
  - **Five frames a second is not always better than two.** On Barcelona's first 200 s, two frames a second announce 13 of the 15 plays labelled by eye and five 7 of them; on Los Angeles five find more. On Barcelona, where the hands stay low over the table, three ways a higher rate loses plays were found: a hand over the cards nearby keeps resetting the hand rule's half-second clock; an unsure card spends its 12 reads in under 2.5 s; and boxes settle faster, so a card's first reads come at another size. Pacing the hand rule and the re-reads every half second, and easing the boxes per second instead of per frame, found none of the missed plays, and were not adopted.
