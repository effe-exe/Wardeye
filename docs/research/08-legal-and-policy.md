# 08: Legal and policy constraints

> **Risk analysis, not legal advice.** Riot's developer pages were read through verbatim third-party copies because the official pages were unreachable from the research environment. **Re-read the live pages** (developer.riotgames.com/docs/riftbound, developer.riotgames.com/policies/riftbound, riotgames.com/en/legal) before relying on any quote. State as of September 2026.

## 8.1 Summary

| Area | Constraint | Wardeye's response |
|---|---|---|
| **Riot: registration and key** | Apps need a written licence **or** an approved, app-specific Riot API key. Products that serve players must register "regardless of whether or not your product uses official documented APIs" Wardeye does **not** use the Riot API ([D-015](../decisions.md#d-015-no-riot-api-no-riot-assets-distributed)). It stays free and non-commercial for Riftbound, carries the notice, and acts on any request from Riot |
| **Riot: assets** | "Your App may only use Riftbound assets (including cards) provided by the Riot API. No external or unofficial materials." Wardeye **distributes no card images or text**. The extension loads them from Riot's public card gallery in the viewer's browser; releases carry only models and a vector index keyed by collector code |
| **Riot: metagame data** | Apps that "publish or retain metagame-defining data" will not be approved: play rates and win rates of decks or cards, matchup differentials | **No cross-match statistics.** Per-match timelines only. Ask Riot before any aggregate feature |
| **Riot: monetisation** | Needs an Approved key, a free tier, and transformative paid content. Subscriptions, donations and crowdfunding are accepted; no "middle-man" resale of API data to third parties **No paid Riftbound features.** Any commercial Riftbound use, including tools for organisers, needs a written licence from Riot first |
| **Riot: integrity and brand** | No unfair advantage to players; no implied endorsement; the LJJ §6 notice; no Riot logos; no Riot trademarks, trade names or character names in domains or social handles | Public-information-only rule; notices in [NOTICE](../../NOTICE); name review (§8.2) |
| **Broadcast footage** | Twitch's terms exclude downloading and scraping "except as expressly permitted". YouTube's developer policies forbid storing content without written approval | Recordings only **directly from organisers or Riot**, with written permission. No bulk platform downloading |
| **Embedding players** | Twitch: nothing may be overlaid on the embedded player. YouTube: the player may not be modified or built upon | The web app puts the timeline **beside** the player |
| **Personal data** | Players' faces and names appear in frames | Crop training data to the table, blur faces, set retention limits, purge by source |
| **Browser store** | Single purpose; no remotely hosted code; no YouTube downloading | Bundle all code and WASM; fetch only data; never save video |

## 8.2 Riot Games policies

Three documents apply together: the **Legal Jibber Jabber** (LJJ, Riot's fan-content policy), the **Riftbound Digital Tools Policy** and the **Riftbound Developer API Policy**.

**The licence to use Riot IP is narrow** (LJJ §1): "personal, non-exclusive, non-sublicenseable, non-transferable, revocable, limited … strictly for noncommercial (except as specifically provided below) community use."

**Commercial projects** (LJJ §2) need a written licence, except for three carve-outs: passive ads, donations or subscriptions while streaming, and "commercial Projects that both (1) comply with our API Terms and API Policies; and (2) use a currently valid Riot API key." Note that "any Project that involves a business or legal entity" counts as commercial. If a key is revoked, "you must immediately shut down your Project."

**Apps** (LJJ §3): "we prohibit the use of our IP in games and apps", and specifically no app-store apps without a licence or a valid key.

**What the Digital Tools Policy approves** (verbatim):

- An app needs "either a written license from us or a valid App-specific API key".
- The Riot API gives "authorized access to select Riftbound assets—including card art, rulesets, and other materials".
- The approved examples are **deckbuilders and card libraries**. Wardeye is a new category, a spectator companion, so approval is not guaranteed and the application must explain it well.
- Riot is "not looking to approve Apps that enable automated rules enforcement". It wants tabletop play "without the assistance of digital trackers and interactions".
  - **Wardeye is not a play aid.** It is for spectators of broadcast games, many of them time-shifted. Keep that framing, and keep the public-information-only rule, in the application.

**Monetisation** requires all of the following:

- Your App must not simulate or replicate Riftbound gameplay.
- Your App must have a valid API key approved by Riot.
- It must have a free tier of access for players.
- Any content you charge for must be transformative: "you added value to original materials and data by creating new information, new aesthetics, new insights, or new understandings".
- No betting or gambling.
- Practices must not be "unfair, as decided by Riot".

**Card display rules:**

- Show the official English text, or Riot's official translation.
- **Use only assets provided by the Riot API.**
- Label preview cards as "previewed and unreleased".

**The Developer API Policy adds:**

- Registration is mandatory for every product that serves players.
- **One product per key.**
- "Your API key may not be included in your code, especially if you plan on distributing a binary."
- "Products cannot create an unfair advantage for players".
- Use cases that **will not be approved:**
  - "Apps with a small, personal audience";
  - "'middle-man' type of data usage, such as an app that obtains data from our API and gives or sells it to a third-party company";
  - "Apps that publish or retain metagame-defining data for Riftbound".
- The two pages disagree on scope: the Tools Policy lists the metagame ban under gameplay simulators, while the API Policy applies it to every app. **Assume the stricter reading.**

**Riot's licence to your project** (LJJ §7): Riot "may use, copy, modify, distribute, and make derivative works of your Project in any form, on a royalty-free, non-exclusive, irrevocable, transferable, sub-licensable, worldwide basis, for any purpose". That sits alongside, and outside, the AGPL ([07 §7.6](07-licensing-and-governance.md#76-riots-policies-sit-above-all-of-this)).

**Content reuse** (LJJ §4): "Don't just rip off or add some light commentary to existing content (e.g., esports matches, other players' vods)." Wardeye never re-hosts VODs; it adds a transformative layer next to the official players.

**Names and marks** (LJJ §5):

- No Riot logos or trademarks.
- No domains or social accounts using Riot's "trademarks, trade names, character names, etc."
- No Riot IP names as "keywords or internet search tags". Mind the store-listing keywords.

**The name.** The working name, "RiftEye", was not a Riot mark, but it leaned on "Rift" (Riftbound, Summoner's Rift, Wild Rift). It also sat close to the existing competitor **RiftSight**, a tool that does the same kind of thing, and an unrelated "RiftEye" mod already exists. **A neutral, game-agnostic name is safer**, and it fits the multi-game engine strategy ([07 §7.6](07-licensing-and-governance.md#76-riots-policies-sit-above-all-of-this)). So the project is now Wardeye ([D-020](../decisions.md#d-020-the-product-is-called-wardeye)), which a trademark search must still clear.

**Required notice** (LJJ §6, required by both Riftbound policies "in a place that's clear and easy to find"):

> Wardeye was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.

## 8.3 Broadcast footage and platform terms

**Twitch's Terms of Service** (last modified 2026-08-12):

- They license use for "personal use or internal business use only".
- They exclude "use of any data mining, robots, or similar data gathering or extraction methods" and "downloading (except page caching) … except as expressly permitted".
- They ban robots and scrapers "for any purpose".
- **Bulk-downloading VODs risks breach of contract and an account ban, even with the rights holder's consent.**

**YouTube:**

- **Terms:** downloading and automated access are prohibited except with written permission, or "as permitted by applicable law".
- **Developer Policies:** forbid storing copies of YouTube content without YouTube's written approval, and forbid modifying or building upon the player.

**Who owns what:**

- **The broadcast** belongs to its producer: Riot for Riot-run events, the organiser for third-party events, the streamer for personal streams.
- **The card art** belongs to Riot.
- Ask each party.

**Practical rule:** training footage comes **as files, directly from organisers or Riot, under a written agreement**. The agreement covers the use (training and evaluation), storage, retention, face blurring and takedown. The permission log is `sources.yaml` ([04 §4.4](04-data-and-evaluation.md#44-stream-footage-the-data-engine)). Organisers are also the reliable archive, because platforms expire VODs.

**The extension is different.** It processes the frames the user is already watching, locally, like any other client-side rendering aid. Nothing is downloaded, recorded or uploaded, apart from the opt-in single-card correction crop.

## 8.4 Text and data mining law

| Jurisdiction | Rule | Fit for Wardeye |
|---|---|---|
| **EU**, DSM Directive 2019/790 | **Art. 3:** research organisations and cultural-heritage institutions only. **Art. 4:** any purpose, including commercial, with lawful access, *unless* rights are reserved "in an appropriate manner, such as machine-readable means". The recitals count website terms as such a reservation. Art. 7(1) protects Arts. 3, 5 and 6 from contract override, not Art. 4 | A sole proprietorship is not a research organisation. Platform terms likely reserve rights, so **do not rely on Art. 4** for platform downloads |
| **EU AI Act** | Recital 105 treats gathering AI training data as text and data mining under 2019/790 | Same conclusion |
| **Switzerland**, Copyright Act Art. 24d | Copies for **scientific research** made by a technical process, with lawful access | Training a model for a product is a grey zone. It is also not an exception to platform contracts |

**Conclusion: permission first.** The exceptions are a backstop for research, not a licence to scrape.

## 8.5 Personal data

- Broadcast frames show **players' faces and names**. That is personal data under GDPR (which applies to services aimed at the EU) and the Swiss FADP.
- **Training data:** crop to the table region, blur faces anywhere else, drop player-cam regions, set retention limits, log provenance, and purge by source.
- **The extension:** there are no accounts in v1. Settings stay local. The only upload is the opt-in correction ([04 §4.10](04-data-and-evaluation.md#410-opt-in-corrections-from-the-extension)). A privacy policy ships before the store listing.

## 8.6 EU AI Act

- It applies to providers placing systems on the EU market, including a Swiss provider (Art. 2(1)).
- Systems released under free and open-source licences are exempt unless they are high-risk or fall under Art. 5 or Art. 50 (Art. 2(12)). A card recogniser is neither high-risk nor within Art. 50's transparency cases.
- What remains is the Art. 4 **AI literacy** duty, which has applied since 2025-02-02. Most other provisions have applied since 2026-08-02.
- Whether later amendments (the "Digital Omnibus") changed any of this is unverified.

## 8.7 Store and platform policies

- **Chrome Web Store:**
  - A single, narrow purpose, with collected data strictly necessary to it.
  - **No remotely hosted code:** all JS and WASM are bundled. Model weights and the catalogue are data.
  - No YouTube downloading features.
  - A privacy disclosure that matches the actual payload.
- **Twitch Extensions** (later): a review of metadata, policy, code and function. No in-extension payment flows beyond Bits and subscription gating.
- **Firefox Add-ons** (later): the same bundled-code rule applies.

## 8.8 Checklist

- [x] **M0:** decided not to use the Riot API ([D-015](../decisions.md#d-015-no-riot-api-no-riot-assets-distributed)). No Riot assets are distributed, and the project stays free and non-commercial for Riftbound.
- [ ] **M0:** settle the name after a trademark search ([07 §7.4](07-licensing-and-governance.md#74-trademark)).
- [ ] **M0–M1:** written footage agreements with organisers; `sources.yaml`; face-blurring in the ingest pipeline.
- [ ] **M2:** a gallery adapter that loads card data from Riot's public gallery at display time, with nothing cached beyond the browser. Show the LJJ notice in the extension's about page and the store listing.
- [ ] **M3:** privacy policy, store disclosures and data-retention settings before the public beta.
- [ ] **Before any commercial Riftbound feature:** a written licence from Riot.
- [ ] **Always:** act promptly on any request from Riot, organisers or broadcasters.
