# 07: Licensing and governance

**The goal:** RiftEye should be genuinely open source and easy to contribute to. The Maintainer should keep the ability to offer commercial terms, run paid services, or sell the project later. And nobody should be able to take the code closed and sell it as their own.

This chapter explains the setup that achieves that and the alternatives that were rejected. It also covers the one constraint that sits above any software licence: Riot Games' policies ([08](08-legal-and-policy.md)).

> This is a research summary, not legal advice. Have the CLA and any commercial licence reviewed by a lawyer before relying on them.

## 7.1 The setup

| Piece | Choice | What it does |
|---|---|---|
| Code licence | **AGPL-3.0-only** ([LICENSE](../../LICENSE)) | OSI open source. Anyone may use, modify and host it, but modified versions offered over a network must offer their source (§13). |
| Contributions | **CLA** based on Harmony, outbound option five ([CLA.md](../../CLA.md)) | Lets the Maintainer license every line of the project, including commercially: the basis of dual licensing |
| Name and logo | **Trademark policy** ([TRADEMARKS.md](../../TRADEMARKS.md)) | Forks get the code, not the name. AGPL §7(e) explicitly allows declining trademark rights. |
| Model weights | Released per model, **AGPL-3.0-only unless a release says otherwise** | Community models ship in the extension. Larger hosted models are not distributed at all. |
| Datasets | **Not distributed** | Footage belongs to broadcasters; only manifests are public ([04 §4.9](04-data-and-evaluation.md#49-dataset-governance-rules)) |
| Embeddable pieces | **Apache-2.0** for the data schemas and any embed SDK, when they exist | Tournament sites and tools can read RiftEye timelines without AGPL questions. Grafana and Plausible use the same split for their SDKs. |

[LICENSING.md](../../LICENSING.md) maps directories to licences, and each file will carry an SPDX header.

## 7.2 Options that were considered

| Licence model | OSI open source? | Stops a closed hosted fork? | Owner can still sell? | Community friction | Precedent |
|---|---|---|---|---|---|
| **AGPL-3.0 + CLA** ✔ | Yes | Partly: hosting a modified version requires publishing its source | Yes, with the CLA | Medium: some companies ban AGPL, and the CLA adds a step | Ultralytics (AGPL + Enterprise licence, same domain), Grafana, Plausible. Redis and Elastic both *returned* to AGPL after leaving open source |
| GPL-3.0 + CLA | Yes | No: no network clause | Yes | Medium | Poor fit for a hosted service |
| MPL-2.0 | Yes | No | Yes | Low | OpenTofu |
| MIT/Apache + open core | Yes | No | Only through closed add-ons, and so can anyone | Lowest | GitLab |
| FSL-1.1 (converts to Apache after 2 years) | No | Yes, for 2 years per release | Yes | Medium–high | Sentry |
| BUSL-1.1 | No | Yes | Yes | High | HashiCorp → the OpenTofu fork |
| Elastic-2.0, PolyForm, Sustainable Use | No | Yes | Yes | High; PolyForm Noncommercial would also block organisers and monetised streamers | Elastic (since re-added AGPL), n8n |

**Why not source-available (FSL/BSL):** it is not open source. It would deter exactly the contributors the project wants, and recent history is not encouraging. HashiCorp's move to BSL produced OpenTofu; Redis and Elastic eventually went back to AGPL.

**Why not MIT/Apache:** anyone could ship a closed, paid "RiftEye Pro" built on the community's work, and the project would have no answer.

**The lesson from 2026:** Cal.com, AGPL with a commercial folder, went closed-source in April 2026 and re-released its free code under MIT, and took visible criticism for it. Holding the rights makes a change possible. It does not make it free, in trust.

## 7.3 The CLA

Without a CLA, GitHub's terms make contributions "inbound = outbound": you receive them under the AGPL only. A DCO sign-off certifies origin but grants no extra rights. Either way, the Maintainer could not offer contributed code under commercial terms.

RiftEye's CLA ([CLA.md](../../CLA.md)) is adapted from the **Harmony** individual agreement with **outbound option five**:

- **It is a licence, not an assignment.** Contributors keep their copyright. Assignment clauses are also unenforceable in some EU countries, for example Germany.
- **The grant** is perpetual, irrevocable, worldwide and sublicensable, and covers copyright (including database rights, which matter for labels and datasets) and patents.
- **Outbound:** the Maintainer may license contributions under **any licence, including commercial or proprietary**. In exchange, the Maintainer must **also** keep each contribution available under the licence used when it was submitted. That condition is what makes the agreement fair enough for contributors to sign.
- **Assignment clause:** the agreement can move to a company the Maintainer forms, or to an acquirer, without re-signing everyone. This is what keeps "sell it later" possible.
- **Swiss law and courts.** The Maintainer operates from Switzerland.
- **Signed by a PR comment.** The GitHub account, numeric ID, time, PR and CLA version are recorded on a dedicated branch ([§7.8](#78-tooling)).

## 7.4 Trademark

The AGPL covers code, not names. Forks are welcome, but under their own name, so the official builds, extension-store listing and hosted service stay identifiable. Precedents: Debian's Iceweasel, OpenTofu's rename, and Rust's 2023–2025 policy rewrite. That last one is a reminder to keep community use generous, as [TRADEMARKS.md](../../TRADEMARKS.md) does.

**Registration costs**, if and when the name is final:

- **Switzerland (IPI):** CHF 350 for up to 3 classes when filed electronically.
- **EU (EUIPO):** €850 for the first class, €50 for the second, €150 for each further class.
- **Classes:** 9 (software, browser extension) and 42 (SaaS); 41 is optional.

**Clear the name first.** Search "RIFT*" in classes 9, 41 and 42 on WIPO's Global Brand Database, EUIPO eSearch and Swissreg. See the naming risk in [08 §8.2](08-legal-and-policy.md#82-riot-games-policies).

## 7.5 Models, weights and data

Code, weights and data are licensed separately.

**How dependencies constrain the licence choice.** This is why [decision D-002](../decisions.md) exists.

| Dependency type | Effect on RiftEye's ability to offer commercial terms |
|---|---|
| Apache-2.0 / MIT / BSD code or weights | Fine. They can go into an AGPL work and into a commercial edition, as long as NOTICE files are kept. |
| **AGPL library** (e.g. Ultralytics YOLO) | Blocks it: the Maintainer cannot license someone else's AGPL code commercially. A commercial edition would need a third-party enterprise licence. |
| **Research-only weights** (e.g. Apple's MobileCLIP, `apple-amlr`: "exclusively for Research Purposes", fine-tunes included, revocable) | Cannot ship in anything used commercially, including by monetised streamers |
| **Platform-licensed weights** (e.g. RF-DETR XL/2XL under Roboflow's PML) | Incompatible with open distribution. RF-DETR N/S/M/L are Apache-2.0 and fine. |
| Models trained with an AGPL framework (e.g. community YOLO11 card detectors) | Treat the weights as AGPL-encumbered. Use them as a reference, never as a shipped component. |

**RiftEye's defaults:**

- **Community models** (the ones in the extension) are released under AGPL-3.0-only with a model card, unless a release states otherwise.
- **Hosted models** (larger server-side detectors and embedders, and teacher models) are not distributed, so no licence question arises.
- **Datasets** built from broadcast footage are never distributed. They depend on per-source permissions ([04](04-data-and-evaluation.md#44-stream-footage-the-data-engine)).

## 7.6 Riot's policies sit above all of this

No software licence or CLA overrides Riot's terms for using its IP. Three points matter directly for "control" and "sell it later" (details in [08](08-legal-and-policy.md)):

1. **Riot gets a licence to the project.** Legal Jibber Jabber §7 lets Riot use, copy, modify and distribute fan projects "on a royalty-free, non-exclusive, irrevocable, transferable, sub-licensable, worldwide basis, for any purpose", and that applies regardless of the AGPL.
2. **Monetising a Riftbound app requires an approved Riot API key or a written licence**, a free tier, and transformative paid content. Selling API data to third parties is a "middle-man" use Riot says it will not approve.
3. **RiftEye does not use the Riot API** ([D-015](../decisions.md#d-015-no-riot-api-no-riot-assets-distributed)). So for Riftbound it stays **free and non-commercial**. Any paid Riftbound feature, including tools sold to organisers, would first need a written licence from Riot.

**Structural consequence.** The durable, sellable asset is the **game-agnostic engine** plus the brand and the data engine ([D-008](../decisions.md)). Riftbound is its first game pack, operated within Riot's rules and ideally in partnership with Riot or its organisers.

## 7.7 Sustainability model

In broad terms, with Riftbound-specific monetisation subject to Riot's approval:

- **Free and open:** the extension, local inference, timelines and hover, community models, training and evaluation code.
- **Hosted services:** processed VOD timelines at scale, and heavier server-side models.
- **Tools for broadcasters and organisers:** live overlays and multi-table production features.
- **Commercial licences** for organisations that cannot accept AGPL terms.
- **Sponsorship:**
  - GitHub Sponsors takes no fees on sponsorships from personal accounts, up to 6% from organisation accounts, and supports Switzerland.
  - Open Source Collective charges a 10% host fee.

## 7.8 Tooling

**The CLA bot is a small workflow in this repository** (`.github/workflows/cla.yml`), built on GitHub's own `actions/github-script`, pinned by commit SHA.

The usual choice, CLA Assistant Lite (`contributor-assistant/github-action`), was **archived in March 2026**. Its last release targets Node 20, which GitHub Actions removed on 2026-09-23. The hosted cla-assistant.io (SAP) is still maintained and is a fine alternative if a hosted service is preferred.

The workflow:

- runs on `pull_request_target` and PR comments;
- **never checks out or runs PR code**, because the event carries a write token;
- stores signatures in `signatures/cla-v1.json` on an **unprotected** `cla-signatures` branch;
- sets a `CLA` commit status, which branch protection can then require.

## 7.9 Changing the licence later

If a change is ever needed:

1. Audit authors (`git log --format='%aN <%aE>'`) against the CLA signatures, and get consent from, or rewrite the code of, anyone not covered.
2. Audit dependencies and weights (§7.5).
3. Announce ahead of time, with a FAQ.
4. Update `LICENSE`, the SPDX headers, the package metadata and the README. If the CLA text changes, bump it to a new version with a new signature file.
5. Remember that **past releases stay AGPL forever**, and a fork can start from the last one.

If the project is incorporated or sold, move the copyrights, CLA rights, trademarks, domains and store listings to the company by written assignment.

## Sources

- AGPL-3.0 text (§7(e), §13) and Apache-2.0 text (§4(d), §6); Apache–GPL compatibility: apache.org/licenses/GPL-compatibility.html
- Ultralytics licence, Enterprise licence and CLA: github.com/ultralytics/ultralytics (README, `docs/en/help/CLA.md`)
- Grafana `LICENSING.md` (Apache-2.0 SDK packages): github.com/grafana/grafana ; Plausible README (MIT tracker): github.com/plausible/analytics
- Redis licence history: github.com/redis/redis (`LICENSE.txt`) ; Valkey: github.com/valkey-io/valkey ; Elasticsearch `LICENSE.txt`
- Terraform BUSL and OpenTofu: github.com/hashicorp/terraform ; linuxfoundation.org/press/announcing-opentofu
- Sentry FSL: github.com/getsentry/sentry (`LICENSE.md`) ; github.com/getsentry/fsl.software
- Cal.com closing and Cal.diy: github.com/calcom/cal.diy ; theregister.com (2026-04-26)
- GitHub Terms of Service §D.6 (inbound = outbound) ; DCO 1.1
- Harmony Agreements: harmonyagreements.org ; Apache ICLA: apache.org/licenses/contributor-agreements.html
- CLA Assistant Lite (archived 2026-03-23): github.com/contributor-assistant/github-action ; CLA Assistant: github.com/cla-assistant/cla-assistant ; Node 20 removal: github.blog/changelog/2026-09-23-node-20-is-no-longer-available-in-github-actions/
- IPI fees: ige.ch (trade mark costs and fees) ; EUIPO fees: euipo.europa.eu (fees and payments)
- MobileCLIP licences: github.com/apple/ml-mobileclip (`LICENSE`, `LICENSE_MODELS`) ; RF-DETR and PML-1.0: github.com/roboflow/rf-detr
- Riot Legal Jibber Jabber: riotgames.com/en/legal
- GitHub Sponsors docs ; Open Source Collective fees: docs.oscollective.org
