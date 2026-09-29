# Licensing

RiftEye is licensed under the **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`, see [LICENSE](LICENSE)), except where this file or a `LICENSE` file inside a directory says otherwise.

| Path | Licence | Why |
|---|---|---|
| Everything, unless listed below | AGPL-3.0-only | The core project |
| `packages/schema/` (when created) | Apache-2.0 | Data formats (timeline, catalogue, layout) that other tools should be able to read and write freely |
| `layouts/` (when created) | CC0-1.0 | Community layout presets are plain coordinates. Anyone may reuse them |
| Trained model weights | Not published and not open source: the Maintainer's own, trained on Riot's card art. They ship only inside the extension | See [D-022](docs/decisions.md#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon) |
| Datasets | Not distributed | See [04 §4.9](docs/research/04-data-and-evaluation.md#49-dataset-governance-rules) |

Source files carry an SPDX header, for example:

```
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
```

**Free for everyone.** No commercial licences, paid tiers or paid services are offered ([D-022](docs/decisions.md#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon)). Every contributor signs the [CLA](CLA.md). It keeps the project's licensing in one hand, and it guarantees that contributions stay available under the AGPL.

**Not covered by any of these licences:**

- The RiftEye name and logo. See [TRADEMARKS.md](TRADEMARKS.md).
- Riot Games' intellectual property: card names, text, artwork and Riftbound itself. RiftEye redistributes none of it; the extension loads card data from Riot's public card gallery in the viewer's browser. See [NOTICE](NOTICE).
