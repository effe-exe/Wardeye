# Licensing

RiftEye is licensed under the **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`, see [LICENSE](LICENSE)), except where this file or a `LICENSE` file inside a directory says otherwise.

| Path | Licence | Why |
|---|---|---|
| Everything, unless listed below | AGPL-3.0-only | The core project |
| `packages/schema/` (when created) | Apache-2.0 | Data formats (timeline, catalogue, layout) that other tools should be able to read and write freely |
| `layouts/` (when created) | CC0-1.0 | Community layout presets are plain coordinates. Anyone may reuse them |
| Model weights (release assets) | Stated in each model card; AGPL-3.0-only by default | See [07 §7.5](docs/research/07-licensing-and-governance.md#75-models-weights-and-data) |
| Datasets | Not distributed | See [04 §4.9](docs/research/04-data-and-evaluation.md#49-dataset-governance-rules) |

Source files carry an SPDX header, for example:

```
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and RiftEye contributors
```

**Commercial licences.** Organisations that cannot accept the AGPL's terms can license RiftEye under other terms from the Maintainer. Open an issue with the `licensing` label or contact [@effe-exe](https://github.com/effe-exe). This is possible because every contributor signs the [CLA](CLA.md). The CLA also guarantees that contributions stay available under the AGPL.

**Not covered by any of these licences:**

- The RiftEye name and logo. See [TRADEMARKS.md](TRADEMARKS.md).
- Riot Games' intellectual property: card names, text, artwork and Riftbound itself. RiftEye loads card data at runtime from sources Riot allows and redistributes none of it. See [NOTICE](NOTICE).
