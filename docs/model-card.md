---
license: other
license_name: wardeye-models
license_link: LICENSE.md
tags:
  - onnx
  - wardeye
---

<!-- The card of the Hugging Face repository the Wardeye extension downloads its models from (decision D-038 in the
     Wardeye repository, docs/decisions.md). Upload it there as README.md, with LICENSE.md below as its own file. -->

# Wardeye models

The recognition models of [Wardeye](https://github.com/effe-exe/wardeye), the free browser extension that names the face-up cards on the table in Riftbound streams. The extension downloads them from here the first time it reads a video, checks each against the sha256 it was built with, and keeps it in the browser. Nothing is sent about the viewer.

| File | What it is | Size |
|---|---|---|
| `detector-v0.onnx` | The card detector, float32: finds each card's four corners in 576 px tiles | 144,993,784 bytes |
| `embedder-v1.onnx` | The card embedder, float32: a 256-number fingerprint of a card's picture | 87,905,236 bytes |
| `embedder-v1.fp16.onnx` | The same embedder in float16, for GPUs that support it | 44,453,867 bytes |

The sha256 of each file is in the `standalone.json` of the Wardeye release that uses it.

## Terms

**These models are not open source, and they are not under the AGPL that covers Wardeye's code.** All rights are reserved. They may be used only to run Wardeye. See [LICENSE.md](LICENSE.md).

They were trained on Riot Games' card art. Wardeye was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.

---

<!-- LICENSE.md, a file of its own in the repository: -->

# Wardeye models: terms

Copyright (C) 2026 Federico Vietti. All rights reserved.

The files in this repository may be downloaded and used only by the Wardeye browser extension, to recognise cards in the video its user is watching. Any other use, copy, modification, redistribution or training on them needs the copyright holder's written permission.

They are provided as they are, without warranty of any kind.
