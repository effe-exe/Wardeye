# @rifteye/schema

The data formats RiftEye reads and writes, as TypeScript types plus small dependency-free validators:

- `TimelineDocument` / `TimelineEvent`: what happened in a match, tied to video time
- `Card` / `Printing`: catalogue entries (gameplay identity vs printed face)
- `EmbeddingIndexManifest`: which encoder built a vector index, and the row order
- `LayoutPreset`: where the table camera sits in a broadcast frame

This package is **Apache-2.0** (unlike the rest of RiftEye, which is AGPL-3.0-only), so tournament sites and other tools can produce and consume RiftEye data without licence questions. See [LICENSING.md](../../LICENSING.md).
