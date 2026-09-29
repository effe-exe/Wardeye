# @rifteye/schema

The data formats Wardeye reads and writes, as TypeScript types plus small dependency-free validators:

- `TimelineDocument` / `TimelineEvent`: what happened in a match, tied to video time
- `Card` / `Printing`: catalogue entries (gameplay identity vs printed face)
- `EmbeddingIndexManifest`: which encoder built a vector index, and the row order
- `LayoutPreset`: where the table camera sits in a broadcast frame
- `ReviewPack` / `ReviewAnswers`: model proposals a person marks correct or wrong, and their answers

This package is **Apache-2.0** (unlike the rest of Wardeye, which is AGPL-3.0-only), so tournament sites and other tools can produce and consume Wardeye data without licence questions. See [LICENSING.md](../../LICENSING.md).
