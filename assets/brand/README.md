# Gradeon's look

Wardeye wears the look of [Gradeon](https://gradeon.ai), the AI card pre-grading app by the same maintainer ([D-020](../../docs/decisions.md#d-020-the-product-is-called-wardeye)): near-black and precise, with Gradeon's purple and Space Mono. The pages and the extension's overlay use these tokens. Nothing here imitates Riot's or Riftbound's design, and no Riot logo appears anywhere.

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0a0a0a` | The page |
| `--panel` | `#111111` | Panels, the hover card |
| `--raised` | `#1a1a1a` | Buttons |
| `--rule` | `#1a1a1a` | Rules between regions |
| `--line` | `#333333` | Borders of controls |
| `--text` | `#ffffff` | Text |
| `--muted` | `#999999` | Secondary text |
| `--faint` | `#666666` | Faint marks, such as face-down cards |
| `--accent` | `#6153cc` | Gradeon's purple: primary buttons, selection |
| `--accent-2` | `#7b6fd4` | Its light tint: card outlines, times |
| `--good`, `--warn`, `--bad` | `#00ff88`, `#ff6600`, `#ff0044` | Sure, unsure, error: Gradeon's grade colours |

- **Type:** Space Mono for headings, labels, buttons, numbers and times. The system sans-serif for running text.
- **Corners:** 2 px for controls, badges and thumbnails; 4 px for panels and the player.

## Files

- `SpaceMono-Regular.ttf` and `SpaceMono-Bold.ttf`: Space Mono, unmodified, from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/spacemono). Copyright 2016 The Space Mono Project Authors. They are under the SIL Open Font License 1.1 ([OFL.txt](OFL.txt)), not the AGPL.
  - sha256 `95837e182baeeada83368f7748db28357f0a1b75c6b84ff7065b5edf933c8e18` (Regular)
  - sha256 `405e73d41afb7e5906efce206a326af5c956f38e255f35421c260e861e599c59` (Bold)
- `brand.mjs`: the builds' helper. It inlines the fonts into a page's stylesheet, so the page works from `file://` too.

The live runner's page (`ml/rifteye_ml/live/static/`) has the same tokens. The Python package carries no fonts, so that page falls back to the system's monospace.
