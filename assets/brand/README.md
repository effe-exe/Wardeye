<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/lockup.svg">
    <img src="logo/lockup-light.svg" alt="Wardeye" width="260">
  </picture>
</p>

# Wardeye brand book

**Place the ward. See the table.** This is the reference for anything that shows Wardeye: the extension, the tools' pages, the README, the web page and social posts. Alpha, September 2026. A community project by Federico Vietti ([D-024](../../docs/decisions.md#d-024-wardeye-has-its-own-brand-book)).

## 1. What Wardeye is

In MOBAs, a ward is what you place to gain vision. Wardeye brings that idea to Riftbound streams: turn it on for a broadcast and it gives you sight of every card on the table (art, name, text) and a timeline of the match. Free, open-source and local. Nothing leaves the viewer's machine.

**Mission.** Place the ward on any Riftbound stream. See the table clearly, without pausing, searching or sending video anywhere.

**What it will do**
- **Hover to inspect:** point at a card on the table and see its art, name and text.
- **Match timeline:** cards played, spells cast, units moved, turns and score. Click any event to jump there.
- **Match at a glance:** each player's legend, battlefields and every card seen so far.

## 2. The five non-negotiables

1. **Public information only.** Hand cams, face-down cards and hidden deck contents are never processed. Wardeye sees only what the broadcast already shows; a published decklist only helps name what is face up ([D-026](../../docs/decisions.md#d-026-legends-and-published-decklists-narrow-the-search-and-never-reveal-anything)).
2. **Local first.** Inference runs in the viewer's browser, with WebGPU and a WASM fallback. No video leaves the machine.
3. **Measured, not claimed.** Every model ships with results on real broadcasts it was not trained on. Accuracy is reported, not marketed.
4. **Open and clean.** AGPL code. Permissively licensed dependencies and base models only. No hidden telemetry. The trained weights ship inside the extension but are not published ([D-022](../../docs/decisions.md#d-022-free-for-everyone-closed-weights-a-showcase-for-gradeon)).
5. **No footage, art or card text in Wardeye.** Card data comes from Riot's public card gallery, in the browser. Broadcasts belong to their organisers.

## 3. Logo

The mark is a ward: a stake topped with a vision orb. You place it (turn the extension on) and it gives sight of the table. It is abstract and original, not a copy of any game's ward. It pairs with the wordmark, set in Space Grotesk Bold.

| File | Use |
|---|---|
| [`logo/lockup.svg`](logo/lockup.svg) | Primary lockup, mark and wordmark, on dark backgrounds |
| [`logo/lockup-light.svg`](logo/lockup-light.svg) | The lockup on light backgrounds: the full-colour mark with the dark wordmark |
| [`logo/mark.svg`](logo/mark.svg) | The mark alone, full colour |
| [`logo/mark-white.svg`](logo/mark-white.svg), [`mark-muted.svg`](logo/mark-muted.svg) | On dark backgrounds, where colour would compete |
| [`logo/mark-black.svg`](logo/mark-black.svg) | One colour on light backgrounds (print, stamps): the ring, the pupil and the highlight are cut out |
| [`logo/wordmark.svg`](logo/wordmark.svg), [`wordmark-dark.svg`](logo/wordmark-dark.svg) | The wordmark alone, for dark and light backgrounds |

- **Clear space:** around the lockup, the height of the mark.
- **Colour:** the full-colour mark on dark and on light backgrounds alike. The white, muted and black marks are for one-colour uses.
- **Minimum size:** the mark 24 px tall; the full lockup 120 px wide. The browser's 16 px toolbar icon is the one exception.
- **Never** stretch, recolour or add effects to the mark.

The SVGs are drawn from the brand book's own vectors, and the wordmark's letters are outlines, so no font is needed to show them. The name and the logo are trademarks ([TRADEMARKS.md](../../TRADEMARKS.md)); they are not under the AGPL.

## 4. Colour

Wardeye inherits Gradeon's dark-first system. Near-black surfaces, high-contrast text and a single violet primary keep the UI calm, so the card previews and the timeline stay the focus. The tokens live in [`tokens.css`](tokens.css).

| Token | Value | Use |
|---|---|---|
| `--wd-primary` | `#8B7CF6` | Actions, links, focus, recognised cards |
| `--wd-primary-light` | `#A99BFF` | Hover, highlights |
| `--wd-primary-dark` | `#6D5BD0` | Pressed, active |
| `--wd-accent` | `#C4B5FD` | Soft emphasis |
| `--wd-bg` | `#0A0A0B` | Background |
| `--wd-surface` | `#121214` | Surface: panels |
| `--wd-surface-2` | `#1A1A1E` | Surface 2: raised panels, controls |
| `--wd-border` | `#2A2A30` | Borders, rules |
| `--wd-text` | `#F4F4F5` | Text |
| `--wd-muted` | `#A1A1AA` | Secondary text |
| `--wd-dim` | `#71717A` | Faint text and marks |
| `--wd-success` | `#34D399` | Measured results, on target, a sure read |
| `--wd-warning` | `#FBBF24` | Alpha status, caution, an unsure read |
| `--wd-error` | `#F87171` | A detection miss, a failure |

**Accessibility.** Primary on the background meets WCAG AA for large text; text on the background exceeds AAA. Never put body text on primary: use primary for interactive elements and short labels only.

## 5. Type

| Typeface | For | Weights |
|---|---|---|
| **Space Grotesk** | Display and brand moments: page titles, section heads, the wordmark | 500, 600, 700 |
| **Inter** | UI and body: everything read at small sizes | 400, 500, 600 |
| **JetBrains Mono** | Code, metrics and ids: timings, scores, printing ids | 400, 500 |

| Step | Size / weight | Use |
|---|---|---|
| Display | 28–36 / Bold | Hero, page titles |
| H1 | 22 / SemiBold | Section heads |
| H2 | 16 / Medium | Card titles, panels |
| Body | 13–14 / Regular | Paragraphs, descriptions |
| Caption | 11 / Regular | Labels, metadata |
| Micro | 9–10 / Medium | Badges, timestamps |

Section labels (eyebrows) are Inter SemiBold, 11 px, in capitals and primary, as in "01 · BRAND OVERVIEW". The fonts in [`fonts/`](fonts/) are Latin subsets of the variable fonts, under the SIL Open Font License 1.1 (the `OFL-*.txt` files there), not the AGPL.

## 6. Voice

For players and viewers who already care about the game: precise, calm, a little technical. Never hype, never corporate, never vague.

- **Precise.** Name the thing: "card detector", not "AI magic". Numbers over adjectives.
- **Calm.** No urgency, no fear of missing out. The product is free and local; the copy can afford to be quiet.
- **Technical when useful.** Report accuracy on held-out broadcasts, WebGPU, frame rates. Assume the reader can handle it.
- **Honest about status.** Alpha is alpha. Say what works, what doesn't, and what the next milestone is.
- **Respectful of rights.** Broadcasts and card art belong to others. Credit organisers; load art from Riot's gallery.

| Write | Don't write |
|---|---|
| Place the ward. Hover a card and see its art, name and text. | Revolutionary AI that transforms how you watch! |
| Trained and tested on real broadcasts. Results published. | The smartest card recognition on the internet. |
| Inference stays in your browser. No video leaves the machine. | |

## 7. Messages

- **Elevator pitch:** Wardeye is the ward you place on a Riftbound stream. Hover any card to inspect it, follow the match as a timeline, and see the whole board at a glance: open-source, in your browser, from the video alone.
- **One-liner:** Place the ward. See the table.
- **For viewers:** Place the ward once. Hover any card, jump to any play, stay in the stream.
- **For the community:** Free, AGPL, local-first. Built in the open, measured on real broadcasts.
- **For organisers:** No footage is stored or uploaded. Card art is loaded from Riot's public gallery.
- **For developers:** The models come with results on broadcasts they never saw in training. The code is AGPL; every dependency is permissive.

**Naming.** Wardeye was called RiftEye until September 2026 ([D-020](../../docs/decisions.md#d-020-the-product-is-called-wardeye)). Internal code names may still say `rifteye`. Public copy always says Wardeye.

## 8. Visual language

The product lives on top of the Twitch and YouTube players, so it stays minimal and the stream remains the hero.

- **Overlays are translucent, borders are hairline**, and colour is kept for recognition confidence and interactive states.
- **Cards on the table are marked, not boxed:** a short stroke along each edge at each corner, 2 px, with a faint dark halo. The primary marks a named card, the warning colour an unsure read, muted a card still being read. Face-down cards and runes get no mark. The full outline, with a faint fill, is for the card under the pointer, and a card just named flashes it once.
- **Names are chips:** the surface at 88%, a hairline border, 6 px corners, Inter SemiBold at 10 to 13 px, growing with the player. A name that would cover another goes under its card, and waits for the pointer when both places are taken. The name of the card pointed at is edged in the primary.
- **The viewer chooses what stays on the video:** outlines and names, outlines only, or clean, where nothing shows until a card is pointed at. Pointing at a card shows it in every view.
- **Card preview**, on hover: a surface panel 224 px wide, with a 1.5 px border (the primary; the warning colour for an unsure read) and 8 px corners. The official art (from Riot's gallery) on top, then the name (Inter SemiBold), the type (muted) when the state has it, how sure the read is (a thin meter and the number, "90% sure", primary, small), and the cards lying under it, each with a small picture and counted when there are several.
- **The badge** says only the essentials (the cards named, and "on the processor" when WebGPU is not there). The engine's timings show when it is pointed at, in JetBrains Mono.
- **Timeline:** primary dots on a hairline, with short labels (Turn 3, Unit, Spell, Score).
- **Icons** are simple and geometric, with a 1.5 px stroke, like Lucide's open icons.
- **Motion** is restrained: 150–250 ms ease-out for hover and panel open, and one flash of a card's outline when it is named. Nothing moves continuously.

## 9. Where it appears

- **The extension**, the main surface: the mark in the toolbar and on the player control; the card previews and the timeline in the full colour system; the overlay in the dark theme only.
- **Documentation and the README:** Markdown first, the lockup in the header, Space Grotesk for titles and Inter for body where HTML allows. Link to this book and the principles.
- **GitHub:** the repository is `wardeye` (older `rifteye` paths may remain). The description is the one-liner. The social preview: the dark background, the lockup and the one-liner.
- **The web page:** Gradeon's dark layout: a near-black background, violet calls to action, generous space, short sentences, an alpha badge.
- **Social:** the avatar is the mark on dark or on primary; banners carry the mark, the wordmark and the one-liner.

**Gradeon.** Wardeye wears Gradeon's look: the same dark base, the same violet family, the same calm technical voice. It is a separate community project by the same maker. Say so ("by Federico Vietti, who also makes Gradeon"); never present Wardeye as a Gradeon product.

## 10. Do and don't

| Do | Don't |
|---|---|
| Use the full lockup on dark backgrounds | Stretch, recolour or add effects to the mark |
| Report model results measured on held-out broadcasts | Claim accuracy without published numbers |
| Keep overlays translucent; let the stream lead | Process hand cams, face-down cards or private deck data |
| Say "alpha" and name the current milestone | Upload or store broadcast video |
| Load card art only from Riot's public gallery | Reintroduce the name RiftEye in public copy |
| Credit organisers and streamers when showing footage | Use Gradeon's name as if Wardeye were a Gradeon product |

## 11. Files

- `logo/`: the SVGs above.
- `tokens.css`: the colour, type, shape and motion tokens.
- `fonts/`: Space Grotesk, Inter and JetBrains Mono (WOFF2, Latin), with their licences.
- `brand.mjs`: the builds' helper. It puts the fonts (inlined) and the tokens in front of a page's stylesheet, so pages work from `file://` too; the extension points it at its own copies of the fonts.
