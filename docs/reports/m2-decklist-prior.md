# M2: legends and decklists as priors

**Date:** 2026-09-29. **Status:** measured in the Python pipeline on the Barcelona crops. Not in the extension yet.

The question from [ARCHITECTURE §3.5](../ARCHITECTURE.md#35-matcher-priors-and-fusion): when the players' legends, or their published decklists, are known, how many more of the cards on the table are named right? Numbers only, all of them in [m2-decklist-prior.csv](m2-decklist-prior.csv). The decklists, the crops and the per-crop results stay private (D-006). Card ids appear here; players do not.

## Summary

- **The legends alone do most of the work, with no list.** Every card in a deck must fit its legend's two domains, runes included (Core Rules 103, [01 §1.6](../research/01-game-model.md#16-deck-rules-as-recognition-priors)). Hold each half of the table to its own legend's domains, and the hardest match rises **from 91.8% to 97.8%** of crops named right (28 fixed, none broken). Every confident read is right: 403 of 403, 86.9% of the crops.
- **The runes decide it.** The same rule with every rune allowed reaches only 93.5%. Most of the mistakes left (24 of 30 crops) were Order Runes read as the alternate-art Fury Rune, and only holding the runes to the legend separates them.
- **A list names one printing; players play others.** In the final, 54 of 154 crops (35%) showed a different printing of a listed card: an alternate-art champion, an alternate-art rune, and runes reprinted in a later set. So a listed card stands for all its printings (alternate art, overnumbered, signature, reprints), and tokens are always allowed.
- **Battlefields are shared.** Both players' battlefields lie on the midline, so a battlefield from one list turns up on the other player's half (11 crops in the final). Each half gets both lists' battlefields.
- **A wrong list is dangerous; one guard removes the danger.** Used as a hard filter, another match's lists name only 8.4% of the crops. Used as a soft bonus they still cost a little at +0.01, and more as the bonus grows. The guard: use a list only on the half whose legend, as seen on the table, it names, and the legend rule everywhere else. That scores 97.8% where the lists were wrong, the same as the legend rule.
- **The players' own lists had nothing left to fix here.** The whole gallery already names all 154 crops of the final. On the hard match, a right list would land between the legend rule and the upper bound: 97.8% to 100% (see [What it does not show](#what-it-does-not-show)).

## The set-up

- **Crops:** the Barcelona Regional labels (the PlusRB restream), held out of the embedder's training. There are 947 face-up crops of 176 tracks, the set behind the 96.0% in the [README](../../README.md#measured-not-claimed). Left out: sleeves (237), cards the gallery lacks (115) and unsure labels (4).
- **Model:** embedder v1 (the extension's model before its ONNX export). The gallery is embedded at 120, 140 and 160 px, and all four turns of each crop are searched, over 1,189 printings of 936 cards. A read is **confident** at p ≥ 0.85 (temperature 0.0347), the live runner's threshold for naming a card.
- **Halves:** each crop's centre against the table window's midline.
- **Legends:** as labelled on each half, so assumed read right.
- **Lists:** the finalists' published lists, 29 and 32 cards. Each has 56 main-board copies (40 main deck, the legend, 3 battlefields, 12 runes) and 9 or 10 side-board copies. All four export formats were read (the deck code, text with collector codes, the tourney sheet, JSON), and they agree card for card. The other two matches have no list; there, the final's lists stand in for a wrong list.
- **Hard and soft:** a hard prior lets only the allowed printings compete; a soft one adds a bonus to their cosine scores.

| Match | Crops (tracks) | Left / right half | Legends on the table |
|---|---:|---:|---|
| Swiss round 10 | 464 (77) | 119 / 345 | Emperor of the Sands · Rogue Assassin |
| A match mid-way through the day | 329 (62) | 248 / 81 | Wuju Bladesman · Heart of the Tempest |
| The final | 154 (37) | 134 / 20 | Fire Below the Mountain · Heart of the Tempest (taken from its list; no crop of it was labelled) |

## Swiss round 10: where the priors matter

"Named right" means the right card, in any printing. The fixed and broken counts compare each prior against the whole gallery.

| Prior | Named right | 95% interval | Confident | Confident and right | Fixed / broken |
|---|---:|---:|---:|---:|---:|
| None: the whole gallery | 91.8% | 89.0–94.0% | 74.4% | 95.1% | |
| Both legends' domains, every rune | 92.2% | 89.5–94.3% | 82.1% | 95.5% | 2 / 0 |
| Both legends' domains, runes held to them | 92.2% | 89.5–94.3% | 82.8% | 94.8% | 2 / 0 |
| Each half its legend's domains, every rune | 93.5% | 90.9–95.4% | 89.0% | 95.9% | 8 / 0 |
| **Each half its legend's domains, runes held to them** | **97.8%** | **96.1–98.8%** | **86.9%** | **100%** | **28 / 0** |
| Wrong lists (the final's), hard | 8.4% | 6.2–11.3% | 35.8% | 22.3% | 24 / 411 |
| Wrong lists, soft +0.01 | 91.4% | 88.5–93.6% | 73.9% | 95.0% | 0 / 2 |
| Wrong lists, soft +0.03 | 87.5% | 84.2–90.2% | 73.3% | 95.3% | 2 / 22 |
| Wrong lists, soft +0.12 | 72.8% | 68.6–76.7% | 74.4% | 92.8% | 7 / 95 |
| Wrong lists, by legend | 97.8% | 96.1–98.8% | 86.9% | 100% | 28 / 0 |
| Upper bound: each half's "list" is the cards it showed | 100% | 99.2–100% | 100% | 100% | 38 / 0 |

- **By legend:** neither wrong list names a legend on this table, so both halves fall back to their own legend's rule, and the result is the legend rule's.
- **The mistakes left under the legend rule** (10 crops):
  - the alternate-art Order Rune (OGN-214a) read as Zenith Blade (OGN-262), a Calm and Order spell (4 crops);
  - the legend Rogue Assassin (VEN-139) read as a battlefield (SFD-217, 4 crops) and as a spell (VEN-140, 2 crops).
- **Of the right cards, 11 crops show the wrong printing:**
  - the alternate-art Order Rune (OGN-214a) read as the standard one (VEN-R06, 9 crops);
  - VEN-R01 read as OGN-007 (2 crops): the Fury Rune, reprinted in a later set with the same art (a mean pixel difference of 10 of 255).

## The final: other printings and shared battlefields

Every prior names all 154 crops except one: each half held strictly to its own list, which scores 92.9%.

- **Printings.**
  - 99 crops show the printing listed.
  - 54 crops show a different printing of a listed card:
    - SFD-058a, an alternate-art champion unit (21 crops);
    - VEN-R02, a Calm Rune from a later set (17);
    - OGN-166a, an alternate-art Chaos Rune (9);
    - VEN-R03, a Mind Rune from a later set (7).
  - 1 crop is a token.
  
  A list read printing for printing would get every one of those 54 printings wrong, and the hover card would show an art the player is not using.
- **Battlefields.** Zaun Warrens (OGN-298), from the Heart of the Tempest list, lay on the other half for 11 crops. Held to each half's own list, all 11 are read wrong, as a Calm Rune and as Seat of Power. With both lists' battlefields allowed on both halves, the result is 154 of 154.
- **Printing mistakes left under every prior.**
  - VEN-R03 read as OGN-089 (7 crops): the Mind Rune's reprint in a later set.
  - UNL-T07 read as OGN-274 (1 crop): the Sprite token and its numbered printing.
  
  Both pairs share their art: the mean pixel difference between them is 4–7 of 255, against 60–83 between different arts. No prior can separate them, and the hover card shows the same picture either way.

## The mid-day match

Every prior names all 329 crops, except the wrong lists read hard: 36.8%, with 208 crops broken. By legend, the Heart of the Tempest list goes to the half with that legend, and the other half gets its legend's rule: 329 of 329. Nothing shows whether that half's player used exactly that list; nothing broke.

## What each prior allows

| Known | Printings that can compete (of 1,189) |
|---|---|
| Nothing | 1,189 (936 cards) |
| One legend, runes held to its domains | 391–406, including 6 of the 18 rune printings |
| A list of 29–32 cards | 36–42 printings of its cards; 50–56 with the tokens |

## What it does not show

- **What a right list gains.** The one match with the players' lists was already named in full. On the Swiss round, a right list would score between 97.8% and 100%. Every card of a legal list fits its legend's domains, so the list allows a subset of the legend rule's printings. A right list also holds every card shown, so it allows a superset of the upper bound's.
- **Legends read by the model.** The legends here come from the labels. Live, the prior must wait until the legend is pinned, after agreeing reads ([ARCHITECTURE §3.5](../ARCHITECTURE.md#35-matcher-priors-and-fusion)). A legend read wrong would hold its half to the wrong domains, just as a wrong list does.
- **More than one broadcast.** This is Barcelona only: 947 crops in three matches, and the Swiss round carries the result. Los Angeles and Shenyang come next.
- **The zone rule.** Legends appear only in the legend zone. That rule would remove the legend read as a battlefield or a spell (6 of the 10 crops left). It is not applied here.

## In the pipeline and the engine

The legend rule is now in the live runner (`live/pipeline.py`) and in the extension's engine (`packages/engine`), on by default. The shared code is `rifteye_ml/priors.py`, ported as `packages/engine/src/priors.ts`.
- **When it applies:** once a side's legend is pinned. Until then, the whole gallery competes.
- **What it covers:** the tracker's reads and the change gate's alike.

Measured on 29 September 2026:

- **Parity.** On the Los Angeles grand final replay, the engine gives Python's board and events at every step, with the rule on:
  - 240 of 240 steps in Chromium, and 18 of 18 in the camera-cut scenario.
- **The Los Angeles clip** (two minutes, 21 labelled crops). This match was already named at 99.4%, so there is little to gain:
  - With the rule, 12 crops are named right, 3 wrong and 4 unsure, and 2 have no track. Without it: 11 right, 2 wrong, 6 unsure, 2 no track.
  - One unsure Body Rune gets named.
  - One unsure spot is now named Sand Soldier where the label says Emperor's Dais. A token standing on that battlefield is the likely reason.
  - One impossible play goes: a Mind unit, which neither legend (Calm and Chaos, Fury and Body) allows.
- **A problem the rule does not touch.** Both runs announce plays at the bottom edge of the frame, where the broadcast shows its showdown banner. The detector takes the banner's art for cards. That needs the table window to leave the banner out, not a prior.
- **Not measured yet:** Barcelona end to end. The 97.8% above is per crop, with legends from the labels. The live runner has to read the legend first.

## Next

1. **Barcelona end to end,** with the legend read by the model, and the showdown banner kept out of the table window.
2. **Decklist import in the extension.** The viewer pastes each player's deck code or export. A list is used only on the half whose legend it names, each listed card stands for all its printings, and both lists' battlefields are allowed on both halves.
3. **The zone rule:** legends only in the legend zone, battlefields only in their slots.

A prior only helps name the cards face up on the table. It is never shown, and it is never used to guess or reveal a hand, a face-down card or the rest of a list (D-005).

## Reproducing it

```bash
cd ml && python -m rifteye_ml.decklist check --catalog catalog.jsonl deck.json deck.txt deck-tourney.txt deck-code.txt
python -m rifteye_ml.decklist evaluate --match final=12:18:30-12:44:00 --deck final=a.json --deck final=b.json \
    --match swiss10=03:29:00-03:57:30 --match midday=07:38:30-08:09:30 --out PRIVATE/m2-decklist-prior.json
```

The command reads the private labels, crops and embedding caches under `RIFTEYE_DATA`, and refuses to write its results inside the repository.
