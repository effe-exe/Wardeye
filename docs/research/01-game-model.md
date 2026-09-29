# 01: Riftbound, as a computer-vision problem

What Wardeye needs to know about the game: the card pool it must recognise, what the table looks like, which events are visible from an overhead camera, and which rules shrink the search space. Everything here is current as of **September 2026**. The game adds a set roughly every three months, so the numbers go stale fast: update this chapter with each release.

Rule numbers refer to the *Riftbound Core Rules* (last updated 2026-07-16) and the *Riftbound Tournament Rules* (last updated 2026-07-16) [R1, R2].

## 1.1 The card pool

| Set (code) | English release | Main set | Printings in the official gallery | Notes |
|---|---|---|---|---|
| Origins (OGN) | 2025-10-31 (China 2025-08-01) | 298 | 352 | Alternate arts (`a`), overnumbered legends, signature (`*`) cards [R3, R4] |
| Proving Grounds (OGS) | 2025-10-31 | 24 | 24 | Starter box [R4] |
| Spiritforged (SFD) | 2026-02-13 (China 2025-12-12) | 221 | 288 | [R3, R5] |
| Unleashed (UNL) | 2026-05-08 (China 2026-04-10) | 219 | 288 | Includes the first "Ultimate Rare", Baron Nashor [R3, R5] |
| Vendetta (VEN) | 2026-07-31 (worldwide) | 166 | 228 on 2026-07-28, 237 by 2026-08-21 | Release dates synchronised with China from here on [R3, R4, R5] |
| Radiance (RAD) | 2026-10-23 | ~180 + 60+ showcase | — | Preview season runs 2026-09-25 to 2026-10-09 [R6] |
| Legacy (set 6) | 2027-01-29 | 346 + 93 showcase | — | One common slot becomes a Legend-or-Battlefield slot [R7] |

Counts computed from a snapshot of the official card gallery (2026-07-28) [R3]:

- **1,180 printings** across OGN, OGS, SFD, UNL and VEN, with **935 distinct card names**. 179 names have 2–5 printings.
- **By type:** 629 units, 233 spells, 114 gear, 118 legends, 66 battlefields, 18 runes.
- **Beyond the gallery:** promos (organised-play, judge, event and regional promos) and special products bring community databases to about 1,420–1,450 printings [R4].
- **After Radiance,** expect about 1,650–1,700 printings before counting language and foil variants.

**Languages:** English, Simplified Chinese, French (since 2026-05-29), Korean (Origins launched 2026-09-18) and Traditional Chinese [R4, R8]. Cards in any language are tournament-legal, and different-language printings count as the same card (Core Rules 132.3; Tournament Rules 420.1). **Identification must rely on art, not text.**

## 1.2 Physical facts that matter to the vision pipeline

| Card type | Orientation | Where it lives | Notes |
|---|---|---|---|
| Legend | Portrait | Legend Zone, face up all game | Always exactly **two domains**; sets the deck's domain identity (103.1) |
| Unit (incl. Champion and Signature) | Portrait | Base or a battlefield | **Enters the board exhausted** unless Accelerate or similar (143.4) |
| Chosen Champion | Portrait | Champion Zone at start, then played | Shares the Legend's champion tag |
| Spell | Portrait | Goes on the Chain, then to the Trash | Visible briefly |
| Gear (Equipment attaches to units) | Portrait | Base, or attached to a unit | Attached equipment is tucked so only its effect text and bonus show |
| Rune | Portrait | Rune deck (12), channeled into Base | Six basic runes plus alternate arts |
| Battlefield | **Landscape** (all 66 printings) | Battlefield Zone; each player brings 3 and uses 1 | Two on the table in 1v1 |
| Token | Card objects | Created during play | Printed token cards exist; tournament rules require card objects that show ready state (508.8) |

- **Three card backs:** main deck, rune deck, and battlefields plus legends (129.2). The back tells you which pile a face-down card belongs to.
- **Image geometry:** official images are 744 × 1039 px (battlefields 1039 × 744), the standard 63 × 88 mm card ratio [R4].
- **Face layout:** energy cost upper-left, name roughly mid-card, domain symbols lower-right, and set code, collector number and rarity along the bottom edge. At stream resolution, only the art and the overall colour layout survive ([02 §2.2](02-vision-pipeline.md#22-how-big-is-a-card-on-a-stream)).
- **Collector numbers:** `OGN-001/298` (standard), `OGN-007a/298` (alternate art), `OGN-304*/298` (signature), `UNL-238/219` (overnumbered), `UNL-T01` (token), `VEN-R01` (rune), `VEN-SP1/006` (special) [R3, R4].

## 1.3 The table in a 1v1 match

Zones (106–109):

- **On the board:** each player's **Base** (units, gear and runes), the **Battlefield Zone** (each battlefield has a one-card facedown zone), and each player's **Legend Zone**.
- **Off the board but public:** the **Chain**, the **Trash** (face up), the **Champion Zone**, Banishment.
- **Secret:** main deck, rune deck, and **hand**. The hand's card count is public, its contents are not.

**Tournament table layout** (Tournament Rules 508; required at high-level play):

- **Rows:** runes sit closest to each player, and everything else is further in, toward the opponent (508.2–508.3).
- **Sides:** main deck and trash sit together on one side, and the rune deck on the other (508.4–508.5). Legend and Chosen Champion sit together on one side, with the legend between the other cards and the champion (508.6).
- **Orientation:** **ready cards face their controller, and exhausted cards are rotated 90°, all in the same direction** (508.10). For Wardeye this is a gift. A card's facing tells you who controls it, even on the shared battlefields in the middle, and its rotation tells you whether it is exhausted.

## 1.4 Turn structure and what the camera sees

| Phase (315–317) | What happens | Visible signature |
|---|---|---|
| Awaken | Turn player readies everything they control | **Mass rotation back to ready on one side → `turn_start`** |
| Beginning | Start-of-turn effects; score "Hold" on controlled battlefields | Point tracker changes |
| Channel | Channel 2 runes from the rune deck; 3 on the second player's first turn [R16] | **Two runes (three) appear in that player's rune row** |
| Draw | Draw 1 (an empty deck means "Burn Out") | Hidden |
| Main | Play cards, move units, spells, showdowns, combat | Most events |
| Ending | Heal, "this turn" effects end, rune pool empties | Little |

Scoring is to **8 points**. Players score by conquering and holding battlefields, and a battlefield can be scored once per turn (194, 467–472). In high-level play, scores must be tracked in writing (Tournament Rules 415.3). The broadcast's own point track is usually easier to read.

## 1.5 Events and their visual signatures

| Wardeye event | Visual evidence | Rules |
|---|---|---|
| `card_played` (unit) | New face-up card in Base or at a battlefield, arriving **exhausted** unless it has Accelerate | 143.4, 355.2 |
| `card_played` (gear) | New card in Base, arriving ready | 149 |
| `spell_cast` | Card appears briefly (the Chain) and then lands face up on the owner's trash | 108.1, 157 |
| `card_played` (Chosen Champion) | Card leaves the Champion Zone and enters play | 108.3.d |
| `card_hidden` | A face-down card is placed at a battlefield (Hide). **Identity is secret and never inferred.** | 421, 811 |
| `card_revealed` | That face-down card turns face up (played for 0) | 811 |
| `runes_channeled` | Two new runes in the rune row, three on the second player's first turn | 315.3, 430 |
| Energy paid | Runes rotate (exhaust). **The number of runes tapped is the energy cost of the card being played.** | 164.2 |
| Power paid | A rune goes back to the bottom of the rune deck | 164.2 |
| `card_moved` | A unit, exhausting, moves between Base and a battlefield (or battlefield to battlefield) | 144 |
| Legend ability | The legend rotates | 174.8 |
| `turn_start` | All of one player's objects rotate back to ready | 315.1 |
| Take-back | Players may reverse their most recent action (509.3–509.4), so the engine must **retract** an event whose card leaves the board within seconds | 509 |

The **rune-tap cost cue** deserves emphasis. When a player taps *N* runes and then a new card appears, that card very likely costs *N* energy. The card's printed cost is in the catalogue, so this is a free, strong prior on its identity ([02 §2.8](02-vision-pipeline.md#28-from-similarities-to-a-committed-identity)).

## 1.6 Deck rules as recognition priors

Registration requirements (Tournament Rules 402.1; Core Rules 103):

- A main deck of **exactly 40 cards**, including the Chosen Champion.
- **1 legend**, **12 runes** and **3 battlefields** with unique names.
- A sideboard of at most 10 cards; runes, legend and battlefields stay fixed.
- Every card must fit the legend's **domain identity**. A multi-domain card needs all of its domains.
- The Chosen Champion shares the legend's champion tag.
- At most **3 copies** of a named card, and at most 3 signature cards, all tagged to the legend.

Measured on the 2026-07-28 gallery snapshot, the effect on the candidate set per player is:

| What is known | Candidate main-deck printings |
|---|---|
| Nothing | 976 units, spells and gear |
| **Legend identified** (two domains, plus colourless) | **314–324** (about a third of the pool, about 260 names) |
| **Published decklist** | **40 cards**, typically 15–25 distinct names |

The legend is the easiest card on the table to identify: it is large, static, face up from the first frame, and there are only 118 legend printings. So the domain prior is available almost immediately in every match. Professional events should run with **open decklists** (Tournament Rules 401.5.b), and qualifier top decks are published afterwards, which makes the decklist prior available for many VODs.

## 1.7 Formats

- **1v1:** best-of-1 ("Duel") or best-of-3 ("Match"); organised play is mostly best-of-3 (Core Rules 485–489; Tournament Rules 404.2).
- **Other modes:** 2v2 to 11 points, and 3–4 player free-for-all. These come after v1.
- **Rounds:** Swiss rounds of 60 minutes. When time runs out, the current turn ends plus 3 more turns (Tournament Rules 408.2).

## 1.8 The broadcast landscape

**Organised play.** 2026 has weekly Nexus Nights and Summoner Skirmishes, then Regional Qualifiers: Utrecht, Bologna, Barcelona and others in Europe, plus Atlanta, Las Vegas, Vancouver, Hartford, Singapore and Los Angeles. Regional Championships follow: Stuttgart on 2026-11-06 to 08, Las Vegas on 2026-12-11 to 13. From 2027 the structure becomes Regionals and Continentals, leading to Worlds 2027 [R9]. Third-party "Showdown Series" events and France's "Le Rift Tour" were added in May 2026 [R10].

**Official channels:** twitch.tv/riftbound and youtube.com/@riftbound [R11].

**Production** [R12, R13]:

- US qualifier broadcasts use a time-shifted replay workflow: matches are captured independently and aired back-to-back. Official VODs are therefore *not* live, which lowers the stream-sniping risk. Community streams may be live.
- Riot commissioned a board-state overlay and a replay system in which casters move cards on screen. **Official graphics already show the featured card.** Matching that graphic against the catalogue gives exact, time-stamped labels for training and evaluation ([04](04-data-and-evaluation.md)).
- A community graphics package modelled on the qualifier layout crops the table camera to about **1214 × 1080** inside a 1920 × 1080 frame [R14]. That puts a card's long side around 100–140 px, consistent with the scale table in [02 §2.2](02-vision-pipeline.md#22-how-big-is-a-card-on-a-stream).

**Example VODs (YouTube):** Barcelona top 8 `irYSCjcbdPs`, Vancouver top 8 `_PEjx43XIMM`, Singapore top 8 `7ft1nhgYjr0`, Hartford top 8 `kMxYbM1Aumc` [R15].

## 1.9 Implications for Wardeye

1. **Art-first identification.** Text is unreadable at stream scale, and any language is legal.
2. **Legend → domains → a third of the pool,** usable from the first seconds of every match. The rune-tap cost cue and the decklist prior stack on top of it.
3. **Orientation is information.** Facing tells you the controller, and a 90° rotation means exhausted (Tournament Rules 508.10).
4. **Hide is hidden.** A face-down card at a battlefield is recorded as `card_hidden`, with no identity until it is revealed.
5. **A new set every three months.** The index must accept new printings zero-shot from catalogue art on release day, with preview cards labeled as unreleased, as Riot's policy requires ([08](08-legal-and-policy.md)).
6. **Official graphics are free labels.** The featured-card overlay should feed the data engine from day one.

## Sources

- [R1] Riftbound Core Rules, 2026-07-16. playriftbound.com/en-us/rules-hub. Text mirror: `github.com/DevsD20/riftbound-core-rules-text`
- [R2] Riftbound Tournament Rules, 2026-07-16. Text mirror: `github.com/benediktwerner/riftbound-archive` (`Tournament-Rules-2026-07-17.txt`)
- [R3] Official card gallery snapshot, 2026-07-28, via `github.com/slimtreble/Riftbound-card-data`. The counts in this chapter were computed from it.
- [R4] Third-party source map, re-verified 2026-08-21: `github.com/aimanzahar/riftbound_inventory` (`docs/sources.md`). Collectability announcement: playriftbound.com/en-us/news/announcements/collectability-in-riftbound-origins/
- [R5] Release dates: sheepesports.com (China launch), riftdaily.com/riftbound-release-date/, playriftbound.com/en-us/news/announcements/the-vendetta-overview/
- [R6] playriftbound.com/en-us/news/announcements/the-radiance-overview/ ; riftbound.gg/radiance/
- [R7] playriftbound.com/en-us/news/announcements/products-and-sets-into-2027/ ; gamespot.com (2027 lineup)
- [R8] playriftbound.com/en-us/news/announcements/koreas-rift-opens-on-september-18/ ; French roadmap announcement on riftbound.leagueoflegends.com
- [R9] playriftbound.com/en-us/news/organizedplay/riftbound-organized-play/ ; …/2026-regional-championship-info/ ; …/riftbound-premier-play-early-2027-events/
- [R10] playriftbound.com/en-us/news/organizedplay/announcing-the-showdown-series-le-rift-tour/
- [R11] playriftbound.com/en-us/news/organizedplay/all-eyes-on-barcelona/
- [R12] weareatomic.com/success-stories/riftbound-regional-qualifiers/
- [R13] esportsinsider.com/2025/10/riot-games-riftbound-esports-tcg-interview
- [R14] `github.com/sammor327/sideways-studio`
- [R15] youtube.com/@riftbound (videos listed above)
- [R16] The second player's first Channel: the Maintainer; runesandrift.com/riftbound-turn-order/ (2026-09-28)
