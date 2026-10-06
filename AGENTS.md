# Wardeye: notes for coding agents (and people)

Wardeye is a free, open-source Chrome and Edge extension that names the face-up cards on the table in Riftbound
Twitch streams, live, in the viewer's browser. It is on the Chrome Web Store. It is a community project by Federico
Vietti, not a Gradeon product.

This file is the one set of instructions for every coding agent (`CLAUDE.md` points here). People: start with
[CONTRIBUTING.md](CONTRIBUTING.md), whose "Finding your way around" shows how a frame travels through the code.

## Commands

```bash
npm install                       # Node 22 (.nvmrc); TypeScript workspaces: packages/*, apps/*
npm run check                     # typecheck, unit tests, and the media and licence guards: run before every commit
npm run test:e2e                  # Playwright in Chromium; set RIFTEYE_CHROMIUM=/path/to/chromium if Playwright has none

cd ml && python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
pytest -q                         # the Python side, in ml/
```

- **Private data is optional.** The tests that need it skip without it:
  - `RIFTEYE_M3`: real frames and Python's results on them, for the parity tests;
  - `RIFTEYE_DATA`: models and assets, for packaging.
- **None of that data is ever committed** (see the rules below).

## Where things are

| Path | What it is |
|---|---|
| `apps/extension` | The extension: the overlay (`src/content.ts`), the background worker, the engine document and worker, the side panel. Its README describes every file |
| `packages/engine` | The recognition pipeline in TypeScript: detector, embedder, gallery search, tracker (`src/recognizer.ts`) |
| `packages/schema` | Data formats and validators (Apache-2.0) |
| `ml/` | Python: catalogue, synthetic boards, training, evaluation, and `rifteye_ml/live/pipeline.py`, the reference tracker |
| `docs/` | Decisions, architecture, roadmap, privacy, releasing, research and reports. Index: [docs/README.md](docs/README.md) |
| `apps/bench`, `apps/logger`, `apps/reviewer`, `apps/viewer` | Model timing, a timeline logger, a label reviewer, a match viewer |

## How to change things

- **The tracker (what counts as a card, a play, a move): Python first.**
  1. Change `ml/rifteye_ml/live/pipeline.py`, the reference.
  2. Make the same change in `packages/engine/src/recognizer.ts`.
  3. Add a case to `packages/engine/test/gen/recognizer.py`, then regenerate its vectors:
     `RIFTEYE_DATA=/tmp python packages/engine/test/gen/recognizer.py`.
  4. Add a test on each side.

  The parity tests fail when the two engines disagree. Never loosen a parity check to make it pass.
- **The gallery names the embedder it was made with,** and the engine refuses another. Never loosen that check:
  rebuild the gallery (`ml/rifteye_ml/web_assets.py`).
- **A change in behaviour gets a decision record.** Append it to `docs/decisions.md` (`D-0NN`): date, status,
  decision, and why, with the measured numbers. Records are never edited afterwards; a new one supersedes an old one.
- **Numbers are measured, never assumed.**
  - Accuracy claims come from the real held-out broadcasts, not from synthetic boards.
  - A play counted as real or false is checked against the frames.
  - Say what was measured, on what, and what was not.
- **Keep changes small,** with tests and docs in the same commit. Commits follow Conventional Commits
  (`feat(extension): …`, `fix(engine, ml): …`, `docs: …`).
- **Style:**
  - TypeScript strict (no unused locals or parameters, no `var`, strict equality). Python 3.11+.
  - Comments and docs in plain English: short sentences that say why, not what.
  - Match the surrounding code.

## Rules that are never broken

- **Public information only (D-005).** Never process hand cams, face-down cards or any hidden information, even if a
  broadcast shows it.
- **Nothing leaves the viewer's machine (D-006).** Frames, crops, VODs and models stay local.
- **No third-party media in git.** No stream frames, crops, VODs, audio, card images, playmat art or datasets, even
  small ones for tests. Use synthetic fixtures, or reference data by manifest. `npm run check:media` guards this.
- **Permissive dependencies only (D-002).** MIT, BSD, Apache-2.0, ISC, zlib or similar, for code and for model
  weights. No AGPL, GPL or LGPL, no Ultralytics, no "research only" or "non-commercial" weights.
  `npm run check:licenses` guards the shipped ones. Record a new dependency in
  `docs/research/03-models-and-licensing.md`.
- **No GPL surfaces (D-014).** No native OBS plugin (OBS is GPL); a broadcaster kit talks to obs-websocket.
- **Riot (D-015).** No Riot API.
  - Card images and text are never bundled, committed or re-hosted: the extension reads them from Riot's public card
    gallery in the viewer's browser.
  - No Riot logos, and nothing that imitates Riot's design.
  - Keep both Riot notices in `NOTICE` and the README.
- **Free for everyone (D-022).** No paid tiers, accounts or ads. The trained weights stay private: never commit them.
- **No cross-match statistics (D-012).** Per-match timelines only: no play rates, win rates or matchup stats.
- **Name and brand (D-020, D-024).**
  - The product is Wardeye. Public copy never says RiftEye. The internal names (`rifteye_ml`, `@rifteye/*`,
    `RIFTEYE_*`) stay.
  - Pages and the overlay follow the brand book in `assets/brand/`.
- **No secrets in the client.** The extension is public code.
