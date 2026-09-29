# RiftEye

Open-source computer vision for Riftbound TCG streams: a timeline of every card played, and hover-to-inspect cards on Twitch/YouTube. The project is in M0. Start with `docs/ARCHITECTURE.md`, `docs/ROADMAP.md` and `docs/research/README.md`.

Checks: `npm run check` (typecheck + vitest + guards), `npm run test:e2e` (Playwright; locally set `RIFTEYE_CHROMIUM` to a Chromium binary if Playwright's own is missing), and `cd ml && pytest -q`.

## Non-negotiable rules

- **Licence hygiene.** RiftEye is AGPL-3.0 with a CLA. It is free for everyone, and its trained weights are not published (D-022). Every dependency that ships (npm, pip, model weights) must be MIT, BSD, Apache-2.0, ISC, zlib, or another permissive licence. No AGPL/GPL/LGPL code, no Ultralytics, no weights licensed "research only" or "non-commercial". Check the licence of the *weights*, not just the code. Record new dependencies in `docs/research/03-models-and-licensing.md`.
- **No third-party media in git.** Never commit stream frames, crops, VODs, audio, card images, playmat art or datasets. Reference datasets by manifest (sha256) only. `.gitignore` already blocks the common extensions. Do not force-add them.
- **Public information only.** Never process hand cams, face-down cards or any hidden information, even if a broadcast shows it.
- **No secrets in the client.** The extension is public code. Anything that needs an API key goes behind a server.
- **Riot IP (decision D-015).** RiftEye does not use the Riot API. Never bundle, commit, re-host or cache on a server any card image or card text. The extension reads them from Riot's public card gallery in the viewer's browser, and releases ship only models plus a vector index keyed by collector code. No Riot logos. Keep both Riot notices in `NOTICE` and the README. RiftEye stays free and non-commercial for Riftbound.
- **No cross-match statistics** (D-012). Never compute, store or publish play rates, win rates or matchup stats. Per-match timelines only.
- **No GPL surfaces.** No native OBS plugin (OBS is GPL); the broadcaster kit talks to obs-websocket (D-014).

## Engineering conventions

- TypeScript strict for `apps/*` and `packages/*`. Python 3.11+ for `ml/*`. Models are exchanged as ONNX.
- `packages/core` has no DOM and no ML runtime, so it can run in browser, worker and Node.
- The index manifest names its encoder. Matching refuses to run on a mismatch; never "fix" this by loosening the check.
- Model changes need a leaderboard row (`ml/evalsuite`) measured on the **real** test set. Synthetic accuracy is not a result.
- Commits follow Conventional Commits (`feat(extension): ...`, `fix(ml): ...`, `docs: ...`).
