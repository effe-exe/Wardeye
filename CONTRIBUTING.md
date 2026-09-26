# Contributing to RiftEye

Thanks for helping. RiftEye is in the **design phase**: the architecture and research docs are written and code lands milestone by milestone ([ROADMAP](docs/ROADMAP.md)). Useful contributions right now:

- **Critique the design.** Open an issue against anything in [`docs/`](docs/) that looks wrong, risky or naive.
- **Broadcast layouts.** Tell us which Riftbound broadcasts you watch and how they frame the table (see [Layout presets](#4-layout-presets)).
- **Prior art.** Links to tools, papers or datasets we missed.
- **Broadcaster contacts.** Organisers who might allow their VODs to be used for training ([why permission matters](docs/research/04-data-and-evaluation.md#44-stream-footage-the-data-engine)).

## 1. The Contributor Licence Agreement (CLA)

Before your first pull request can be merged, you sign the [RiftEye CLA](CLA.md). A bot comments on your PR with instructions. Signing takes one comment and covers all your future contributions.

**What it means, in plain words** (the [CLA](CLA.md) itself is what counts):

- **You keep the copyright** to your contribution.
- You give the Maintainer a broad, permanent licence to use it. That includes distributing it under licences other than the AGPL-3.0, **including commercial licences**. RiftEye's plan is AGPL for everyone, plus paid licences for organisations that cannot accept AGPL terms. That only works if the Maintainer can license every line of the project ([why](docs/research/07-licensing-and-governance.md)).
- Whatever other licences the Maintainer offers, **the version you contributed to stays available under the AGPL-3.0**. That cannot be revoked.
- You confirm the contribution is yours to give, for example that your employer does not own it.

If you cannot sign, you can still open issues, review designs and report bugs.

## 2. Ground rules

These are the rules reviewers enforce most often.

1. **No third-party media in the repository.** No stream frames, crops, VODs, audio, card images, playmat art or datasets, even small ones for tests. Use synthetic fixtures you generated yourself, or reference data by manifest.
2. **Permissive dependencies only.** Anything that ships (npm or pip packages, WASM, model weights) must be MIT, BSD, Apache-2.0, ISC, zlib or similarly permissive. No AGPL, GPL or LGPL code, and no "research-only" or "non-commercial" weights. Check the **weights'** licence separately from the code's. Say which licence it is in your PR.
3. **Public information only.** No feature may read hand cams, face-down cards or other hidden information.
4. **No secrets in client code.** The extension is public. Anything keyed goes behind a server.
5. **Respect Riot's IP.** Never add card images or card text to the repository or to release assets; the extension loads them from Riot's public card gallery at display time. No cross-match statistics (play rates, win rates). No Riot logos ([D-015](docs/decisions.md#d-015-no-riot-api-no-riot-assets-distributed), [08](docs/research/08-legal-and-policy.md)).
6. **Model changes need numbers.** A PR that changes a model or its pre-/post-processing includes a leaderboard row measured on the real test set. Maintainers run it for you if you do not have access ([04 §4.7](docs/research/04-data-and-evaluation.md#47-metrics-and-the-leaderboard)).
7. **AI-assisted contributions are welcome.** You are responsible for them as for your own work: review them, test them, and make sure they do not reproduce code under incompatible licences. You sign the CLA for the whole pull request.

## 3. Pull requests

- Keep PRs small and focused. Open an issue first for anything larger than a bug fix.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(extension): ...`, `fix(ml): ...`, `docs: ...`.
- Describe **what** changed and **how you verified it**.
- Run the checks before you push (see [Development setup](#development-setup)).

### Development setup

Requirements: Node 22+, Python 3.11+.

```bash
npm install                      # TypeScript workspaces: packages/*, apps/*
npm run check                    # typecheck, unit tests, media guard, licence guard
npm run test:e2e                 # browser tests (runs `npx playwright install chromium` once first)

cd ml && python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]' && pytest -q
```

| Path | What it is | Licence |
|---|---|---|
| `packages/schema` | Data formats and validators | Apache-2.0 |
| `apps/logger` | Timeline logger for ground truth | AGPL-3.0-only |
| `ml/` | Catalogue, stream simulator, encoders, M0 spike, index builder | AGPL-3.0-only |


## 4. Layout presets

A layout preset is a small JSON file (`layouts/<broadcaster>.json`) telling RiftEye where the overhead table camera sits in a broadcast's frame ([schema](docs/ARCHITECTURE.md#6-data-contracts)). To propose one:

1. Open an issue with the **public** VOD URL and a timestamp where the table view is visible.
2. Give the region coordinates as fractions of the frame (0–1). Include which side player A sits on, and the date range the layout was used.
3. Do not attach screenshots. A timestamp link is enough.

## 5. Labeling

Labeling happens on a private annotation server, because the frames belong to broadcasters who allowed their use on those terms. Labelers sign the CLA and a short confidentiality agreement. Open an issue with the `labeling` label if you want to help.

## 6. Conduct

Everyone in the project follows the [code of conduct](CODE_OF_CONDUCT.md).

## 7. Security

Please do not open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).
