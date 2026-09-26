# 09: Risks and open questions

## 9.1 Risk register

| # | Risk | Likelihood | Impact | Mitigation | Early signal |
|---|---|---|---|---|---|
| R1 | **Riot declines, delays or later revokes the API key.** Without it there are no card assets and no monetisation ([08 §8.2](08-legal-and-policy.md#82-riot-games-policies)) | Medium | **Critical** | Apply in M0 with a clear spectator-companion mock-up. Ask the Developer Support Platform first. Keep the engine game-agnostic ([D-008](../decisions.md#d-008-game-agnostic-core-riftbound-as-the-first-game-pack)) | Application status; support reply |
| R2 | **Accuracy at stream resolution is too low** (hypotheses H1–H3 fail) | Medium | High | Priors (legend domains, decklists, rune-tap cost), production graphics, caster audio, VOD mode; target official broadcasts first | The M0 spike's accuracy-vs-pixels curves |
| R3 | **Organisers do not share footage** | Medium | High | Synthetic data; **RiftEye's own recording sessions** at local events with players' consent (full rights, known camera setup); graphics-derived labels | Replies to M0 outreach |
| R4 | **Browser performance** on common laptops | Medium | Medium | CNN or OBB detectors, detect-on-ROI, identify-once-per-track, eco mode, server timelines for known VODs | M2 benchmarks |
| R5 | **Competition** (RiftSight for hover, Riftbound Vision for broadcasters) | High | Medium | Lead with the timeline and measured accuracy; open tooling; consider collaborating ([06](06-prior-art-and-starting-point.md)) | Their releases |
| R6 | **Name conflict** ("Rift", RiftSight, an existing "RiftEye") | Medium | Medium | Choose a neutral name before any public launch or trademark filing | Clearance search |
| R7 | **Dependency licences change** (as DEIMv2 and MobileCLIP did) | Medium | Medium | Pin versions and record licence-check dates; CI licence audit; prefer foundation-backed permissive projects | Upstream licence diffs |
| R8 | **Platform churn** (Twitch or YouTube DOM, Chrome policies, embed rules) | High | Low–Medium | One adapter per site with tests; bundled code; monitor store policy updates | Breakage reports |
| R9 | **Set cadence** (a new set every ~3 months, preview seasons) | Certain | Medium | Zero-shot index updates from catalogue art; a release checklist per set; preview labels | Set announcements |
| R10 | **Misuse for hidden information** or live assistance | Low | High (trust) | No hand cams or face-down cards; public-information rule; the spectator framing | Organiser feedback |
| R11 | **Metagame-data line** unclear | Medium | Medium | No cross-match aggregates until Riot confirms | Riot's answer |
| R12 | **Solo-maintainer bandwidth** | High | Medium | Tight scope per milestone; contributor-friendly tasks (presets, labeling); defer surfaces | Milestone slippage |

## 9.2 Open questions

1. **Riot:** is a spectator tool that identifies cards in broadcasts an approvable use case? Could it be official, for example feeding the board-state overlay Riot's broadcast team already built?
2. **Official broadcast specs:** actual bitrates and renditions; whether hand cams are shown; how long the time shift is; whether clean table-camera recordings can be obtained.
3. **Physical card size:** assumed to be the standard 63 × 88 mm, which matches the official image ratio. Confirm with a real card.
4. **Browser detector:** does RT-DETRv2-OBB export cleanly to ONNX and run fast on WebGPU, or does D-FINE plus the corner refiner win?
5. **The fp16 story on WebGPU** for the chosen detector and embedder. Validate it per model.
6. **Inference host in the extension:** an injected extension iframe vs a content-script worker vs `tabCapture` into an offscreen document ([05 §5.2](05-delivery-surfaces.md#52-browser-extension-manifest-v3)).
7. **Weights licensing:** is AGPL the right default for community weights, and is it enforceable at all? Revisit with counsel before the first model release.
8. **Name:** decide before the Riot application, so the application and the product carry the final name.

## 9.3 Kill criteria

Stop or pivot the Riftbound-specific product if:

- Riot refuses a key for this use case after a clear application; **or**
- after M1, card-level committed precision stays below 90% at usable coverage on official broadcasts, even with priors.

In either case the engine, the evaluation harness and the data tooling remain useful for other paper TCGs with different publishers and policies ([D-008](../decisions.md#d-008-game-agnostic-core-riftbound-as-the-first-game-pack)).
