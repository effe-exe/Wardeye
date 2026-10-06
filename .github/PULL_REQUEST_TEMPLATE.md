## What changed

## How it was verified

## Checklist

- [ ] I have signed the [CLA](https://github.com/effe-exe/wardeye/blob/main/CLA.md) (the bot will ask on your first PR)
- [ ] No stream frames, crops, VODs, audio, card images or datasets are added
- [ ] New dependencies (code **and** model weights) are permissively licensed, and named here with their licence
- [ ] Nothing reads hand cams, face-down cards or other hidden information
- [ ] Model or pre-/post-processing changes are measured on the real held-out broadcasts, or ask a maintainer to run it
- [ ] Tracker changes are made in `ml/rifteye_ml/live/pipeline.py` and `packages/engine/src/recognizer.ts` alike, with the vectors regenerated
- [ ] A change in behaviour has a decision record in `docs/decisions.md`
- [ ] `npm run check` and `pytest -q` (in `ml/`) pass
