# RiftEye bench (M2 feasibility)

Can RiftEye's two models run inside Chrome, fast enough and with the same answers? This extension page runs the card detector and the card embedder with ONNX Runtime Web: the native WebGPU runtime, the older WebGPU runtime (JSEP), and WASM with threads. For each model, precision and runtime it reports:

- how long the runtime and the model take to load;
- the first run (which includes shader compilation);
- the median and p90 time per batch;
- whether the outputs match PyTorch's on a real tile and on real crops.

It ends with an estimate of reads per second: one processed frame's tiles and crops, detector then embedder. Nothing leaves the computer.

The models and their check files are private ([D-006](../../docs/decisions.md)) and never enter the repository. The .onnx files come from `python -m rifteye_ml.detect onnx` and `python -m rifteye_ml.embed onnx` (ml/). Each model has a manifest, `<id>.bench.json`, listed in `models/index.json`. It names the variants, the input and outputs, the batches to time, the tiles or crops one frame costs (`per_frame.items`) and the check. The check gives an input, the expected outputs, a metric (`maxabs` or `cosine`) and a tolerance per precision. A detector's check can also name expected cards (`check.detections`: a file of Python's detections on the check tile, a score threshold, and per precision how far a card's corners, in px, and score may be off): the outputs are then decoded into cards with the engine's port of the detector's postprocess (`@rifteye/engine`) and matched to Python's, which judges float16 where comparing the outputs query by query cannot (it reorders near-tied queries). The rule is in `src/decoded.ts`; it has its own verdict next to the raw check's. The format is parsed in `src/model-manifest.ts`.

## Use it

```bash
npm run build -w @rifteye/bench
npm run pack -w @rifteye/bench -- --out rifteye-bench.zip    # fp16 models only; --precisions fp32,fp16 for both
```

The models come from `$RIFTEYE_DATA/models/onnx` (default `~/rifteye-data/models/onnx`), or from `--models DIR`.

To run it:
1. Unzip the file.
2. Open `chrome://extensions`, turn on Developer mode, click **Load unpacked** and pick the `rifteye-bench` folder.
3. The bench opens in a tab (or click the extension's toolbar button). Press **Run**, then **Copy results**.

Options in the page's address:
- `?runtimes=wasm,webgpu,webgpu-jsep` runs only those runtimes.
- `?quick=1` does one warm-up and one timed run per batch.
- `?stall=SECONDS` sets how long a silent worker is waited for.

## Files

- `src/bench.ts`: the page. It plans a row per model × variant × runtime, runs each row in a fresh worker, and shows the table and the summary to copy.
- `src/row-runner.ts`: one row. It starts the runtime on a one-node model, loads the model, times every batch, checks the outputs and releases the session.
- `src/decoded.ts`: the decoded detector check: the check tile's cards, matched to Python's.
- `src/bench-webgpu.ts`, `src/bench-jsep.ts`, `src/bench-wasm.ts`: one worker per ONNX Runtime Web build. The builds are bundled in `dist/ort/`, never fetched: an extension may not load remote code.
- The pure parts are unit-tested: `stats.ts`, `compare.ts`, `model-manifest.ts`, `plan.ts`, `estimate.ts`, `summary.ts`, `env.ts`, `decoded.ts`, and `tiny-onnx.ts` (the stand-in models).
- `pack.mjs`: the zip for someone who runs the bench. The models go in through symlinks, never copied.

## Tests

```bash
npx vitest run apps/bench
npm run build && npx playwright test apps/bench    # stand-in models made at test time: no models, no media
```
