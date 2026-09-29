# @rifteye/engine (M2, in progress)

RiftEye's recognition pipeline in TypeScript, so the extension can run it in the browser without the live runner. It is a port of `ml/rifteye_ml`. Every part is checked against its Python original on the same input, down to the pixel where Pillow does the work.

| Module | Ported from | What it does |
|---|---|---|
| `types.ts` | | The shared types: `RgbImage`, `CardBox`, `Detection`, `Layout`, `CatalogRow`, `Encoder`, `Finder` |
| `image.ts` | Pillow (`Resample.c`, `Geometry.c`, `Convert.c`) | Crop, resize, rotate, grey, paste: the same bytes as Pillow |
| `geometry.ts`, `detector.ts` | `detect/geometry.py`, `detect/model.py`, RF-DETR's `PostProcess`, `live/pipeline.detector_boxes` | 576 px tiles, the head's outputs to cards, the tile merge, the tracker's boxes |
| `embedder.ts`, `ort.ts` | `encoders.letterbox`, `embed/onnx.py` | Crops to the embedder's input; ONNX Runtime Web sessions (native WebGPU, float16 where the GPU has `shader-f16`, WASM otherwise) |
| `retrieval.ts`, `lsap.ts`, `matcrops.ts`, `changegate.ts`, `layouts.ts`, `autolayout.ts` | `retrieval.py`, SciPy's `linear_sum_assignment`, `matcrops.py`, `changegate.py`, `live/layouts.py`, `live/autolayout.py` | The gallery search, the assignment, the mat and card tests, the change gate, the broadcast layouts, finding the table |
| `recognizer.ts` | `live/pipeline.py` | The tracker: tracks, reads, legends, stacks, plays, and the state the overlay draws |

## Checks

- **Unit tests** (`test/*.test.ts`) run everywhere. They need no private data: synthetic inputs, plus expected values Python computed on the same seeded inputs.
- **Parity tests** compare the port with Python on real frames. They skip unless `RIFTEYE_M3` names the private folder, e.g. `RIFTEYE_M3=~/rifteye-data/m3`, which holds:
  - `frames/la-final/`: 240 JPEG frames of the LA final (2 min at 2 fps) and `frames.json` with each frame's time and the SHA-256 of its decoded RGB. Pillow and Chromium decode them to the same bytes.
  - `fixtures/<part>/`: what Python computed on those frames.
- **Browser tests** (`e2e/*.spec.ts`) run in Chromium through Playwright: JPEG decoding and ONNX Runtime Web as the extension uses them.

The frames and fixtures come from broadcasts and the models trained on card art. They are private ([D-006](../../docs/decisions.md)) and never enter the repository.
