# 03: Models, runtimes and their licences

Every component here was checked for **two** licences: the code's and the **weights'**. RiftEye's dual-licensing plan only works if everything that ships is permissive ([D-002](../decisions.md#d-002-permissive-dependencies-only)). Licences change: two of the models below changed terms during 2025–2026. **Re-check before adopting anything, and record the date you checked.**

State as of **2026-09-26**.

## 3.1 Headline findings

1. **The MobileCLIP and MobileCLIP2 weights are research-only.** On 2025-08-29 Apple replaced the permissive weights licence with the *Apple Machine Learning Research Model* terms. They allow use "exclusively for Research Purposes", which excludes use in commercial products and product development, and the terms extend to fine-tuned "Model Derivatives". **Nothing derived from MobileCLIP can ship in RiftEye.**
2. **DEIMv2 became non-commercial on 2026-08-24** (commit `bb64e5e` replaced Apache-2.0 with the "DEIMv2 License"). Avoid it, and avoid EdgeCrafter from the same lab.
3. **RF-DETR 1.11.0 (2026-09-24) is Apache-2.0** for:
   - detection sizes N, S, M and L;
   - **every segmentation size**;
   - a **keypoint model** that can be fine-tuned with any number of keypoints, so four card corners work out of the box.

   Only the XL and 2XL detectors are under Roboflow's PML licence, which requires a platform account.
4. **Ultralytics YOLO11 and YOLO26** are excellent (native OBB, pose, NMS-free heads) but AGPL-3.0. Ultralytics treats trained weights as covered too. Including them would permanently block a commercial RiftEye edition.
5. **ViT-backbone DETRs are slow in today's browsers.** Public demos show RF-DETR-N at about 0.7–1 s per frame on integrated-GPU WebGPU at fp32, with fp16 silently returning nothing. **Browser: small CNN-backbone or OBB detectors. Server: RF-DETR.**

## 3.2 Detectors

| Model | Code / weights licence | Params | COCO AP (latency, T4 TRT fp16) | Output | Verdict |
|---|---|---|---|---|---|
| **RF-DETR N/S/M/L** | Apache-2.0 / Apache-2.0 | 30.5–33.9M | 48.4–56.5 (2.3–6.8 ms) | Boxes; **Seg** (all sizes); **Keypoint** preview (custom K) | ✔ **Server and teacher** |
| RF-DETR XL/2XL | PML-1.0 | 126M | 58.6–60.1 | Boxes | ✖ Platform licence |
| **RT-DETRv2-OBB** (RiO-DETR, ECCV'26) | Apache-2.0 / Apache-2.0 | N 4.0M, S 8.2M | DOTA-v1 AP50 69.8 (N), 78.1 (S) | **Native oriented boxes** | ✔ **Browser candidate A** |
| **D-FINE** N/S | Apache-2.0 / Apache-2.0 | 4M / 10M | 42.8 / 48.5 | Boxes | ✔ **Browser candidate B**, with the corner refiner |
| DEIM (v1) | Apache-2.0 | 4–62M | 43.0–56.5 | Boxes | ✔ Alternative to D-FINE |
| RT-DETR v1–v4 | Apache-2.0 | 20–76M | up to 57.0 | Boxes | ✔ (v4 used a DINOv3 teacher in training only) |
| LW-DETR | Apache-2.0 | 12–118M | 42.9–58.3 | Boxes | ✔ |
| PP-YOLOE-R, RTMDet-R | Apache-2.0 | ~5–8M (small) | DOTA 73.8 / 75.4 | OBB | ✔ Fallback OBB options |
| YOLOX, NanoDet-Plus, PP-PicoDet | Apache-2.0 | 0.7–9M | 25.8–42.6 | Boxes | ✔ Very small, weaker |
| DEIMv2, EdgeCrafter | **Non-commercial** | | | | ✖ |
| YOLO-NAS weights | **No commercial use** | | | | ✖ |
| Ultralytics YOLO11 / YOLO26 | **AGPL-3.0** (code *and* trained weights) | 2.4–55.7M | 40.9–57.5 | Boxes, seg, pose, OBB | ✖ Blocks dual licensing |
| YOLOv6/v7/original v9 | GPL-3.0 | | | | ✖ |
| YOLOv10, YOLOv12 | AGPL-3.0 | | | | ✖ |

**Recommendation**, to be decided by the M0/M1 benchmarks:

- **Browser:** start with **RT-DETRv2-OBB-S**. An overhead camera has little perspective, so an oriented box plus an affine warp rectifies the card, and the angle directly encodes exhausted versus ready. Fall back to **D-FINE-S** plus the corner refiner if OBB export to ONNX or WebGPU proves troublesome.
- **Server / VOD pipeline and pseudo-labeling teacher:** **RF-DETR keypoint with 4 corners**, which handles perspective exactly, or RF-DETR-Seg-M with a quadrilateral fit.

## 3.3 Rectification

- **With an OBB detector:** affine warp from the rotated box. That is enough for near-overhead views.
- **Corner refiner (optional, and for tilted cameras):** a heatmap and soft-argmax network, the architecture the Maintainer already uses on phone photos. **Retrain it from a permissive initialisation.** timm and torchvision ImageNet-pretrained weights come with the caveat "assume the original dataset licence applies", and ImageNet's terms are non-commercial. The fix is cheap for a task this narrow: start from a DINOv2 or Perception Encoder backbone, or train from scratch on synthetic boards.

## 3.4 Embedders

| Model | Licence (weights) | Vision params | fp16 size | Verdict |
|---|---|---|---|---|
| MobileCLIP / MobileCLIP2 | **Apple ML Research Model terms** (research only) | 11–86M | | ✖ |
| **DINOv2 ViT-S/14** | Apache-2.0 | 21M | ~42 MB | ✔ **First candidate** |
| **Perception Encoder Core S16** (Meta) | Apache-2.0 | ~20M | ~40 MB | ✔ **Second candidate** |
| SigLIP 2 B/16 | Apache-2.0 | 86M | ~172 MB | ✔ Server only (too big for the browser) |
| DINOv3 | DINOv3 Licence: commercial use allowed, but derivatives must use the same terms, trade controls apply, and Meta may change the terms | 21M+ | | ⚠ Avoid in shipped models |
| OpenCLIP LAION | MIT, but the model card says deployed use is out of scope | | | ⚠ |
| C-RADIO (NVIDIA) | NVIDIA Open Model Licence | large | | ⚠ Custom terms |
| PP-ShiTuV2 feature model | Apache-2.0 | ~19 MB | | ✔ Product-retrieval baseline |

**Training objective.** With a closed gallery of 1–3k printings, **Sub-center ArcFace** (one class per printing) is the first choice. ArcFace-style embeddings also work open-set, which is how face recognition handles people it was never trained on, so new cards still match zero-shot from catalogue art. InfoNCE with a negatives queue, the Maintainer's existing recipe, is the baseline to beat. Both are in `pytorch-metric-learning` (MIT).

**Search.** A few thousand 256-d vectors is under a million multiply-adds per query: brute force in JS, no index needed.

## 3.5 Browser runtimes

| Runtime | Version (Sept 2026) | Licence | Notes |
|---|---|---|---|
| **ONNX Runtime Web** | 1.30.0 (2026-09-14) | MIT | **Primary.** WebGPU EP (the native WebGPU EP is now the recommended path; WebGL and JSEP are deprecated), WASM SIMD + threads. |
| Transformers.js | 4.3.0 (2026-09-16) | Apache-2.0 | Built on ORT Web. Supports RF-DETR, D-FINE, RT-DETR, DINOv2, SigLIP, Whisper, Moonshine. Good for prototypes. |
| LiteRT.js | `@litertjs/core` 2.5.3 | Apache-2.0 | `.tflite` models on WebGPU or WASM; RF-DETR exports to it. Keep as plan B. |
| TensorFlow.js | 4.22.0 (Oct 2024) | Apache-2.0 | Effectively superseded |
| WebNN | Origin trial | — | Not a production target |

Practical constraints for ORT Web:

- **int8 is not GPU-accelerated in WebGPU.** The op table lacks QLinearConv and ConvInteger. Use **fp16 or fp32 on WebGPU** (validate fp16 per model; DETR decoders can overflow) and **uint8 on WASM**.
- **WASM threads need a cross-origin-isolated context.** In an extension, that means an extension page (iframe or offscreen document) with COOP/COEP, not a service worker.
- **WebGPU availability:** Chrome/Edge desktop from 113, Android from 121, Linux on Intel Gen12+ from 144. Firefox on Windows from 141 and macOS from 145/147; Linux is still in Nightly. Everything else uses the WASM fallback.

## 3.6 Tracking, OCR, ASR and labeling

| Need | Choice | Licence | Notes |
|---|---|---|---|
| Tracking | Port **roboflow/trackers** (clean-room SORT, ByteTrack, OC-SORT) to TypeScript, with `kalman-filter` and `munkres` from npm | Apache-2.0 / MIT | No maintained npm tracker exists; a few hundred lines. ✖ BoxMOT (AGPL) |
| Overlay OCR (scores, names, graphics) | **PaddleOCR.js** (`@paddleocr/paddleocr-js`) with PP-OCRv6-tiny | Apache-2.0 | About once a second, overlay regions only; on-card text is too small |
| Caster ASR, browser | **Moonshine** streaming Tiny / Small | MIT (streaming and English models) | ✖ Its legacy non-English models are non-commercial |
| Caster ASR, server | Whisper (turbo, via whisper.cpp) | MIT | Parakeet is CC-BY-4.0 (unverified); attribution needed |
| Annotation | **CVAT** (self-hosted) | MIT | Built-in SAM is only in the paid cloud edition; the self-hosted edition uses serverless functions |
| Auto-labeling | **SAM 3** offline with a "trading card" text prompt, then quad fit and human review | SAM Licence | ⚠ Offline tool only, never shipped. SAM 2 is Apache-2.0 |
| Open-vocabulary boxes | Grounding DINO, OWLv2 | Apache-2.0 | Axis-aligned boxes only |

## 3.7 The shipped-component rule of thumb

| ✔ Safe (permissive) | ⚠ Caution (custom or attribution terms; keep out of shipped builds) | ✖ Avoid |
|---|---|---|
| RF-DETR N–L, all Seg sizes, Keypoint; RT-DETR v1–v4; RT-DETRv2-OBB; D-FINE; DEIM v1; LW-DETR; YOLOX; NanoDet; PicoDet; PP-YOLOE-R; RTMDet-R | DINOv3; SAM 3; C-RADIO; Parakeet (CC-BY); LAION OpenCLIP; **ImageNet-pretrained timm/torchvision weights** | MobileCLIP / MobileCLIP2 weights; DEIMv2; EdgeCrafter; YOLO-NAS weights; RF-DETR XL/2XL |
| DINOv2; Perception Encoder; SigLIP 2; PP-ShiTuV2 | RT-DETRv4 (DINOv3 teacher) | Ultralytics (all versions), YOLOv6/v7/v9 (GPL), YOLOv10/v12 (AGPL), BoxMOT |
| ORT Web; Transformers.js; LiteRT.js; trackers; supervision; Norfair | | Community models trained with Ultralytics (e.g. existing YOLO11 card detectors) |
| PaddleOCR(.js); RapidOCR; docTR; Tesseract.js; Whisper; Moonshine (streaming/EN); CVAT; Label Studio; SAM 2 | | |

## 3.8 Research tooling in `ml/` (not shipped)

These run on the Maintainer's and contributors' machines for the spike, training and evaluation. None of them ships in the extension or in release assets. Only the ONNX models exported from permissive weights and the float16 index do.

| Package or weights | Licence | Notes |
|---|---|---|
| numpy, Pillow, pytest, scipy | BSD-3-Clause, MIT-CMU, MIT, BSD-3-Clause | scipy runs the bootstrap mat detector (`matcrops`) |
| imageio-ffmpeg | BSD-2-Clause | Its bundled ffmpeg binary includes libx264 and is **GPL**. It is invoked as a separate process for the H.264 pass, on the developer's machine only, and never shipped or linked |
| torch, torchvision | BSD-3-Clause | CPU wheels from download.pytorch.org are enough for the spike |
| timm, huggingface_hub, safetensors | Apache-2.0 | timm's ImageNet weights keep the caveat in §3.7; the two backbones below are not ImageNet-trained |
| DINOv2 ViT-S/14 (`timm/vit_small_patch14_dinov2.lvd142m`) | Apache-2.0 (weights) | First embedder candidate |
| Perception Encoder Core S16 (`timm/vit_pe_core_small_patch16_384.fb`, run at 224 px) | Apache-2.0 (weights) | Second embedder candidate |

## 3.9 Evaluated: typed-decision models (Laya)

Suggested by the Maintainer as a "layer 1" decision on whether a card was added to the table. Checked on 2026-09-26:

| Model | What it is | Licence | Fit |
|---|---|---|---|
| Laya (github.com/NandhaKishorM/laya, Convai Innovations) | Non-autoregressive "System 1" decision engine: `choice`, `score` and `noul` (calibrated yes/no) in one forward pass. ModernBERT-large or mmBERT, 322–421M. PyTorch and ONNX, no browser runtime | Apache-2.0, code and weights | Text only. The README says base checkpoints are near chance on specialised tasks until fine-tuned |
| Jev (TypeSafe AI) | The hosted, closed-weights typed-decision API that Laya is the open counterpart of. Early access, priced per input token | Proprietary service | ✖ Closed weights and a network call per decision: cannot run in the extension or be retrained on our data |
| Laya Vision (github.com/r33drichards/laya-vision, independent fork, experimental) | The same API over an image plus text: SmolVLM-256M-Instruct cut to 20 layers, 201M. About 41 ms per question on an L4 GPU | Code Apache-2.0. **Weights CC BY-NC-SA 4.0** (training data includes ScienceQA and CrisisMMD) | ✖ The published weights cannot ship. Our own checkpoint can: SmolVLM-256M (Apache-2.0) as the base, the Apache-2.0 code, and our own training data |

Neither can label for us either ([D-018](../decisions.md#d-018-labels-come-from-reviewing-model-proposals)). Labels come from a person reviewing RiftEye's own guesses.

Verdict: not layer 1. The pixel change gate ([ARCHITECTURE §3.1.1](../ARCHITECTURE.md#311-change-gate-layer-1)) does that job at almost no cost and already works on real footage. A Laya-style model is a candidate **verifier of the gate's events** in M2, trained on labeled events, and it must beat the gate plus detector rules.

## Sources

- RF-DETR README, exports, keypoint docs, PML: github.com/roboflow/rf-detr, github.com/roboflow/rf-detr_plus ; PyPI `rfdetr`
- RT-DETRv2-OBB (RiO-DETR): github.com/RicePasteM/RiO-DETR ; RT-DETR: github.com/lyuwenyu/RT-DETR ; RT-DETRv4: github.com/RT-DETRs/RT-DETRv4
- D-FINE: github.com/Peterande/D-FINE ; DEIM: github.com/ShihuaHuang95/DEIM ; DEIMv2 licence change: github.com/Intellindust-AI-Lab/DEIMv2 (commit `bb64e5e`)
- LW-DETR, YOLOX, NanoDet, PaddleDetection (PicoDet, PP-YOLOE-R), MMRotate (RTMDet-R), YOLO-NAS licence (`LICENSE.YOLONAS.md`)
- Ultralytics YOLO26/YOLO11 docs and licence: github.com/ultralytics/ultralytics ; ultralytics.com/license
- MobileCLIP licences and the 2025-08-29 change: github.com/apple/ml-mobileclip (`LICENSE_MODELS`, commit `e15d36e`)
- DINOv2: github.com/facebookresearch/dinov2 ; DINOv3: github.com/facebookresearch/dinov3 ; Perception Encoder: github.com/facebookresearch/perception_models ; SigLIP 2: github.com/google-research/big_vision ; RADIO: github.com/NVlabs/RADIO ; timm licences: github.com/huggingface/pytorch-image-models#licenses
- pytorch-metric-learning: github.com/KevinMusgrave/pytorch-metric-learning ; ArcFace arXiv:1801.07698 ; SupCon arXiv:2004.11362
- Jev vs Laya comparison (Laya's authors' framing, third-party Jev numbers): jev-ai.pro/compare/jev-vs-laya ; laya.convaiinnovations.com
- Laya: github.com/NandhaKishorM/laya ; Laya Vision: github.com/r33drichards/laya-vision and huggingface.co/thaitea/laya-vision (model card licence field `cc-by-nc-sa-4.0`) ; SmolVLM: huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- ONNX Runtime Web: npmjs.com/package/onnxruntime-web ; v1.29.0 release notes ; WebGPU operator table (`js/web/docs/webgpu-operators.md`)
- Transformers.js v4: github.com/huggingface/transformers.js ; LiteRT.js: github.com/google-ai-edge/LiteRT (`litert/js`) ; WebGPU status: github.com/gpuweb/gpuweb/wiki/Implementation-Status
- In-browser RF-DETR measurements: github.com/Richard-S16/VISION-LAB
- roboflow/trackers, supervision, Norfair, BoxMOT (PyPI) ; PaddleOCR and `@paddleocr/paddleocr-js` ; Moonshine licence: github.com/moonshine-ai/moonshine ; whisper.cpp
- CVAT: github.com/cvat-ai/cvat ; SAM 2 / SAM 3: github.com/facebookresearch/sam2, github.com/facebookresearch/sam3
