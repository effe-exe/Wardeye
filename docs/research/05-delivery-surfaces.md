# 05: Delivery surfaces

Where RiftEye runs, and in which order. State as of September 2026.

## 5.1 Summary

| Surface | Needs the streamer? | Reach | Input quality | Sync | Infra cost | Effort | Monetisation (subject to Riot, [08](08-legal-and-policy.md)) |
|---|---|---|---|---|---|---|---|
| **Browser extension, local inference** | **No** | Desktop Chrome/Edge users who install it; Twitch and YouTube, live and VOD | What the viewer is watching (720p–1080p, compressed) | **Exact**: it analyses the frame on screen | ≈ 0 (viewer's GPU) + a CDN for model files | Medium (6–10 weeks to an MVP) | Free tier plus transformative extras |
| **Web app with a VOD library** | No, but needs rights to process the recordings | Anyone, including mobile; searchable | **Best for VODs**: source recordings, multi-pass, human review | Through the embedded player's time API; results sit **beside** the player, never on it | ≈ $0.04–0.26 per VOD-hour | Medium–large | Hosted features |
| **Broadcaster sidecar (OBS)** | Yes | Every viewer of adopting streams, **including mobile and VOD**, because graphics are burned into the broadcast | **Best overall**: the clean camera before encoding | Real-time, inherently in sync | Streamer's machine | Medium (sidecar); large for a native plugin | Tools for organisers |
| **Twitch video-overlay extension** | Yes, and it needs a feed from the broadcaster kit | Desktop viewers of adopting channels; **live only, not VODs or mobile** | As good as its feed | Delay messages by `hlsLatencyBroadcaster` | Small backend | Medium + Twitch review | Bits (80% creator / 20% developer) |
| Desktop viewer app | No | Low: install friction and OS permissions | Scaled screen pixels | Exact, but overlay alignment is fiddly | ≈ 0 | Large | — |

**Order:** extension first, VOD pipeline alongside it (it is the data engine), then the web VOD library, then the broadcaster kit and Twitch Extension with a partner. **A desktop viewer app is not planned.**

This matches [ARCHITECTURE §5](../ARCHITECTURE.md#5-runtime-topologies) and the [roadmap](../ROADMAP.md).

## 5.2 Browser extension (Manifest V3)

**Frame access works.**

- Twitch and YouTube play video through Media Source Extensions (`blob:` URLs). Under the HTML standard, such media is *CORS-same-origin*, so drawing it to a canvas does not taint it.
- Existing extensions confirm it: a YouTube screenshot extension draws the `<video>` and calls `toBlob()`, and a Twitch userscript does `createImageBitmap(video)`.
- DRM (EME) video cannot be read. It is limited to licensed premium streams (e.g. certain sports on Twitch), not ordinary channels.
- APIs: `requestVideoFrameCallback` (Chrome 83+, Firefox 132+) to sample presented frames at 2–5 Hz; `createImageBitmap(video)` or `new VideoFrame(video)` (Chrome 94+) to grab them.

**Alternatives are worse.**

| API | Problem |
|---|---|
| `tabs.captureVisibleTab` | 2 calls per second, and captures the whole viewport as JPEG |
| `tabCapture` | Needs a user gesture per session, and gives the composited tab (chat, our own overlay, scaled player) |
| `getDisplayMedia` | Needs a picker dialog every time |

**Where inference runs: to be prototyped in M2.**

- Extension messaging (`runtime.sendMessage`) only carries JSON, so frames cannot move zero-copy to the service worker or an offscreen document.
- Content scripts are subject to the host page's CSP, which may block WASM or workers.
- The planned design is a **hidden, web-accessible extension iframe** inside the page, acting as an *inference host*:
  - The content script transfers `ImageBitmap`s to it with `postMessage`; `ImageBitmap` is transferable.
  - The iframe runs under the extension's own CSP (`'wasm-unsafe-eval'`), can be cross-origin isolated, and spawns a dedicated worker running ONNX Runtime Web (WebGPU, WASM fallback).
- Fallbacks to measure against: inference directly in the content script, or `tabCapture` into an offscreen document.

**UI.**

- **Overlay:** hitboxes and a hover card, in a Shadow DOM root inside the player container.
  - A fullscreen element renders in the browser's top layer, so on `fullscreenchange` the overlay moves into `document.fullscreenElement`. A `popover` is the alternative.
  - A `ResizeObserver` recomputes the letterboxed video rectangle from `videoWidth` and `videoHeight`, which also covers theatre mode.
  - A `MutationObserver` re-attaches after in-page navigation.
  - Selectors for the player DOM live in one small, well-tested adapter per site, because they will break.
- **Timeline:** `chrome.sidePanel` (Chrome 114+). `open()` needs a user gesture. Firefox uses `sidebarAction` instead.

**Store policies.**

- **Single purpose:** "narrow and easy to understand". RiftEye's is *identify the cards in Riftbound videos and log what was played*. Collected data must be strictly necessary to that purpose (2026 policy update).
- **No remotely hosted code:** all JS and WASM, including ONNX Runtime's WASM files, ship in the package (`wasmPaths` pointed at bundled files; CDN-loaded ORT WASM has caused rejections).
  - Model weights and the catalogue are **data**, and fetching data is allowed. Hugging Face's own MV3 sample downloads weights at runtime.
  - Plan: ship a baseline model in the package, fetch versioned updates, verify hashes ([ARCHITECTURE §8](../ARCHITECTURE.md#8-model-and-index-versioning)).
- **Package limit:** 2 GB. Irrelevant at RiftEye's sizes (tens of MB).
- **No downloading of YouTube video.** RiftEye never saves or uploads frames; the opt-in correction payload is a single card crop ([04 §4.10](04-data-and-evaluation.md#410-opt-in-corrections-from-the-extension)).

**Firefox.**

- No `tabCapture` and no offscreen documents. The direct `<video>` approach needs neither.
- WebGPU is on Windows and macOS; Linux uses the WASM fallback.
- Firefox is a port after the Chrome MVP.

## 5.3 Web app with a VOD library

- **Twitch embed:** requires the `parent` hostname, and Twitch's embed requirements say you **cannot overlay anything on top of the player**. `getCurrentTime()` and `seek()` work on VODs.
- **YouTube IFrame API:**
  - `getCurrentTime()`, `seekTo()` and `onStateChange` are available.
  - The developer policies forbid modifying or building upon the player.
  - They also forbid downloading or storing YouTube audiovisual content without YouTube's written approval.
- **So:** a **time-synced card rail and timeline beside the player**, and clicking an event seeks the video. Hover-on-video stays an extension feature. The extension can fetch the server's timeline for a known VOD and draw it on the video, because it runs as the viewer's own user agent.
- **Processing only licensed recordings.**
  - Get files directly from organisers or Riot rather than downloading them from platforms ([08 §8.3](08-legal-and-policy.md#83-broadcast-footage-and-platform-terms)).
  - Downloader tools exist and are permissively licensed (yt-dlp: Unlicense; Streamlink: BSD-2; TwitchDownloader: MIT), but platform terms, not tool licences, are the constraint.
- **Frame sampling** with ffmpeg:
  - `fps=2` to `fps=5`, plus `select='gt(scene,0.3)'` for scene changes;
  - `mpdecimate` to drop near-duplicates.
- **Cost (estimate)** for small detector + embedder on 7,200–18,000 frames per hour of 1080p:

  | Where | Price | Per VOD-hour (5–15 GPU-minutes) |
  |---|---|---|
  | Cloud Run L4 | $0.67/h GPU + CPU/RAM ≈ **$1.05/h** | **≈ $0.09–0.26** |
  | GCE Spot g2-standard-4 (1× L4) | ≈ **$0.42/h** | **≈ $0.04–0.11** |

- **An optional, privacy-preserving flywheel.** With consent, extensions can contribute their *local detections* (card IDs and timestamps only, no pixels) for a VOD. The server merges them into a shared timeline, and no video is ever downloaded.

## 5.4 Broadcaster kit

- **Use a sidecar, not a native OBS plugin.**
  - OBS and its plugin template are GPL-2.0, so a native plugin would have to be GPL-compatible, which conflicts with dual licensing ([07](07-licensing-and-governance.md)).
  - A separate process talking to **obs-websocket** (built into OBS 28+) is independent software. `GetSourceScreenshot` returns any named source at full resolution, *before* overlays and encoding. That frame rate needs measuring; each call encodes an image.
- **Other clean feeds:** NDI (DistroAV), or a capture card shared with the sidecar.
- **Output:**
  - An OBS **Browser Source** overlay (card pop-ups, board graphics). It is burned into the stream, so it reaches mobile and VOD viewers too.
  - Optionally a Twitch Extension feed.
- **Precedent:** obs-backgroundremoval ships ONNX Runtime models inside OBS (GPL-3.0). It proves the runtime side, and its licence is a reminder of why RiftEye stays out of process.
- **Competition:** Riftbound Vision already offers a beta OBS plugin and Twitch extension. Pursue this surface with an organiser partner ([06](06-prior-art-and-starting-point.md)).

## 5.5 Twitch Extensions

- **Types:** panel, video overlay (full player) and video component. A channel runs at most one overlay.
- **No pixel access.** Extensions run in a sandboxed iframe, and the helper's `onContext` exposes only metadata. Detection must happen at the broadcaster or on a server.
- **Data path:** Extension Backend Service → **Extension PubSub**. Limits are 100 messages per minute per channel and 5 KB per message. The legacy PubSub shutdown (2025-04-14) did not affect Extension PubSub.
- **Sync:** hold each message for `hlsLatencyBroadcaster` seconds before showing it. That is the Hearthstone Deck Tracker extension's `AsyncQueue` pattern.
- **Limits:** overlays show **only while live** and **only on desktop**. Review takes about 3 business days, sometimes more.
- **Monetisation:** Bits split 80/20 between creator and developer; features can be gated on subscription status.

## 5.6 Desktop viewer app: not planned

Electron (`desktopCapturer`, `onnxruntime-node` with DirectML, CUDA or CoreML) or Tauri (the `scap` crate, the `ort` crate) would give the strongest local GPU access. The costs: install friction, code signing, per-OS capture and overlay code, and scaled screen pixels. The extension covers the same viewers with far less friction. Revisit only if organisers need a companion UI.

## Sources

- Local-mode media and canvas tainting: WHATWG HTML (media, canvas) ; MSE spec (w3c/media-source)
- Existing frame-reading extensions: github.com/FutureMillennium/Screenshot-YouTube ; github.com/eramdam/userscripts (Twitch)
- `requestVideoFrameCallback` / `VideoFrame` compatibility: github.com/mdn/browser-compat-data
- `captureVisibleTab` limit: chromium `chrome/common/extensions/api/tabs.json` ; `tab_capture.idl` ; offscreen `offscreen.webidl` ; side panel `side_panel.idl`
- Screen-capture and offscreen guides: github.com/GoogleChrome/developer.chrome.com (`docs/extensions/mv3/screen_capture`, `reference/offscreen`)
- Fullscreen top layer: whatwg/fullscreen ; popover: WHATWG HTML
- Chrome Web Store program policies, MV3 requirements (remote code): developer.chrome.com/docs/webstore/program-policies ; transformers.js issue #839 (bundled WASM) ; transformers.js-examples `browser-extension`
- WebGPU availability: github.com/gpuweb/gpuweb/wiki/Implementation-Status
- Twitch Extensions: dev.twitch.tv/docs/extensions ; `twitch-ext` typings (DefinitelyTyped) ; PubSub limits (twitchdev/issues #612) ; HDT extension `AsyncQueue.ts`: github.com/HearthSim/twitch-hdt-frontend ; Bits revenue share: help.twitch.tv
- obs-websocket protocol (`GetSourceScreenshot`): github.com/obsproject/obs-websocket ; OBS licence: github.com/obsproject/obs-studio ; obs-backgroundremoval: github.com/royshil/obs-backgroundremoval ; DistroAV
- Twitch embed requirements: dev.twitch.tv/docs/embed ; YouTube API Services Developer Policies (Open Terms Archive copy)
- yt-dlp (PyPI), Streamlink (PyPI), TwitchDownloader (GitHub) ; ffmpeg `doc/filters.texi`
- Cloud Run pricing: cloud.google.com/run/pricing ; Spot VM pricing: cloud.google.com/spot-vms/pricing ; RF-DETR latency table: github.com/roboflow/rf-detr
