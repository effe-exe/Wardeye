# Wardeye extension (alpha)

Wardeye on the Twitch player itself: watch any Riftbound replay or live stream on twitch.tv, and the cards on the table are marked at their corners and named on the video as it plays. Point at a card to see it ([ARCHITECTURE §5.1](../../docs/ARCHITECTURE.md#51-browser-extension-first-public-surface)).

The frames are read one of two ways, and the overlay is the same for both:

- **Standalone** (the private build): the engine runs inside the extension, on the GPU, with ONNX Runtime Web (`@rifteye/engine`). Nothing else is needed.
- **Companion** (the public build, and the fallback): the live runner on the same computer reads them (`python -m rifteye_ml.live --source browser`), and the extension only shows what it answers.

```
                                                  ┌──▶ standalone: engine document ──▶ engine worker (onnxruntime-web, @rifteye/engine)
Twitch player ──frame──▶ content script ──port──▶ worker ─┤                                     │
      ▲                                                   └──▶ companion: POST /frame ──▶ live runner (127.0.0.1)
      └────────────── marks, names, hover card ◀───────────────── board (state JSON) ◀─────────────┘
```

- **Nothing leaves the computer.** The engine reads the frames in the browser, the runner on 127.0.0.1, and only the table window is looked at: hand cams, player cams and the broadcast's hand lists are hidden information and never read ([D-005](../../docs/decisions.md)). Face-down cards are shown as face-down and never identified.
- **It follows the video.** Paused, nothing is sent and the board stays. A jump in the video starts a new board; another video (another page) finds its own table. The engine reads up to five frames a second (the runner about two), so the marks trail the picture by a fraction of a second.
- **Off and on.** Three ways turn Wardeye off in a tab: the power button on the badge, the toolbar button (`chrome.action`; it says OFF while it is off), and Alt+R (Option+R on a Mac). Off, the overlay is hidden and no frame is read or sent. The toolbar button or Alt+R turns it on again.
- **The plays panel.** The list button on the badge opens the browser's side panel (`chrome.sidePanel`) on `panel.html`: each player's legend and what is face up on their side of the table, and the plays as they happened, newest first, as the live runner's page shows them. A play clicked on a replay sends the video to just before it. The panel follows the active tab and connects to its content script while it is open (a port named `plays`); the content script keeps that video's plays in memory (`src/plays.ts`) and sends the board at most once a second, a new play at once. The engine's events reach it with each board (worker, engine document and engine worker pass them on); the live runner's page has its own list, so companion mode sends none. `panel.html?tab=<id>` follows that one tab.
- **Decklists.** The panel's Decklists tab has a box for each player (a list still counts only for the player whose legend it names). A list pasted in a box goes to the content script (the `plays` port), which sends the tab's lists with every frame, so an engine started afresh has them too. The engine reads them through the gallery's rows (`@rifteye/engine` decklist: deck code, text, tourney sheet or JSON), gives them to the tab's board (and to every board after a jump), and says what it made of each in the state (`lists`: the legends a list names, its cards, the lines it could not read). A side whose pinned legend a list names reads only that list's cards (with both lists' battlefields and the tokens); the other side keeps the legend rule. Lists go with the video, as the plays do.

## Use it

The public build reads with the live runner:

```bash
npm run build -w @rifteye/extension          # apps/extension/dist
cd ml && . .venv/bin/activate
python -m rifteye_ml.live --source browser --detector detector-v0.pth --encoder embedder:embedder-v1.pth
```

Then in Chrome: `chrome://extensions`, turn on Developer mode, **Load unpacked**, pick `apps/extension/dist`, and play a Riftbound video on twitch.tv. The badge on the player says what is going on (finding the table, waiting for the table camera, cards named). The runner takes the first free port from 8765; the extension looks for it on 8765–8774.

The private standalone build adds the models and the data. They are made from Riot's card art and never enter the repository ([D-006](../../docs/decisions.md)); they go only into the zip. It needs Chrome 137 or newer (WebGPU and WebAssembly JSPI).

```bash
# 1. the gallery, the catalogue and the hover pictures (once; about 35 minutes on two threads; nothing is fetched)
cd ml && . .venv/bin/activate
RIFTEYE_DATA=~/rifteye-data python -m rifteye_ml.web_assets OUT --models ~/rifteye-data/models/onnx
# 2. the zip, in parts under 30 MiB, streamed: the whole zip is never on disk
npm run build -w @rifteye/extension
npm run pack -w @rifteye/extension -- --models ~/rifteye-data/models/onnx --assets OUT --out parts/rifteye-standalone.zip
npm run pack -w @rifteye/extension -- --models ... --assets OUT --size-only     # what it would weigh; nothing is written
cat parts/rifteye-standalone.zip.part-* > rifteye-standalone.zip       # whoever gets the parts joins them
```

Unzip, then **Load unpacked** on the `rifteye-standalone` folder. The zip holds the detector in float32 and the embedder in float16 (`--precisions detector:fp32,embedder:fp16`, the default): the detector's float16 file fails on native WebGPU, the embedder's float16 file agrees with its float32 one (cosine 0.9998). A GPU without `shader-f16` needs the float32 embedder too (`embedder:fp32+fp16`). The `standalone.json` in the folder says how it runs:

| field | |
|---|---|
| `runtime` | `auto` (the default), `webgpu`, `wasm` or `companion`. `wasm` runs plain WASM even with no GPU (about 25 times slower than WebGPU); `companion` never uses the engine. |
| `layout` | a preset (`la-rq`, `plusrb`, `shenyang`) for every video, instead of finding the table from the footage. |
| `fps`, `threads` | the frames a second the boards are built for (default 5); the WASM threads (default up to 4). |

## The Chrome Web Store build

The store's version ([D-025](../../docs/decisions.md#d-025-release-on-the-chrome-web-store-now-and-apply-to-riot-in-parallel)) is standalone only, and one zip with the manifest at its root:

```bash
npm run build:store -w @rifteye/extension                      # dist-store/: no companion code, no access to 127.0.0.1
RIFTEYE_DATA=~/rifteye-data npm run pack:store -w @rifteye/extension -- --out wardeye-VERSION.zip
```

- **What the zip holds:** the models and the embedding gallery, keyed by printing id. It holds no catalogue and no hover pictures: `src/feed.ts` loads the card names, types and pictures from Riot's public card gallery as the viewer watches ([privacy policy](../../docs/PRIVACY.md)). `pack.mjs --store` refuses a zip that holds a card picture, name or text, or lacks the licences.
- **A printing the list does not name:**
  - another art of a listed one (`SET-NNNa`) takes that card's name and type;
  - a token by its code is named "Token";
  - the rest are named by their ids.
- **If the list cannot be read,** recognition still runs, with printing ids for names. The list is asked for again a minute later, at most five times.
- **Its browser tests:** `e2e/store.spec.ts`, against a local fake of the gallery (`e2e/fake-riot.ts`, which needs `openssl`).

## How the standalone mode works

- **The worker** (`src/worker.ts`) gets each frame over the content script's port. A Twitch tab that connects makes the engine document. A build with no `standalone.json` (the public build) makes nothing and asks nothing: it is companion mode, as before. When the browser has no WebGPU adapter, or the engine cannot start, the worker says why in the console, uses the live runner, and asks the engine again after five minutes.
- **The engine document** (`src/offscreen.ts`; a `chrome.offscreen` page, reason `WORKERS`) holds the engine and answers the worker's runtime messages, each with a tab's id: `hello` (can the engine run here), `frame` (a JPEG and the video's time; answered with the state) and `forget` (the tab is gone). One engine serves every Twitch tab: frames wait their turn, a tab's waiting frame is dropped when its next one comes, and each tab has its own board. The worker keeps nothing the engine needs, so it can be put to sleep and woken at any time (the overlay reconnects, the document goes on). The document closes itself after five minutes without a frame; the next frame makes it again.
- **The engine worker** (`src/engine-worker.ts`, started by the document from `engine-webgpu.js` or `engine-wasm.js`) decodes each frame the way that gives Pillow's bytes (`createImageBitmap` with no colour conversion, an `OffscreenCanvas`, `getImageData`), finds the layout (`autoLayout` over the first table frames: a look a second, over the last five; the presets take over when the footage will not give one and a preset's mat fills its table window), and runs the `Recognizer`. As in the live runner's loop, a jump in the video starts a new board, and another video (the page the overlay names) finds its own layout. Besides the runner's fields, the state carries `engine`: how it runs, the reads a second, and the frame's time in ms (decode, detect, embed, track, total). The badge's second line, shown when the badge is pointed at, says the reads a second, how it runs and where the frame's time goes; its first line says "on the processor" when the engine runs on WASM.
- **The way it runs** (`src/mode.ts`): WebGPU on the native JSPI build of onnxruntime-web, the detector in float32 and the embedder in float16 where the GPU has `shader-f16` (else float32, if the package holds it); plain WASM for a browser that has WebGPU but cannot run the native build, or whose GPU cannot run the embedder in the precision the package holds. A worker that dies starts again with the next frame; three frames in a row that fail start it afresh (a lost GPU device does not come back).
- **The package** holds `ort/` (the JSPI build and the plain WASM build; not JSEP, which breaks GridSample in float16), `models/`, and `data/`: `gallery/L<px>.bin` (float16) with its `index.json`, `catalog.json` and `thumbs/`. The gallery is the one the live runner computes with the ONNX embedder (`ml/rifteye_ml/web_assets.py`), for every level a table may need: 80 to 200 px in steps of 10, of which a layout uses the runner's three (`round(px * f / 10) * 10` for f in 0.8, 0.9, 1). Its index records the embedder it was made with: the float32 file's SHA-256 and the float16 copy's. The gallery is keyed on the level (px) alone; what ties it to the model is the hash, and the engine refuses an embedder whose file's SHA-256 is not the one the index gives for its precision. The hover pictures come from `data/thumbs/`, never from the network.
- **The manifest** has the `offscreen` and `sidePanel` permissions, a content security policy with `'wasm-unsafe-eval'` for extension pages, and the cross-origin isolation onnxruntime-web's threads need. Everything is bundled: an extension may not load remote code.

## Files

- `src/content.ts`: on twitch.tv, finds the player, grabs frames, draws the overlay and the hover card (card pictures are drawn on canvases, so a page that forbids outside images still shows them). Each card is keyed by track id, so it keeps its marks, name and hover across updates: its shape (an SVG polygon the pointer finds, outlined while pointed at), its corner marks (one path) and its name (an HTML chip). Names that would cover one another move under their card, or wait for the pointer. The view chosen in the panel's settings (`src/view.ts`, in `chrome.storage.local`) is a class on the overlay's root: outlines and names, outlines only, or clean.
- `src/overlay.css`, `src/mark.ts`: the overlay's rules, in Wardeye's look ([brand book §8](../../assets/brand/README.md#8-visual-language)), and the brand mark's shapes for the badge (a test holds them to `assets/brand/logo/mark.svg`). The build writes `dist/overlay.css` as the brand's tokens (`assets/brand/tokens.css`, on `.rifteye-root` alone, so nothing reaches Twitch's own styles), the three typefaces and then these rules. The rules use the tokens and no colour of their own; the dark theme is the only one. The typefaces are copied to `dist/fonts/` (with their licences) and are web-accessible to `https://www.twitch.tv/*` only. Motion is a 150 ms fade, with a slight rise, of the hover card and one 500 ms flash of the outline of a card that has just been named (never looping, and none in the clean view), both off under `prefers-reduced-motion`.
- `icons/`: the toolbar icons (16, 32, 48 and 128 px), rendered from the brand mark by `scripts/brand-icons.mjs`; the build copies them to `dist/icons/`, and `pack.mjs` puts them in the zip.
- `src/worker.ts`, `src/standalone.ts`, `src/companion.ts`: the worker, which way it reads (the engine, or the runner), and the runner's client.
- `src/offscreen.ts`, `src/offscreen.html`, `src/controller.ts`, `src/router.ts`, `src/engine-client.ts`: the engine document, its logic (what can run, starting and restarting the engine, the frame queue) and its link to the engine worker.
- `src/engine-worker.ts`, `src/engine-webgpu.ts`, `src/engine-wasm.ts`, `src/engine-host.ts`, `src/session.ts`: the engine worker for each build of onnxruntime-web, the host (a board for each tab, what it adds to the state) and one tab's video (layout, jumps, videos).
- `src/parts.ts`, `src/parts-engine.ts`: what the host needs of the engine, as interfaces, and the engine behind them (the only file that reaches into `@rifteye/engine`).
- `src/panel.html`, `src/panel.ts`, `src/panel.css`, `src/plays.ts`: the plays panel, in the brand's dark look (the build writes `dist/panel.css` as the tokens on `:root`, the typefaces from `dist/fonts/`, then the rules; nothing moves), and its model: the board per player, the plays of a video, the video clock.
- `src/geometry.ts`: the pure part of the overlay (where the picture sits in the player, what each track draws and says, the badge), unit-tested. The other pure modules: `assets.ts` (the package's files), `levels.ts`, `mode.ts`, `presets.ts`, `thumbs.ts`, `timer.ts`, `decode.ts`.
- `pack.mjs`: the private build as zip parts, or with `--store` the store's zip. `build.mjs`: the public build, or with `--store` the store's.
- The companion side: `ml/rifteye_ml/live/browser.py` and `POST /frame` in `ml/rifteye_ml/live/server.py`, which refuses frames from web pages (only the extension or this machine may post).

## Tests

```bash
npx vitest run apps/extension                    # the host's logic, the overlay, the build, pack.mjs
npm run build && npx playwright test apps/extension
```

The browser tests need no media and no models. The companion test answers frames with a fake runner. The standalone tests run the extension with stand-in models (tiny ONNX graphs made at test time) and a stand-in recogniser, on a page that records its own video: on WebGPU (a software adapter), on plain WASM, with no WebGPU adapter, and in the public build. `e2e/replay.spec.ts` runs the whole engine on the 240 frames of the LA final, in Chromium, and compares every step's state and events with the Python reference; it skips unless `RIFTEYE_M3` names the private folder, and takes many minutes.
