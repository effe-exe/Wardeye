# RiftEye extension (alpha, companion mode)

RiftEye on the Twitch player itself: watch any Riftbound replay or live stream on twitch.tv, and the cards on the table are boxed and named on the video as it plays. Point at a card to see it ([ARCHITECTURE §5.1](../../docs/ARCHITECTURE.md#51-browser-extension-first-public-surface)).

In this first version the recognition runs in the live runner on the same computer, not in the browser: the extension grabs the player's current frame a few times a second and hands it to the runner, which answers with the board. Moving the models into the browser (ONNX Runtime Web, M2) removes the runner later; the overlay stays the same.

```
Twitch player ──frame──▶ content script ──port──▶ extension worker ──POST /frame──▶ live runner (127.0.0.1)
      ▲                                                                                     │
      └──────────────── boxes, names, hover card ◀──────────── board (state JSON) ◀────────┘
```

- **Nothing leaves the computer.** Frames go to 127.0.0.1 only, and the runner looks at the table window only; hand cams, player cams and the broadcast's hand lists are hidden information and never read ([D-005](../../docs/decisions.md)). Face-down cards are shown as face-down and never identified.
- **It follows the video.** Paused, nothing is sent and the board stays. A jump in the video starts a new board; another video (another page) makes the runner find its table again (`--layout auto`). About two frames a second on a laptop, so boxes trail the picture by about half a second.
- **Keys.** Alt+R hides or shows the overlay.

## Use it

```bash
npm run build -w @rifteye/extension          # apps/extension/dist
cd ml && . .venv/bin/activate
python -m rifteye_ml.live --source browser --detector detector-v0.pth --encoder embedder:embedder-v1.pth
```

Then in Chrome: `chrome://extensions`, turn on Developer mode, **Load unpacked**, pick `apps/extension/dist`, and play a Riftbound video on twitch.tv. The badge on the player says what the runner is doing (finding the table, waiting for the table camera, cards named). The runner takes the first free port from 8765; the extension looks for it on 8765–8774.

## Files

- `src/content.ts`: on twitch.tv, finds the player, grabs frames, draws the overlay (boxes keyed by track id, so a card keeps its box and hover across updates) and the hover card (card pictures are drawn on canvases, so a page that forbids outside images still shows them).
- `src/worker.ts`: the only part that talks to the runner (a page may not reach 127.0.0.1 itself).
- `src/geometry.ts`: the pure part (where the picture sits in the player, what each track draws and says), unit-tested.
- The runner side: `ml/rifteye_ml/live/browser.py` (the frame source) and `POST /frame` in `ml/rifteye_ml/live/server.py`, which refuses frames from web pages (only the extension or this machine may post).

## Tests

```bash
npx vitest run apps/extension                 # geometry, labels, hover cards, badge
npm run build && npx playwright test apps/extension   # the extension loaded in Chromium, on a stand-in Twitch page
```

The browser test records its own video in the page and answers frames with a fake runner, so it needs no media and no models.
