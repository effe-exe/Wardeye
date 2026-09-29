# Wardeye viewer (preview)

A static page that plays a recorded match with Wardeye's output on top, the way the extension will show it on a live stream ([ARCHITECTURE §5.1](../../docs/ARCHITECTURE.md#51-browser-extension-first-public-surface)):

- **Point at a card** on the table to see it. When Wardeye is sure, the hover card shows the card and how sure it is. When it is not, it shows the best three guesses side by side ([§3.5](../../docs/ARCHITECTURE.md#35-matcher-priors-and-fusion)). A face-down card says so, and is never identified.
- **The timeline** lists changes on the table with the card involved. Click one to jump to a moment before it.
- **On the table now** lists what Wardeye sees at this moment.

It is a preview, not the product: everything is computed ahead of time by the M0 pipeline in [`ml/rifteye_ml/demo.py`](../../ml/rifteye_ml/demo.py). That pipeline runs the bootstrap mat detector a few times a second, the colour-grid identifier with a gallery pyramid, tracks of one physical card, and the change gate. The bootstrap detector only sees cards lying on their own, so stacked and overlapping cards have no box yet; the trained detector comes in M1.

## Build a demo

```bash
npm run build -w @rifteye/viewer          # apps/viewer/dist: index.html, app.js, style.css
cd ml && . .venv/bin/activate
python -m rifteye_ml.changegate --video seg.mp4 --start 90 --duration 120 --table 0.15,0.10,0.88,0.884 --out gate.json
python -m rifteye_ml.demo --video seg.mp4 --start 90 --duration 120 --table 0.17,0.09,0.86,0.884 --long 131 \
  --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art --embed-cache ~/rifteye-data/embed-cache \
  --gate gate.json --title "Swiss round 11, game 1" --out ~/rifteye-data/demo/r11g1
ffmpeg -ss 90 -t 120 -i seg.mp4 -vf scale=1280:720,fps=30 -c:v libx264 -crf 25 -c:a aac -b:a 64k -ac 1 \
  -movflags +faststart ~/rifteye-data/demo/r11g1/clip.mp4
cp apps/viewer/dist/{index.html,app.js,style.css} ~/rifteye-data/demo/r11g1/
```

Then open `index.html` in that folder. The bundle (`data.js`, `art/`, the clip) holds broadcast footage and card art, so it is private: never commit or publish it ([D-006](../../docs/decisions.md#d-006-no-third-party-media-in-git), [D-015](../../docs/decisions.md#d-015-no-riot-api-no-riot-assets-distributed)). Some Chromium builds, such as Playwright's, cannot play H.264; Chrome, Edge, Firefox and Safari can.

## Tests

```bash
npx vitest run apps/viewer                                  # box timing, geometry, hover states
npm run build -w @rifteye/viewer && npx playwright test     # end to end with a generated bundle and WebM
```
