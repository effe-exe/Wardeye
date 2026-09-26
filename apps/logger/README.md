# RiftEye timeline logger

A small static web page for recording **ground-truth timelines**: what happened in a match, and when, typed in by a person watching the recording. These timelines are the product test set ([docs/research/04 §4.5](../../docs/research/04-data-and-evaluation.md#45-ground-truth-timelines)). The engine's output is scored against them.

It works entirely in your browser. The video, the card list and the log never leave the page until you press **Export**.

## Build and open

```bash
npm install          # at the repository root
npm run build -w @rifteye/logger
open apps/logger/dist/index.html    # or double-click it; no server needed
```

## Use

1. **Open video:** a match recording you have permission to use, for example a file from the organiser.
2. **Load cards:** the `catalog.jsonl` from `ml/`, generated on your machine, for name autocomplete. See [ml/README.md](../../ml/README.md).
3. Fill in the title, the players and the legends (Player A sits at the bottom of the table camera).
4. Play the video and press a key the moment something happens. The video pauses and a small form opens with the timestamp already captured.
   - **Card events** (<kbd>P</kbd> played, <kbd>C</kbd> spell cast, <kbd>M</kbd> moved, <kbd>X</kbd> left play, <kbd>R</kbd> revealed): type part of the name, <kbd>Enter</kbd> picks the suggestion, <kbd>Enter</kbd> again saves.
   - **Other events** (<kbd>T</kbd> turn start, <kbd>U</kbd> runes channeled, <kbd>S</kbd> score, <kbd>G</kbd>/<kbd>E</kbd> game start/end, <kbd>H</kbd> card hidden): <kbd>Enter</kbd> saves straight away.
   - <kbd>A</kbd>/<kbd>B</kbd> switch the active player; <kbd>Z</kbd> undoes the last event; <kbd>?</kbd> lists every shortcut.
5. **Export** writes `<video>.timeline.json`, a `TimelineDocument` from [`@rifteye/schema`](../../packages/schema). **Import** reopens one to continue or fix it.

Your work is also autosaved in this browser, per video file name. Reopening the same file offers to continue. The exported file is the real record.

**Hidden cards** (<kbd>H</kbd>) are logged without a name, always. RiftEye never records hidden information ([D-005](../../docs/decisions.md#d-005-public-information-only)).

## Tests

```bash
npx vitest run apps/logger                                   # logic: state, export, card search
npm run build -w @rifteye/logger && npx playwright test      # end to end in Chromium
```

The end-to-end test records its own short WebM inside the browser, so no media files are needed in git.
