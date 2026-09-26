# RiftEye reviewer

A small static web page for turning a model's guesses into labels. The model proposes an answer for every item in a **review pack**. You only say whether it is right:

- <kbd>Y</kbd>: correct.
- <kbd>1</kbd>–<kbd>9</kbd>: wrong, it is the numbered alternative.
- <kbd>N</kbd>: wrong. Type what it is and press <kbd>Enter</kbd>, or press <kbd>Enter</kbd> on the empty box if you don't know.
- <kbd>S</kbd>: can't tell (blurry, covered, not a card). Saying <kbd>S</kbd> is better than guessing: a wrong label does more harm than a missing one.
- <kbd>Z</kbd>: undo the last answer. <kbd>←</kbd>/<kbd>→</kbd>: previous or next item.

That takes about two seconds per item, so 400 items take 15–20 minutes. Labelling the same crops by name from scratch would take hours. Why this is the labelling method, and how packs are chosen: [docs/research/04 §4.4](../../docs/research/04-data-and-evaluation.md#44-stream-footage-the-data-engine) and [D-018](../../docs/decisions.md#d-018-labels-come-from-reviewing-model-proposals).

It works entirely in your browser. The pack and your answers never leave the page until you press **Export**.

## Build and open

```bash
npm install          # at the repository root
npm run build -w @rifteye/reviewer
open apps/reviewer/dist/index.html    # or double-click it; no server needed
```

## Use

1. **Open pack:** a `*.json` review pack (a `ReviewPack` from [`@rifteye/schema`](../../packages/schema)). Packs made by `python -m rifteye_ml.reviewpack` carry their pictures inside, so a pack is one file.
2. Answer with the keys above, or click the buttons and the numbered alternatives.
3. **Export** writes `<pack id>.answers.json` (`ReviewAnswers`). Send that file back. `python -m rifteye_ml.reviewpack apply` turns it into labels.

Your answers are also autosaved in this browser, per pack. Reopening the same pack offers to continue. **Import answers** continues from an exported file, for example on another computer. The optional name field is stored in the export so reviews can be told apart.

Two kinds of pack exist:

- **identity** ("Is this the card?"): the crop the model saw, its surroundings with the card outlined, the model's guess with its confidence, and the next three guesses. One item stands for a *track*, the same physical card over several frames, so one answer labels all of them.
- **event** ("What happened in the purple box?"): the table before and after a change the change gate reported, and its guess of the kind (a card put down, taken away, changed, or nothing).

Packs, answers and everything in them are private: they hold broadcast crops and card art ([D-006](../../docs/decisions.md#d-006-no-third-party-media-in-git), [D-015](../../docs/decisions.md#d-015-no-riot-api-no-riot-assets-distributed)). Never commit them.

## Tests

```bash
npx vitest run apps/reviewer                                   # logic: answering, undo, restore, search
npm run build -w @rifteye/reviewer && npx playwright test      # end to end in Chromium
```

The end-to-end test builds its pack from SVG rectangles, so no images are needed in git.
