# rifteye-ml

Research, training and evaluation tools for RiftEye (Python 3.11+). Right now this is the **M0 feasibility spike**. It measures how well cards can be identified from art as they shrink and get compressed on stream ([docs/research/04 §4.8](../docs/research/04-data-and-evaluation.md#48-the-m0-feasibility-spike)).

| Module | What it does |
|---|---|
| `catalog` | Saves Riot's public card gallery feed, normalises it into `catalog.jsonl` and caches the card images **locally** |
| `degrade` | The stream simulator: cards on a table at a target on-screen height, sleeves, glare, sensor noise, then a **real libx264 encode and decode** at stream bitrates. The `camera` realism level adds tilt, lighting, defocus, occluders and detector box error; `foil` adds a holographic sheen to 20% of cards on top of that |
| `encoders` | `colorgrid` and `dhash` baselines, any pretrained `timm:` backbone (e.g. DINOv2), and fused encoders: `colorgrid/trim0.03+dhash/trim0.03` scores colour and structure together |
| `retrieval` | Brute-force gallery search with 4-rotation matching; printing-level and card-level top-k |
| `spike` | Accuracy vs card height × bitrate (synthetic), or vs crop height (real labeled crops) |
| `report` | Markdown tables and an SVG accuracy-vs-height chart from spike CSVs (numbers only) |
| `changegate` | Layer 1: when and where the table changed (settled changes vs a still-table model; hands and light ignored), from a VOD window |
| `matcrops` | A classical bootstrap detector for M0: isolated cards on a known playmat, straightened into upright crops (not the product detector). `--mask notmat` works on any mat colour |
| `label` | Model-assisted labeling of real crops: ranked candidates and verification sheets; a person decides |
| `demo` | A preview bundle for `apps/viewer`: the M0 pipeline run offline on a VOD window (detector, identifier, tracks, change gate), so a recorded match can be hovered |
| `reviewpack` | Review packs for `apps/reviewer`: the model's guesses, one per track of the same card, least confident first plus a random audit; and exported answers back into labels |
| `adapter` | The M0 "quick fine-tune": a linear head on a frozen backbone, trained on synthetic camera-level crops, evaluated on held-out sets |
| `synth` | M1's synthetic board generator: whole 1v1 boards with stacks, face-down piles, dice and hands, filmed by a camera model into a broadcast layout, through the real H.264 pass, with every card's full quad and visible fraction |
| `index` | Writes the shipped index (float16 matrix + manifest) and refuses to load it with a different encoder |
| `fixtures` | Procedural fake cards for tests and demos. No Riot content |

> **Card images and text are Riot Games IP.** The tools cache them on your machine for research only. Never commit them, upload them or put them in a release ([D-006](../docs/decisions.md#d-006-no-third-party-media-in-git), [D-015](../docs/decisions.md#d-015-no-riot-api-no-riot-assets-distributed)).

## Setup

```bash
cd ml
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'           # numpy, pillow, a bundled ffmpeg with libx264, pytest
pip install -e '.[torch]'         # optional: torch + timm, for pretrained encoders
pytest -q                         # ~15 s, fake cards only
```

`imageio-ffmpeg` provides an ffmpeg binary for the H.264 pass. It is a development tool and is never shipped.

## Run the spike on real cards

**1. Save the card gallery feed.** This is the public JSON behind the card gallery on playriftbound.com: about 1,200 printings in 6 pages. No API key, and not the Riot API.

```bash
mkdir -p ~/rifteye-data/catalog
python -m rifteye_ml.catalog fetch-feed --out ~/rifteye-data/catalog/feed
```

**2. Build the catalogue and cache the images locally.** The images come from Riot's public CDN: about 1,200 PNGs, 1.1 GB.

```bash
python -m rifteye_ml.catalog build --feed ~/rifteye-data/catalog/feed --out ~/rifteye-data/catalog/catalog.jsonl
python -m rifteye_ml.catalog download --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art
```

Other sources can fill gaps: `--gallery` takes a mirror's flattened `cards.json` (e.g. `github.com/slimtreble/Riftbound-card-data`), `--jsonl` a TCGplayer/TCGCSV-style file (fields `name`, `number`, `image`) with promos.

**Printings the gallery lacks.** Some promos, alt-art runes and tokens are in no feed (the Los Angeles grand final: [M0 §5.7](../docs/reports/m0-spike.md#57-the-grand-final-los-angeles)). `catalog supplement` adds them from a JSON list, each with a picture from elsewhere, such as its clearest reviewed crop on a broadcast. The picture goes into the art cache as `supplement://<printing_id>.png`, so every tool loads it like gallery art, and `--merged` writes the catalogue with the supplement appended. Pictures and supplement stay private like the art:

```bash
python -m rifteye_ml.catalog supplement --spec ~/rifteye-data/catalog/supplement-spec.json \
  --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --out ~/rifteye-data/catalog/supplement.jsonl --merged ~/rifteye-data/catalog/catalog-plus.jsonl
# supplement-spec.json: [{"printing_id": "VEN-R04a", "like": "VEN-R04", "variant": "alt_art",
#                         "image": ".../crops/t15h04m20s_01.png", "rotate": 180, "source": "..."}, ...]
```

**Localised printings.** The feed also serves other locales. `zh_CN` has the Simplified Chinese printings of Origins (OGN) and the Origins starter decks (OGS), with the same art and frame and Chinese text; other sets come back in English for now. Localised rows take their `card_id` from the English catalogue:

```bash
python -m rifteye_ml.catalog fetch-feed --out ~/rifteye-data/catalog/feed-zh_CN --locale zh_CN
python -m rifteye_ml.catalog build --feed ~/rifteye-data/catalog/feed-zh_CN --language zh-Hans \
  --card-ids ~/rifteye-data/catalog/catalog.jsonl --out ~/rifteye-data/catalog/catalog-zh-Hans.jsonl
python -m rifteye_ml.catalog download --catalog ~/rifteye-data/catalog/catalog-zh-Hans.jsonl --cache ~/rifteye-data/art
```

**3. Synthetic curve.** Cards at 40–160 px and 2–6 Mbps, identified against the whole gallery. Two realism levels:

- `--realism codec`: perfect crops, downscaling and the codec only. This is an upper bound.
- `--realism camera`: adds camera tilt, white balance, exposure and stage light, defocus, occluders (fingers, counters, overlapping cards) and detector box error. The magnitudes are assumptions until real footage calibrates them (`REALISM` in `degrade.py`).

```bash
python -m rifteye_ml.spike synthetic --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --encoder colorgrid --encoder dhash \
  --encoder timm:vit_small_patch14_dinov2.lvd142m --encoder timm:vit_pe_core_small_patch16_384.fb@224 \
  --realism camera --queries 300 --heights 40,60,80,120,160 --bitrates 4000 \
  --gallery-scales 40,60,80,120,160 --embed-cache ~/rifteye-data/embed-cache --out reports/m0-camera.csv
```

- `--gallery-scales 40,60,80,120,160` embeds the gallery at those on-screen sizes (a pyramid), and each crop is searched against the level nearest its size. Pretrained ViTs need it: with a single sharp gallery, DINOv2-S found 13% of clean 40 px cards; with the pyramid it found 62%. The detector knows each card's size, so the extension can do the same.
- `--embed-cache DIR` keeps gallery embeddings between runs (private `.npy` files).
- `--queries 300` degrades a seeded sample of 300 printings; the gallery stays complete. Pretrained ViTs embed about 10–40 images per second on a laptop CPU, and `rotation=search` embeds every query 4 times.
- `--query-catalog catalog-zh-Hans.jsonl` degrades another language's printings of the same cards instead. For example, Chinese crops searched against the English gallery.
- `--strips top:0.25,top:0.4,left:0.3` also scores every card from only the band a stack leaves visible, against the same band of every catalogue card (CSV column `view`).
- The simulation takes about a minute per setting for the whole catalogue at 1080p. The H.264 pass runs single-threaded so results are bit-exact between runs.

**Quick fine-tune.** A linear head on the frozen backbone, trained on synthetic crops of some sets and evaluated on the others. It asks whether a trained embedder will close the camera gap for cards it never saw:

```bash
python -m rifteye_ml.adapter --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --embed-cache ~/rifteye-data/embed-cache --encoder timm:vit_small_patch14_dinov2.lvd142m \
  --train-sets OGN,OGS,SFD --out reports/m0-adapter.csv
```

Add `--real-crops <folder> --real-labels <labels.csv>` to score the same head on real stream crops too: orientation unknown, per crop, one crop per track, and split by whether the card's set was in training. That is the question that matters for M1: does training on synthetic crops alone transfer to footage?

Encoder specs take a pooling suffix: `timm:vit_small_patch14_dinov2.lvd142m@224/avg` averages the patch tokens instead of using the class token.

Turn a CSV into tables and a chart:

```bash
python -m rifteye_ml.report --csv reports/m0-camera.csv --svg reports/m0-camera.svg --bitrate 6000 --realism camera
```

**4. Real curve** (the one that decides). Put crops from real, permitted footage in a folder. `label propose` ranks catalogue candidates for each crop and renders private verification sheets; a person confirms or corrects each one by eye:

```bash
python -m rifteye_ml.label propose --catalog ~/rifteye-data/catalog/catalog.jsonl \
  --catalog ~/rifteye-data/catalog/catalog-zh-Hans.jsonl --cache ~/rifteye-data/art \
  --crops ~/rifteye-data/real-crops --out ~/rifteye-data/real-crops/proposals.csv --sheets ~/rifteye-data/real-crops/sheets
```

Then write `labels.csv`:

```csv
file,printing_id
vod1_00123_card3.png,OGN-066
vod1_00123_card4.png,SFD-012a
```

```bash
python -m rifteye_ml.spike real --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --crops ~/rifteye-data/real-crops --labels ~/rifteye-data/real-crops/labels.csv \
  --encoder colorgrid --encoder timm:vit_small_patch14_dinov2.lvd142m --out reports/m0-real.csv
```

Crops stay private. The CSV results can be shared, and are what goes into `docs/reports/`.

`--strips top:0.4,top:0.25,left:0.3` also scores each card from one band only, what a stack leaves visible. Each crop is first turned upright by its label (of its four turns, the one the colour grid matches best to its own card), and `full` scores the whole upright card as the baseline. `--skip-types Legend,Battlefield` keeps the cards that go in stacks; `--only-types Rune` keeps one type (types as in the catalogue).

**5. More labels by review.** Naming crops one by one does not scale. `reviewpack identity` crops every overhead frame's cards ahead of time with `matcrops`, links the crops of one physical card across frames into a *track*, and writes a review pack of the model's guesses for [apps/reviewer](../apps/reviewer): one item per track, the least confident first, plus a random 10% of the confident rest as an audit. Tracks as plain as a sleeve back are left out: face-down cards are never named, so nobody is asked about them (`--face-down`, 0 keeps them). The reviewer answers correct or wrong, and `apply` gives every crop of each answered track its label:

```bash
python -m rifteye_ml.matcrops --frames ~/rifteye-data/vods/<vod>/frames/seg-*/ --only overhead-frames.txt \
  --table 0.17,0.09,0.86,0.884 --long 131 --out ~/rifteye-data/real-crops/v2/crops
python -m rifteye_ml.reviewpack identity --crops ~/rifteye-data/real-crops/v2/crops \
  --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art --embed-cache ~/rifteye-data/embed-cache \
  --labels ~/rifteye-data/real-crops/v1/labels.csv --out ~/rifteye-data/packs/identity-v2a.json
# ... the reviewer answers in apps/reviewer and exports identity-v2a.answers.json ...
python -m rifteye_ml.reviewpack apply --pack ~/rifteye-data/packs/identity-v2a.json \
  --answers identity-v2a.answers.json --out ~/rifteye-data/real-crops/v2/labels-review.csv
```

A name the reviewer typed for a card the catalogue lacks (some tokens) becomes `printing_id` `none`, with the name in the `name` column. `--labels` does two things: it fits the softmax temperature that turns match scores into the confidence shown, and it leaves out tracks that already have a label. `apply` also prints how often the model was right, on the reviewed items and on the audit, with 95% intervals. The pack is a single JSON file with its pictures embedded (about 17 MB for 400 items). A sidecar `*.meta.json` next to it maps items to crops and never leaves the machine.

`reviewpack events` does the same for change-gate events: the table before and after each change, and the gate's guess of its kind. The gate records when each changed region last matched the still table (`extra.t_before`, about when the hand arrived), which is where the "before" picture is taken:

```bash
python -m rifteye_ml.changegate --video seg.mp4 --start 90 --duration 600 --table 0.15,0.10,0.88,0.884 --out events.json
python -m rifteye_ml.reviewpack events --events events.json --video seg.mp4 --out ~/rifteye-data/packs/events.json
```

**Reading the results.**

- `rotation=search` is the realistic setting: all four rotations are tried.
- `rotation=oracle` isolates the codec's effect by undoing the known rotation first.
- `*_card` columns roll printings up to gameplay cards, which is the product metric; `*_printing` columns require the exact printing. Printings with identical art (reprints, some signature versions) cap printing-level accuracy below 100%.
- `realism` is the simulator level; `queries` is `same` for degraded gallery images, a language tag such as `zh-Hans` for another language's printings, or `real` (with the type filter, e.g. `real, not Legend/Battlefield`).

## Synthetic boards (M1)

`python -m rifteye_ml.synth` writes whole broadcast frames of plausible 1v1 boards with exact annotations, for training the detector and the embedder ([docs/research/04 §4.3](../docs/research/04-data-and-evaluation.md#43-synthetic-board-generator-mlsynth)):

```bash
python -m rifteye_ml.synth --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --boards 400 --previews 12 --out ~/rifteye-data/synth/v0
python -m rifteye_ml.synth --fixtures 60 --boards 4 --out /tmp/synth-demo   # procedural cards, no data needed
```

- **Layout** (`synth/layout.py`): the tournament table (Tournament Rules 508). Each player has runes in a spread row, a fanned row or short columns with the top of each rune showing; units in the base, some fanned and some with gear tucked under; the legend (often with a die) and the chosen champion; decks and trash as piles; units at the two battlefields; hidden cards in the facedown slots. Ready cards face their controller, and exhausted ones are turned a quarter turn. Counters, markers and hands go on top.
- **Filming** (`synth/compose.py`): sleeves, foil and glare from the M0 stream simulator; a soft shadow under every card; a camera with a quarter turn, yaw, keystone tilt and lens distortion in one mesh warp; white balance, exposure, stage light, a muted broadcast tone and defocus. The camera window fills the frame, sits between side panels with a featured-card graphic (the RQ package), or is a picture-in-picture. The camera frames 550–950 mm of table, so card size follows from the window: about 100–165 px full screen at 1080p, 35–100 px in a picture-in-picture.
- **Codec**: boards are grouped into clips of one resolution (1080p, 720p or 480p) and bitrate (2.5–8 Mbps), held for 10 frames with sensor noise, and pushed through libx264, keeping the last frame of each board. Runs are bit-exact for a seed.
- **Output**: `frames/*.png` (lossless), `ids/*.png` (16-bit: card id + 1 where that card is uppermost, 65535 under a hand, die or counter, 65534 on a broadcast graphic), `annotations.jsonl` and `manifest.json`. Each card has its zone, controller, exhausted state, pile, `quad` (TL, TR, BR, BL of the card as printed, in frame pixels, even where it is covered or cut off), `visible` (the uncovered share of the card), `visible_box` and, when face up, its `printing_id`. Face-down cards never carry an identity.
- **Mats** are procedural (dark and muted, with a faint printed emblem). `--mats` takes a folder of your own mat images; official mats are Riot IP and are never bundled.

**Realism check.** `python -m rifteye_ml.synth.check --run ~/rifteye-data/synth/v0 --catalog … --cache …` cuts every fully visible face-up card out through its quad, with the box a few per cent off as a detector's would be, and names it with the M0 baselines. Synthetic cards should be about as hard to name as the reviewed real crops of the M0 broadcasts; the first version was far easier, and the looks were calibrated ([report](../docs/reports/m1-synth-v0.md)).

Frames made from Riot's art are private, like the art itself (D-015).

## Card detector (M1)

`python -m rifteye_ml.detect` trains the amodal detector of [ARCHITECTURE §3.8](../docs/ARCHITECTURE.md#38-stacks-and-covered-cards): every card's *full* quad, even where other cards, hands or dice cover it. The model is RF-DETR keypoint with four keypoints, the card's corners (code and weights Apache-2.0, [03 §3.2](../docs/research/03-models-and-licensing.md#32-detectors)). Install it with `pip install -e '.[detect]'`.

```bash
python -m rifteye_ml.detect export --run ~/rifteye-data/synth/v0 --out ~/rifteye-data/detect/tiles --scales 2
python -m rifteye_ml.detect train --dataset ~/rifteye-data/detect/tiles --out ~/rifteye-data/detect/v0 --device cuda
python -m rifteye_ml.detect evaluate --run ~/rifteye-data/synth/v0 --checkpoint ~/rifteye-data/detect/v0/checkpoint_best_total.pth \
  --out m1-detector-v0-synth.csv
python -m rifteye_ml.detect pack --checkpoint ~/rifteye-data/detect/v0/checkpoint_best_total.pth --out ~/rifteye-data/detect/detector-v0.pth
python -m rifteye_ml.detect run --frames ~/rifteye-data/vods/<vod>/frames/seg-000 --table 0.17,0.09,0.86,0.884 --card-px 131 \
  --checkpoint ~/rifteye-data/detect/detector-v0.pth --out dets.jsonl
```

`pack` keeps only the float16 weights, about 80 MB instead of 160 MB, for moving a trained detector between machines.

- **Tiles.** `export` scales each frame's camera window so its cards are 45–110 px on the long side, and cuts it into overlapping 576 px squares, in RF-DETR's COCO layout (`train/`, `valid/`; every 10th board is held out). Each card that shows at least 8% of itself in a tile is a target. Its box is the full card, clipped to the tile, and its four keypoints are the corners in image order (the one up and left of the centre first, then clockwise): 2 when the corner shows, 1 when it is covered, 0 outside the tile. The class is `card` or `card_back`, and no identity is ever written.
- **Detection.** `run` and `evaluate` scale the window so cards are about 70 px, detect tile by tile, and merge. Tiles overlap by more than a card, so a detection cut by a shared tile edge is dropped and the whole copy next door is kept. Nothing is suppressed below IoU 0.8, because stacked cards overlap each other by design. Each detection has the quad in frame pixels, the class, the score, and per corner the chance it was found and that it shows. The order of the corners does not say which way the card is printed; the matcher tries all four turns (§3.3).
- **Scores.** A card is found when a detection of its class overlaps its full quad with IoU ≥ 0.75. The usual 0.5 is too loose, since the cards of a rune column overlap each other by 0.5–0.7. Recall is split by how much of the card shows: whole, half or more, a strip (15–50%), a sliver (8–15%).
- **On a GPU machine.** [`scripts/m1-detector.sh`](scripts/m1-detector.sh) does everything from public sources on any Linux machine with an NVIDIA GPU: the catalogue and art, 2,000 synthetic boards on every CPU core, the tiles, training (resumable) and the scores. `SMOKE=1` runs a short version first (8 boards, 1 epoch), and the full run reuses its downloads. It refuses to start while another process uses the GPU. What it writes is private, except the CSV of numbers. For a Google Cloud GPU VM, [`scripts/m1-detector-gce.sh`](scripts/m1-detector-gce.sh) VM PROJECT ZONE runs it from your own machine: it retries the start until the zone has a GPU free, copies your checkout's code to the VM (so the VM needs no GitHub access), runs the short check and then the real run, waits for it, copies the results to `~/rifteye-m1-results` and stops the VM. If your machine goes to sleep, the VM still powers itself off 30 minutes after the run ends. [`scripts/m1-detector-anywhere.sh`](scripts/m1-detector-anywhere.sh) PROJECT does the same on a temporary VM, created in the first zone anywhere with a T4 or L4 free and quota for it, and deletes it afterwards.

RF-DETR 1.11.0 keeps the keypoint checkpoint's class head, two logits, and counts it as one class plus a no-object slot. Its default loss trains both logits as classes, so `card` and `card_back` both learn, but its own `predict()` calls `card_back` "`__background__`". `Detector` decodes the two logits itself.

## Covered cards: the stack review pack (M1)

`python -m rifteye_ml.stacks pack` turns the detector's output on real frames into a review pack of covered cards, the first real stack labels ([ARCHITECTURE §3.8](../docs/ARCHITECTURE.md#38-stacks-and-covered-cards)):

```bash
python -m rifteye_ml.detect run --frames ~/rifteye-data/vods/<vod>/frames/seg-* --only frames.txt --table 0.16,0.06,0.86,0.91 \
  --card-px 131 --checkpoint detector-v0.pth --out dets.jsonl
python -m rifteye_ml.stacks pack --dets dets.jsonl --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art \
  --embed-cache ~/rifteye-data/embed-cache --crops-out ~/rifteye-data/real-crops/stacks-<vod> --out ~/rifteye-data/packs/stacks-<vod>.json
python -m rifteye_ml.reviewpack apply --pack ~/rifteye-data/packs/stacks-<vod>.json --answers stacks-<vod>.answers.json --out labels.csv
```

- **Order:** of two overlapping cards, the one whose corners inside the other still show is on top.
- **Visible part:** the card minus the cards on it and minus anything outside the camera window. Hands and dice are not detected, so a card under a hand looks visible; the reviewer answers "can't tell".
- **Name:** the largest band along one edge that shows (60%, 40% or a quarter of the card) against the same band of every gallery card, colour grid + dHash, both ways up. The pack is an identity pack, so apps/reviewer shows it as it is: the card with its covered part dimmed, the table around it, and the guess. Face-down cards are never named.
- **Checked on exact synthetic detections** (`stacks synth-dets`, 40 frames of synth v0, 449 covered cards): named right from 60% of the card 96% (end) and 96% (side), from 40% 96% and 93%, from a quarter 72% and 46%. The real strips of M0 §7 showed the same order: ends carry the art, sides mostly frame and text.

## Card embedder (M1)

`python -m rifteye_ml.embed` fine-tunes the embedder of [ARCHITECTURE §3.4](../docs/ARCHITECTURE.md#34-embedder): DINOv2 ViT-S/14 (Apache-2.0 weights) read at its class token, a linear neck to 256 dimensions, trained with Sub-center ArcFace over gameplay cards (three centres per card, so alt arts need not share one). Install it with `pip install -e '.[torch]'`.

```bash
python -m rifteye_ml.embed crops --catalog ~/rifteye-data/catalog/catalog.jsonl --cache ~/rifteye-data/art --out bank --seeds 1-16
python -m rifteye_ml.embed crops --catalog ... --cache ... --out bank-eval --eval
python -m rifteye_ml.embed train --catalog ... --cache ... --bank bank --out runs/heldout --train-sets OGN,OGS,SFD --device cuda
python -m rifteye_ml.embed pack --checkpoint runs/heldout/final.pt --out embedder-v0-heldout.pth
python -m rifteye_ml.embed evaluate --catalog ... --cache ... --bank bank-eval --train-sets OGN,OGS,SFD \
  --encoder timm:vit_small_patch14_dinov2.lvd142m --encoder embedder:embedder-v0-heldout.pth --out scores.csv
python -m rifteye_ml.spike real --encoder embedder:embedder-v0.pth ...   # any tool that takes an encoder spec
```

- **Crop bank.** `crops` runs the M0 stream simulator over the whole catalogue once per seed and card height (10 heights, 36–176 px): boards at the camera level with foil sheen, H.264, cropped back out upright. Each task draws its own frame size (1080p or 720p), bitrate (2–8 Mbit/s) and strength (0.6, 1 or 1.4 times the camera level). Before degrading, a fifth of the standard Unit, Spell and Gear printings take the rules text box of another printing of the same type and set (text scrambling: the identity should come from the art, whatever the language). Every CPU core works, one shard per task, and a rerun skips the shards that are there.
- **Training.** A batch mixes bank crops with clean art rendered the way the gallery is (`at_long_side` at a random size), so both sides of a search learn the same space. Random covering: 30% of portrait cards are cut to the band a stack leaves visible (mostly the top, 25–70% of the card), and 15% of crops get another card over one end. Nothing is flipped. Layer-wise decayed learning rates, the margin grows over the first epoch, bfloat16 on CUDA, and `last.pt` every epoch to resume from. Each card's centres start at its printings' clean art.
- **Scores.** `evaluate` uses the M0 quick fine-tune's protocol (M0 report §8): fresh crops at the camera level, 1080p at 4 Mbit/s, seed 0, each searched against the whole catalogue at its height's gallery level, split into held-out and training sets, as the whole card and as the top 40% and top quarter (strips, M0 §7).
- **On a GPU machine.** [`scripts/m1-embedder.sh`](scripts/m1-embedder.sh) does it all from public sources, like the detector's script: the crop bank, `heldout` (trained on Origins, Proving Grounds and Spiritforged only, scored on the other sets against the frozen backbone) and `all` (every set, the one to use). `SMOKE=1` runs a short version. On the Google Cloud VM where the detector runs, [`scripts/m1-embedder-gce.sh`](scripts/m1-embedder-gce.sh) VM PROJECT ZONE `[--delete]` queues it after the detector without touching it: a short check on the CPU while you watch, then a service that waits for the detector's run to end, cancels the power-off that run has pending and takes the GPU. It copies each run's results to `~/rifteye-m1-results` as it finishes and stops (or deletes) the VM at the end. [`scripts/m1-vm.sh`](scripts/m1-vm.sh) is its VM side.

The weights are trained on Riot's card art, so they stay private like the art: never commit or publish them.

## Demo (no data needed)

```bash
python -m rifteye_ml.spike demo --cards 60
```

This runs the whole pipeline on procedural fake cards. It checks the toolchain; its numbers say nothing about Riftbound.
