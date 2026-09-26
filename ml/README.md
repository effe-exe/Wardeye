# rifteye-ml

Research, training and evaluation tools for RiftEye (Python 3.11+). Right now this is the **M0 feasibility spike**. It measures how well cards can be identified from art as they shrink and get compressed on stream ([docs/research/04 §4.8](../docs/research/04-data-and-evaluation.md#48-the-m0-feasibility-spike)).

| Module | What it does |
|---|---|
| `catalog` | Saves Riot's public card gallery feed, normalises it into `catalog.jsonl` and caches the card images **locally** |
| `degrade` | The stream simulator: cards on a table at a target on-screen height, sleeves, glare, sensor noise, then a **real libx264 encode and decode** at stream bitrates. The `camera` realism level adds tilt, lighting, defocus, occluders and detector box error |
| `encoders` | `colorgrid` and `dhash` baselines, plus any pretrained `timm:` backbone (e.g. DINOv2) |
| `retrieval` | Brute-force gallery search with 4-rotation matching; printing-level and card-level top-k |
| `spike` | Accuracy vs card height × bitrate (synthetic), or vs crop height (real labeled crops) |
| `report` | Markdown tables and an SVG accuracy-vs-height chart from spike CSVs (numbers only) |
| `changegate` | Layer 1: when and where the table changed (settled changes vs a still-table model; hands and light ignored), from a VOD window |
| `matcrops` | A classical bootstrap detector for M0: isolated cards on a known playmat, straightened into upright crops (not the product detector) |
| `label` | Model-assisted labeling of real crops: ranked candidates and verification sheets; a person decides |
| `reviewpack` | Review packs for `apps/reviewer`: the model's guesses, one per track of the same card, least confident first plus a random audit; and exported answers back into labels |
| `adapter` | The M0 "quick fine-tune": a linear head on a frozen backbone, trained on synthetic camera-level crops, evaluated on held-out sets |
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

**5. More labels by review.** Naming crops one by one does not scale. `reviewpack identity` crops every overhead frame's cards ahead of time with `matcrops`, links the crops of one physical card across frames into a *track*, and writes a review pack of the model's guesses for [apps/reviewer](../apps/reviewer): one item per track, the least confident first, plus a random 10% of the confident rest as an audit. The reviewer answers correct or wrong, and `apply` gives every crop of each answered track its label:

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

`--labels` does two things: it fits the softmax temperature that turns match scores into the confidence shown, and it leaves out tracks that already have a label. `apply` also prints how often the model was right, on the reviewed items and on the audit, with 95% intervals. The pack is a single JSON file with its pictures embedded (about 17 MB for 400 items). A sidecar `*.meta.json` next to it maps items to crops and never leaves the machine.

`reviewpack events` does the same for change-gate events: the table before and after each change, and the gate's guess of its kind. The gate records when each changed region last matched the still table (`extra.t_before`, about when the hand arrived), which is where the "before" picture is taken:

```bash
python -m rifteye_ml.changegate --video seg.mp4 --start 90 --duration 600 --table 0.15,0.10,0.88,0.884 --out events.json
python -m rifteye_ml.reviewpack events --events events.json --video seg.mp4 --out ~/rifteye-data/packs/events.json
```

**Reading the results.**

- `rotation=search` is the realistic setting: all four rotations are tried.
- `rotation=oracle` isolates the codec's effect by undoing the known rotation first.
- `*_card` columns roll printings up to gameplay cards, which is the product metric; `*_printing` columns require the exact printing. Printings with identical art (reprints, some signature versions) cap printing-level accuracy below 100%.
- `realism` is the simulator level; `queries` is `same` for degraded gallery images, a language tag such as `zh-Hans` for another language's printings, or `real`.

## Demo (no data needed)

```bash
python -m rifteye_ml.spike demo --cards 60
```

This runs the whole pipeline on procedural fake cards. It checks the toolchain; its numbers say nothing about Riftbound.
