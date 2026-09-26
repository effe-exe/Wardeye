# rifteye-ml

Research, training and evaluation tools for RiftEye (Python 3.11+). Right now this is the **M0 feasibility spike**. It measures how well cards can be identified from art as they shrink and get compressed on stream ([docs/research/04 §4.8](../docs/research/04-data-and-evaluation.md#48-the-m0-feasibility-spike)).

| Module | What it does |
|---|---|
| `catalog` | Normalises a card list into `catalog.jsonl` and caches the card images **locally** |
| `degrade` | The stream simulator: cards on a table at a target on-screen height, sleeves, glare, sensor noise, then a **real libx264 encode and decode** at stream bitrates |
| `encoders` | `colorgrid` and `dhash` baselines, plus any pretrained `timm:` backbone (e.g. DINOv2) |
| `retrieval` | Brute-force gallery search with 4-rotation matching; printing-level and card-level top-k |
| `spike` | Accuracy vs card height × bitrate (synthetic), or vs crop height (real labeled crops) |
| `index` | Writes the shipped index (float16 matrix + manifest) and refuses to load it with a different encoder |
| `fixtures` | Procedural fake cards for tests and demos. No Riot content |

> **Card images and text are Riot Games IP.** The tools cache them on your machine for research only. Never commit them, upload them or put them in a release ([D-006](../docs/decisions.md#d-006-no-third-party-media-in-git), [D-015](../docs/decisions.md#d-015-no-riot-api-no-riot-assets-distributed)).

## Setup

```bash
cd ml
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'           # numpy, pillow, a bundled ffmpeg with libx264, pytest
pip install -e '.[torch]'         # optional: torch + timm, for pretrained encoders
pytest -q                         # ~5 s, fake cards only
```

`imageio-ffmpeg` provides an ffmpeg binary for the H.264 pass. It is a development tool and is never shipped.

## Run the spike on real cards

**1. Get a card list** in the official gallery's shape. The public mirror at `github.com/slimtreble/Riftbound-card-data` works:

```bash
mkdir -p ~/rifteye-data && cd ~/rifteye-data
curl -L -o gallery.json https://raw.githubusercontent.com/slimtreble/Riftbound-card-data/main/cards.json
```

Check the mirror's README for the current file name if that URL moves.

**2. Build the catalogue and cache the images locally.** The images come from Riot's public CDN; about 1,200 files.

```bash
python -m rifteye_ml.catalog build --gallery ~/rifteye-data/gallery.json --out ~/rifteye-data/catalog.jsonl
python -m rifteye_ml.catalog download --catalog ~/rifteye-data/catalog.jsonl --cache ~/rifteye-data/art
```

A TCGplayer/TCGCSV-style JSONL (fields `name`, `number`, `image`) can be merged in with `--jsonl` to add promos the gallery lacks.

**3. Synthetic curve.** Every card at 40–160 px and 2–6 Mbps. Start with the baselines, then add DINOv2:

```bash
python -m rifteye_ml.spike synthetic --catalog ~/rifteye-data/catalog.jsonl --cache ~/rifteye-data/art \
  --encoder colorgrid --encoder dhash \
  --encoder timm:vit_small_patch14_dinov2.lvd142m \
  --heights 40,60,80,120,160 --bitrates 2000,4000,6000 --out reports/m0-synthetic.csv
```

This runs about 15 settings. The simulation costs roughly 1–2 minutes per setting for the whole catalogue at 1080p on a laptop CPU; a GPU speeds up the DINOv2 part. Use `--limit 300` for a quick first look.

**4. Real curve** (the one that decides). Put hand-labeled crops from real, permitted footage in a folder, with a `labels.csv`:

```csv
file,printing_id
vod1_00123_card3.png,OGN-066
vod1_00123_card4.png,SFD-012a
```

```bash
python -m rifteye_ml.spike real --catalog ~/rifteye-data/catalog.jsonl --cache ~/rifteye-data/art \
  --crops ~/rifteye-data/real-crops --labels ~/rifteye-data/real-crops/labels.csv \
  --encoder colorgrid --encoder timm:vit_small_patch14_dinov2.lvd142m --out reports/m0-real.csv
```

Crops stay private. The CSV results can be shared, and are what goes into `docs/reports/`.

**Reading the results.**

- `rotation=search` is the realistic setting: all four rotations are tried.
- `rotation=oracle` isolates the codec's effect by undoing the known rotation first.
- `*_card` columns roll printings up to gameplay cards, which is the product metric; `*_printing` columns require the exact printing.

## Demo (no data needed)

```bash
python -m rifteye_ml.spike demo --cards 60
```

This runs the whole pipeline on procedural fake cards. It checks the toolchain; its numbers say nothing about Riftbound.
