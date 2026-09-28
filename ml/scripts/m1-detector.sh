#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
#
# M1 detector v0 on any Linux machine with an NVIDIA GPU, built from this repository and public card
# data only: the public card-gallery feed and Riot's public image CDN (docs/research/04, D-015).
#
#   bash m1-detector.sh [WORK]        # default WORK: ~/rifteye-m1. Safe to re-run: every step resumes.
#   SMOKE=1 bash m1-detector.sh       # a short end-to-end check first: 8 boards, 1 epoch (most of its
#                                     # time is the one-off downloads, which the full run then reuses)
#
# Steps: catalogue and art -> synthetic boards (all CPU cores) -> training tiles -> RF-DETR keypoint
# fine-tune -> scores on held-out boards. Everything written stays in WORK and is private: the art,
# the frames made from it and the weights. The file meant to leave the machine is
# WORK/results/m1-detector-v0-synth.csv (numbers only), with the training curve next to it.
#
# Settings (environment): BOARDS (2000), EPOCHS (10), BATCH (8), ACCUM (2), SEED (2026), RIFTEYE_REV
# (main), CACHE (~/rifteye-cache: the downloaded catalogue, art and pretrained weights, shared between
# runs), PYTHON (a Python that already has CUDA torch, e.g. /opt/conda/bin/python; found if unset),
# ALLOW_SHARED_GPU=1 to run while another process uses the GPU, STOP_WHEN_DONE=1 to power the machine
# off when the script ends, finished or failed (after STOP_DELAY seconds, default 0). For testing the script elsewhere:
# RIFTEYE_DIR (an existing checkout, used as is), VENV (an existing environment), SKIP_INSTALL=1, DEVICE=cpu.
set -euo pipefail

WORK=${1:-$HOME/rifteye-m1}
if [ "${SMOKE:-0}" = 1 ]; then WORK=${1:-$HOME/rifteye-m1-smoke}; BOARDS=${BOARDS:-8}; EPOCHS=${EPOCHS:-1}; VAL_EVERY=${VAL_EVERY:-2}; fi
BOARDS=${BOARDS:-2000}; EPOCHS=${EPOCHS:-10}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}; SEED=${SEED:-2026}
VAL_EVERY=${VAL_EVERY:-10}  # every 10th board of each run is held out
REV=${RIFTEYE_REV:-main}; SRC=${RIFTEYE_DIR:-$WORK/RiftEye}; VENV=${VENV:-$WORK/venv}; DEVICE=${DEVICE:-cuda}
D=$WORK/data; C=${CACHE:-$HOME/rifteye-cache}
mkdir -p "$D" "$C" "$WORK/results" "$WORK/logs"
exec > >(tee -a "$WORK/logs/run.log") 2>&1
step() { echo; echo "== $(date -u +%H:%M:%S) $*"; }
T0=$(date +%s)
# Unattended runs power the machine off when the script ends, whether it finished or failed; the log stays.
# STOP_DELAY (seconds) leaves time to copy the results off first.
if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then
  trap 'echo "== ended; powering off in $(( ${STOP_DELAY:-0} / 60 )) min"; sleep "${STOP_DELAY:-0}"; sudo poweroff' EXIT
fi

if [ "$DEVICE" = cuda ]; then
  step "GPU"
  command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver (nvidia-smi) on this machine"; exit 1; }
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [ "$busy" -gt 0 ] && [ "${ALLOW_SHARED_GPU:-0}" != 1 ]; then
    echo "another process is using the GPU ($busy); not starting. ALLOW_SHARED_GPU=1 overrides."; exit 1
  fi
fi

step "code"
if [ -z "${RIFTEYE_DIR:-}" ]; then
  export GIT_TERMINAL_PROMPT=0  # fail fast rather than wait for a password nobody types
  [ -d "$SRC/.git" ] || git clone -q https://github.com/effe-exe/RiftEye.git "$SRC" || {
    echo "cannot clone RiftEye here (while the repository is private, this machine needs GitHub access)."
    echo "Copy a checkout to this machine and set RIFTEYE_DIR to it; m1-detector-gce.sh does that for Google Cloud VMs."
    exit 1
  }
  git -C "$SRC" fetch -q origin
  git -C "$SRC" checkout -q --detach "origin/$REV" 2>/dev/null || git -C "$SRC" checkout -q --detach "$REV"
fi
git -C "$SRC" log -1 --format='%h %s' 2>/dev/null || echo "code from $SRC"

step "python"
if [ ! -x "$VENV/bin/python" ]; then
  BASE=${PYTHON:-}
  if [ -z "$BASE" ]; then  # prefer a Python whose torch already works with this driver
    for c in /opt/conda/bin/python python3; do
      if "$c" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then BASE=$c; break; fi
    done
  fi
  if [ -n "$BASE" ]; then "$BASE" -m venv --system-site-packages "$VENV"; else python3 -m venv "$VENV"; fi
fi
. "$VENV/bin/activate"
if [ "${SKIP_INSTALL:-0}" != 1 ]; then
  pip install -q --upgrade pip
  pip install -q -e "$SRC/ml[detect]"
fi
if [ "$DEVICE" = cuda ]; then
  python -c 'import torch; assert torch.cuda.is_available(), "torch cannot see the GPU"; print("torch", torch.__version__, "on", torch.cuda.get_device_name(0))'
fi
export RF_HOME=$C/models  # RF-DETR's pretrained weights (Apache-2.0) are cached here

step "catalogue and art (public feed and CDN, about 1.1 GB)"
CAT=$C/catalog/catalog.jsonl
if [ ! -s "$CAT" ]; then
  python -m rifteye_ml.catalog fetch-feed --out "$C/catalog/feed"
  python -m rifteye_ml.catalog build --feed "$C/catalog/feed" --out "$CAT"
fi
python -m rifteye_ml.catalog download --catalog "$CAT" --cache "$C/art"

step "synthetic boards: $BOARDS"
P=$(nproc); P=$(( P > 8 ? 8 : P )); P=$(( P > BOARDS / VAL_EVERY ? BOARDS / VAL_EVERY : P )); P=$(( P < 1 ? 1 : P ))
runs=(); pids=()
for p in $(seq 0 $((P - 1))); do
  n=$(( BOARDS / P + (p < BOARDS % P ? 1 : 0) ))
  r=$D/synth/run$p; runs+=("$r")
  if [ ! -s "$r/manifest.json" ]; then
    rm -rf "$r"
    python -m rifteye_ml.synth --catalog "$CAT" --cache "$C/art" --boards "$n" \
      --seed $((SEED + p)) --out "$r" > "$WORK/logs/synth$p.log" 2>&1 &
    pids+=($!)
  fi
done
# Wait for these runs only: a bare `wait` would also wait for the log's tee, which never ends.
[ ${#pids[@]} -eq 0 ] || wait "${pids[@]}" || true
for r in "${runs[@]}"; do [ -s "$r/manifest.json" ] || { echo "synthetic run $r failed, see $WORK/logs"; exit 1; }; done
tail -n 1 "$WORK"/logs/synth*.log 2>/dev/null || true

step "training tiles"
if [ ! -f "$D/detect/tiles/.done" ]; then
  rm -rf "$D/detect/tiles"
  python -m rifteye_ml.detect export --run "${runs[@]}" --out "$D/detect/tiles" --scales 2 --val-every "$VAL_EVERY"
  touch "$D/detect/tiles/.done"
fi

step "train RF-DETR keypoint: $EPOCHS epochs, batch $BATCH x $ACCUM"
OUT=$D/detect/v0
if [ ! -f "$OUT/.done" ]; then
  resume=(); [ -f "$OUT/last.ckpt" ] && resume=(--resume "$OUT/last.ckpt") && echo "resuming from $OUT/last.ckpt"
  python -m rifteye_ml.detect train --dataset "$D/detect/tiles" --out "$OUT" --epochs "$EPOCHS" \
    --batch-size "$BATCH" --grad-accum "$ACCUM" --device "$DEVICE" --workers "$(( $(nproc) > 8 ? 8 : $(nproc) ))" "${resume[@]}"
  touch "$OUT/.done"
fi
cp "$OUT/metrics.csv" "$WORK/results/m1-detector-v0-training.csv" 2>/dev/null || true

step "score on held-out boards"
python -m rifteye_ml.detect evaluate --run "${runs[@]}" --checkpoint "$OUT/checkpoint_best_total.pth" \
  --val-every "$VAL_EVERY" --device "$DEVICE" --max-frames 400 --out "$WORK/results/m1-detector-v0-synth.csv"

python -m rifteye_ml.detect pack --checkpoint "$OUT/checkpoint_best_total.pth" --out "$WORK/results/detector-v0.pth"

step "done in $(( ($(date +%s) - T0) / 60 )) min"
echo "numbers:  $WORK/results/m1-detector-v0-synth.csv and m1-detector-v0-training.csv (safe to share)"
echo "weights:  $WORK/results/detector-v0.pth (float16, private)"
