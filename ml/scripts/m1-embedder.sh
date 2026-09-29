#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
#
# M1 embedder v0 on any Linux machine with an NVIDIA GPU: DINOv2-S/14 fine-tuned with Sub-center ArcFace on
# synthetic stream crops, built from this repository and public card data only (docs/research/04, D-015).
# Two models:
#   heldout  trained on the Origins, Proving Grounds and Spiritforged sets only and scored on the other
#            sets, like the M0 quick fine-tune (M0 report §8): does it hold for cards it never saw?
#   all      trained on every set: the one to use.
#
#   bash m1-embedder.sh [WORK]        # default WORK: ~/rifteye-m1-embed. Safe to re-run: every step resumes.
#   SMOKE=1 bash m1-embedder.sh       # a short end-to-end check: 48 printings, 1 seed, 1 epoch
#   REAL=real.tgz bash m1-embedder.sh # v1: real crops too (below)
#
# With REAL (a folder or a .tgz of one folder per reviewed real set, each with labels.csv and crops/; private,
# never in the repository) it trains v1 instead: one model on every card set, the synthetic bank plus every real
# set but those in REAL_TEST (default bcn,la-final: a broadcast it never sees, and a match it never sees), each
# real crop REAL_REPEAT (3) times an epoch. The REAL_TEST sets then score it against colour and structure
# together (gallery at REAL_SCALES, default 120,140,160).
#
# Steps: catalogue and art -> crop bank (all CPU cores, about 15 min) -> train heldout -> score it -> train all
# -> score it. Everything written stays in WORK and is private: the art, the crops and the weights. The files
# meant to leave the machine are WORK/results/m1-embedder-v0-*.csv (v1-*.csv with REAL; numbers only).
#
# Settings (environment): SEEDS (16), EPOCHS (8), BATCH (128), TRAIN_SETS (OGN,OGS,SFD), MIN_FREE_GB (12), CACHE
# (~/rifteye-cache, shared with m1-detector.sh), PYTHON and ALLOW_SHARED_GPU as there, VENV (default
# CACHE/venv-embed, shared by the smoke check and the real run), STOP_WHEN_DONE=1 and STOP_DELAY as in
# m1-detector.sh. AFTER_UNIT=<systemd unit> waits for that unit first; when its log (AFTER_LOG) says it has
# ended, the unit is stopped, which also cancels the power-off it has pending, and this job takes the GPU.
# For testing elsewhere: RIFTEYE_DIR (an existing checkout, used as is), SKIP_INSTALL=1, DEVICE=cpu.
set -euo pipefail

WORK=${1:-$HOME/rifteye-m1-embed}
if [ "${SMOKE:-0}" = 1 ]; then
  WORK=${1:-$HOME/rifteye-m1-embed-smoke}; LIMIT=${LIMIT:-48}; SEEDS=${SEEDS:-1}; HEIGHTS=${HEIGHTS:-48,96}
  EVAL_HEIGHTS=${EVAL_HEIGHTS:-40,80}; EPOCHS=${EPOCHS:-1}; BATCH=${BATCH:-16}; CLEAN=${CLEAN:-2}
fi
SEEDS=${SEEDS:-16}; EPOCHS=${EPOCHS:-8}; BATCH=${BATCH:-128}; CLEAN=${CLEAN:-20}; LIMIT=${LIMIT:-0}
HEIGHTS=${HEIGHTS:-}; EVAL_HEIGHTS=${EVAL_HEIGHTS:-}; TRAIN_SETS=${TRAIN_SETS:-OGN,OGS,SFD}
REAL=${REAL:-}; REAL_TEST=${REAL_TEST:-bcn,la-final}; REAL_REPEAT=${REAL_REPEAT:-3}; REAL_SCALES=${REAL_SCALES:-120,140,160}
V=v0; [ -z "$REAL" ] || V=v1
BACKBONE=vit_small_patch14_dinov2.lvd142m
REV=${RIFTEYE_REV:-main}; SRC=${RIFTEYE_DIR:-$WORK/RiftEye}; DEVICE=${DEVICE:-cuda}
D=$WORK/data; C=${CACHE:-$HOME/rifteye-cache}; VENV=${VENV:-$C/venv-embed}
mkdir -p "$D" "$C" "$WORK/results" "$WORK/logs"
exec > >(tee -a "$WORK/logs/run.log") 2>&1
step() { echo; echo "== $(date -u +%H:%M:%S) $*"; }
T0=$(date +%s)
# Never while the job it waits for still runs: that one powers the machine off itself when it ends.
stop_machine() {
  echo "== ended; powering off in $(( ${STOP_DELAY:-0} / 60 )) min"
  sleep "${STOP_DELAY:-0}"
  if [ -n "${AFTER_UNIT:-}" ] && systemctl is-active --quiet "$AFTER_UNIT"; then
    echo "$AFTER_UNIT is still running, so the machine stays on until it ends"
  else
    sudo poweroff
  fi
}
if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then trap stop_machine EXIT; fi

if [ -n "${AFTER_UNIT:-}" ]; then
  step "waiting for $AFTER_UNIT to finish"
  while systemctl is-active --quiet "$AFTER_UNIT"; do
    last=$(grep '^==' "${AFTER_LOG:-/dev/null}" 2>/dev/null | tail -n 1 || true)
    case "$last" in
      "== ended"*)  # it is only waiting to power the machine off: stop it, and the power-off with it
        echo "$AFTER_UNIT has ended; stopping it so it does not power the machine off"
        sudo systemctl kill --signal=SIGKILL "$AFTER_UNIT" || true
        sleep 5 ;;
      *) sleep "${AFTER_POLL:-60}" ;;
    esac
  done
  echo "$AFTER_UNIT is not running. Its last steps:"
  grep '^==' "${AFTER_LOG:-/dev/null}" 2>/dev/null | tail -n 3 || true
fi

if [ "$DEVICE" = cuda ]; then
  step "GPU"
  for i in $(seq 1 30); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done  # a new VM may still be installing it
  command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver (nvidia-smi) on this machine"; exit 1; }
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
  if [ "${ALLOW_SHARED_GPU:-0}" != 1 ]; then
    for i in $(seq 1 24); do  # a job that has just been stopped can hold the GPU for a few seconds
      busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
      [ "$busy" -eq 0 ] && break
      sleep 5
    done
    if [ "$busy" -gt 0 ]; then echo "another process is using the GPU ($busy); not starting. ALLOW_SHARED_GPU=1 overrides."; exit 1; fi
  fi
fi
# The full run writes about 6 GB (the crop bank 5, checkpoints 1). The short check tests for that too, so a
# full disk shows up while someone watches.
need=${MIN_FREE_GB:-12}
free=$(( $(df -Pk "$WORK" | awk 'NR == 2 {print $4}') / 1048576 ))
echo "disk: $free GB free for $WORK; the full run wants $need GB"
[ "$free" -ge "$need" ] || { echo "not enough disk space for the full run (MIN_FREE_GB overrides the check)"; exit 1; }

step "code"
if [ -z "${RIFTEYE_DIR:-}" ]; then
  export GIT_TERMINAL_PROMPT=0
  [ -d "$SRC/.git" ] || git clone -q https://github.com/effe-exe/wardeye.git "$SRC" || {
    echo "cannot clone Wardeye here (while the repository is private, this machine needs GitHub access)."
    echo "Copy a checkout to this machine and set RIFTEYE_DIR to it; m1-embedder-gce.sh does that for Google Cloud VMs."
    exit 1
  }
  git -C "$SRC" fetch -q origin
  git -C "$SRC" checkout -q --detach "origin/$REV" 2>/dev/null || git -C "$SRC" checkout -q --detach "$REV"
fi
git -C "$SRC" log -1 --format='%h %s' 2>/dev/null || echo "code from $SRC"

step "python"
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then rm -rf "$VENV"; fi  # a half-made one
if [ ! -x "$VENV/bin/python" ]; then
  if ! python3 -c 'import ensurepip' 2>/dev/null; then  # Ubuntu ships venv's pip bootstrap separately
    sudo apt-get -qq -o DPkg::Lock::Timeout=900 update
    sudo DEBIAN_FRONTEND=noninteractive apt-get -qq -o DPkg::Lock::Timeout=900 install -y python3-venv >/dev/null
  fi
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
  pip install -q -e "$SRC/ml[torch]"
fi
if [ "$DEVICE" = cuda ]; then
  python -c 'import torch; assert torch.cuda.is_available(), "torch cannot see the GPU"; print("torch", torch.__version__, "on", torch.cuda.get_device_name(0))'
else
  export CUDA_VISIBLE_DEVICES=""  # a CPU run leaves the GPU alone, e.g. while another job trains on it
fi
python -c 'import timm, torch; print("timm", timm.__version__, "torch", torch.__version__)'
export HF_HOME=$C/hf  # the pretrained DINOv2 weights (Apache-2.0) are cached here

step "catalogue and art (public feed and CDN)"
CAT=$C/catalog/catalog.jsonl
if [ ! -s "$CAT" ]; then
  python -m rifteye_ml.catalog fetch-feed --out "$C/catalog/feed"
  python -m rifteye_ml.catalog build --feed "$C/catalog/feed" --out "$CAT"
fi
python -m rifteye_ml.catalog download --catalog "$CAT" --cache "$C/art"
data=(--catalog "$CAT" --cache "$C/art")

step "crop bank: $SEEDS seeds of every printing at every size"
P=$(nproc); P=$(( P > 8 ? 8 : P ))
python -m rifteye_ml.embed crops "${data[@]}" --out "$D/bank" --seeds "1-$SEEDS" ${HEIGHTS:+--heights "$HEIGHTS"} \
  --workers "$P" --limit "$LIMIT"
python -m rifteye_ml.embed crops "${data[@]}" --out "$D/bank-eval" --eval ${EVAL_HEIGHTS:+--heights "$EVAL_HEIGHTS"} \
  --workers "$P" --limit "$LIMIT"
du -sh "$D/bank" "$D/bank-eval"

banks=("$D/bank")
if [ -n "$REAL" ]; then
  step "real crops: every reviewed set but $REAL_TEST"
  R=$REAL
  if [ -f "$REAL" ]; then
    R=$D/real
    [ -f "$R/.unpacked" ] || { rm -rf "$R"; mkdir -p "$R"; tar -xzf "$REAL" -C "$R"; touch "$R/.unpacked"; }
  fi
  for s in "$R"/*/; do
    s=${s%/}; name=$(basename "$s")
    if [ ! -f "$s/labels.csv" ]; then continue; fi
    case ",$REAL_TEST," in *",$name,"*) continue ;; esac
    [ -f "$D/real-$name/real.npy" ] || python -m rifteye_ml.embed real-bank "${data[@]}" --crops "$s/crops" \
      --labels "$s/labels.csv" --out "$D/real-$name" --repeat "$REAL_REPEAT" --limit "$LIMIT"
    banks+=("$D/real-$name")
  done
  [ "${#banks[@]}" -gt 1 ] || { echo "no reviewed real set with labels.csv in $REAL"; exit 1; }
  for t in ${REAL_TEST//,/ }; do [ -f "$R/$t/labels.csv" ] || { echo "no test set $t in $REAL"; exit 1; }; done
fi

W=$(( $(nproc) - 1 )); W=$(( W > 8 ? 8 : W )); W=$(( W < 1 ? 1 : W ))
failed=0
fit() {  # name, training sets (empty: all), weights file
  local out=$D/$1
  if [ ! -f "$out/final.pt" ]; then
    python -m rifteye_ml.embed train "${data[@]}" --bank "${banks[@]}" --out "$out" --train-sets "$2" --epochs "$EPOCHS" \
      --batch-size "$BATCH" --clean "$CLEAN" --device "$DEVICE" --workers "$W"
  fi
  cp "$out/metrics.csv" "$WORK/results/m1-embedder-$V-$1-training.csv"
  python -m rifteye_ml.embed pack --checkpoint "$out/final.pt" --out "$WORK/results/$3"
}
score() {  # csv, encoder specs...: a failed score keeps the weights and the run going
  local out=$1; shift
  local specs=(); for s in "$@"; do specs+=(--encoder "$s"); done
  python -m rifteye_ml.embed evaluate "${data[@]}" --bank "$D/bank-eval" --train-sets "$TRAIN_SETS" "${specs[@]}" \
    --out "$WORK/results/$out" || { echo "!! scoring failed (the weights are kept; score them again later)"; failed=$((failed + 1)); }
}

if [ "$V" = v0 ]; then
  step "train heldout: sets $TRAIN_SETS, $EPOCHS epochs, batch $BATCH"
  fit heldout "$TRAIN_SETS" embedder-v0-heldout.pth
  step "score heldout and the frozen backbone on fresh crops"
  score m1-embedder-v0-heldout-synth.csv "timm:$BACKBONE" "embedder:$WORK/results/embedder-v0-heldout.pth"
fi

step "train all: every set, $EPOCHS epochs, batch $BATCH${REAL:+, with the real crops}"
fit all "" "embedder-$V.pth"
step "score all on fresh crops"
score "m1-embedder-$V-all-synth.csv" "embedder:$WORK/results/embedder-$V.pth"

for t in ${REAL:+${REAL_TEST//,/ }}; do
  step "score on the real set $t, which it did not train on"
  for only in "" "--only-types Legend" "--skip-types Legend"; do
    # shellcheck disable=SC2086  # $only is two words or none
    python -m rifteye_ml.spike real "${data[@]}" --crops "$R/$t/crops" --labels "$R/$t/labels.csv" \
      --encoder colorgrid/trim0.03+dhash/trim0.03 --encoder "embedder:$WORK/results/embedder-v1.pth" \
      --gallery-scales "$REAL_SCALES" --embed-cache "$D/embed-cache" $only --out "$WORK/results/m1-embedder-v1-real-$t.csv" ||
      { echo "!! scoring failed (the weights are kept; score them again later)"; failed=$((failed + 1)); }
  done
done

if [ "$failed" -gt 0 ]; then
  step "finished in $(( ($(date +%s) - T0) / 60 )) min, but $failed scoring step(s) failed (see above)"
  exit 1
fi
step "done in $(( ($(date +%s) - T0) / 60 )) min"
echo "numbers:  $WORK/results/m1-embedder-$V-*.csv (safe to share)"
echo "weights:  $WORK/results/embedder-$V.pth$([ "$V" = v0 ] && echo " and embedder-v0-heldout.pth") (float16, private)"
