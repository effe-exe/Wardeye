#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
#
# The VM side of m1-embedder-gce.sh, run over SSH from the copy of this checkout on the VM:
#
#   bash m1-vm.sh status            one line per job, "name|service state|finished ok|last step", then "ok"
#   bash m1-vm.sh start-embedder    m1-embedder.sh as the service rifteye-m1-embed. It waits for the detector's
#                                   service (rifteye-m1) to finish, then takes the GPU. REAL, REAL_TEST, REAL_REPEAT,
#                                   REAL_SCALES, SEEDS, EPOCHS, CLEAN and BATCH are passed on when set.
#   bash m1-vm.sh resume-detector   m1-detector.sh again as the service rifteye-m1; it resumes where it stopped
#   bash m1-vm.sh pack detector|embedder   the job's results and log in ~/rifteye-m1-out/<job>.tgz
#
# Services belong to the VM's service manager, so they outlive the SSH session that starts them.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
DET=$HOME/rifteye-m1
EMB=$HOME/rifteye-m1-embed

status() {  # name unit work
  local state steps last ok=no
  state=$(systemctl is-active "$2" 2>/dev/null) || true
  steps=$(grep '^==' "$3/logs/run.log" 2>/dev/null) || true
  last=$(printf '%s\n' "$steps" | tail -n 1)
  case "$(printf '%s\n' "$steps" | tail -n 2)" in *" done in "*) ok=yes ;; esac
  echo "$1|${state:-unknown}|$ok|$last"
}

start() {  # unit script VAR=value...
  local unit=$1 script=$2 kv
  shift 2
  if systemctl is-active --quiet "$unit"; then echo "$unit is already running"; return 0; fi
  sudo systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  local env=(--setenv "HOME=$HOME" --setenv "PATH=$PATH")
  for kv in "$@"; do env+=(--setenv "$kv"); done
  sudo systemd-run --quiet --collect --unit "$unit" --uid "$(id -u)" --gid "$(id -g)" \
    --property "WorkingDirectory=$HOME" "${env[@]}" bash "$script" || return 1
  sleep 10
  if ! systemctl is-active --quiet "$unit"; then
    echo "$unit did not stay up. What it said:"
    sudo journalctl -u "$unit" --no-pager 2>/dev/null | tail -n 30
    return 1
  fi
  echo "$unit started"
}

pack() {  # name work
  local out=$HOME/rifteye-m1-out
  rm -rf "${out:?}/$1" "$out/$1.tgz"
  mkdir -p "$out/$1"
  cp -p "$2"/results/* "$out/$1/" 2>/dev/null || true
  cp -p "$2/logs/run.log" "$out/$1/$1-run.log" 2>/dev/null || true
  tar -czf "$out/$1.tgz" -C "$out/$1" .
  echo "packed $(ls "$out/$1" | wc -l | tr -d ' ') files"
}

case "${1:-}" in
  status)
    status detector rifteye-m1 "$DET"
    status embedder rifteye-m1-embed "$EMB"
    echo ok ;;
  start-embedder)
    pass=()
    for v in REAL REAL_TEST REAL_REPEAT REAL_SCALES SEEDS EPOCHS CLEAN BATCH; do
      if [ -n "${!v:-}" ]; then pass+=("$v=${!v}"); fi
    done
    start rifteye-m1-embed "$SRC/ml/scripts/m1-embedder.sh" "RIFTEYE_DIR=$SRC" AFTER_UNIT=rifteye-m1 \
      "AFTER_LOG=$DET/logs/run.log" STOP_WHEN_DONE=1 STOP_DELAY=1800 ${pass[@]+"${pass[@]}"} ;;
  resume-detector)
    start rifteye-m1 "$HOME/RiftEye-src/ml/scripts/m1-detector.sh" "RIFTEYE_DIR=$HOME/RiftEye-src" \
      STOP_WHEN_DONE=1 STOP_DELAY=1800 ;;
  pack)
    case "${2:-}" in
      detector) pack detector "$DET" ;;
      embedder) pack embedder "$EMB" ;;
      *) echo "usage: m1-vm.sh pack detector|embedder"; exit 2 ;;
    esac ;;
  *) echo "usage: m1-vm.sh status|start-embedder|resume-detector|pack <job>"; exit 2 ;;
esac
