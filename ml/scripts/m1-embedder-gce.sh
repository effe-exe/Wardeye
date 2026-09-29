#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and Wardeye contributors
#
# Run the M1 embedder (m1-embedder.sh) on a Google Cloud GPU VM, after the detector run that is going there,
# from your own machine (needs gcloud and a checkout of this repository):
#
#   bash ml/scripts/m1-embedder-gce.sh VM PROJECT ZONE [--delete]
#   REAL=real.tgz bash ml/scripts/m1-embedder-gce.sh ...   # v1, with reviewed real crops (m1-embedder.sh)
#
# A detector run that is going is left alone. This copies this checkout's code to the VM next to the
# detector's copy, runs a short check of the embedder on the CPU while you watch, and starts the embedder as
# a service that waits for the detector to finish and then takes the GPU. It looks at both every 5 minutes,
# copies each one's results to ~/rifteye-m1-results as it finishes, and stops the VM when neither runs any
# more (--delete: deletes it instead, if both finished and their results are here). Safe to run again at any
# point: it picks up where things are. Keep this machine awake (macOS: caffeinate -i bash ...). If it cannot,
# the VM still powers itself off 30 minutes after the last run ends, with the results on its disk. With no
# detector run on the VM, the short check uses the GPU. REAL (a .tgz here) is copied to the VM; it and
# REAL_TEST, REAL_REPEAT, REAL_SCALES, SEEDS, EPOCHS, CLEAN and BATCH go to the embedder when set.
set -eu
VM=${1:?usage: m1-embedder-gce.sh VM PROJECT ZONE [--delete]}; PROJECT=${2:?project}; ZONE=${3:?zone}
DELETE=no
if [ "${4:-}" = --delete ]; then DELETE=yes; fi
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PKG="${TMPDIR:-/tmp}/rifteye-embed-code-$$.tgz"
OUT="$HOME/rifteye-m1-results"
POLL=${POLL:-300}
at() { date +%H:%M; }
on() { gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap -- "$@"; }
copy() { gcloud compute scp "$@" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap; }
retry() { for _ in 1 2 3; do if "$@"; then return 0; fi; sleep 15; done; return 1; }
vm_state() { gcloud compute instances describe "$VM" --project "$PROJECT" --zone "$ZONE" --format="value(status)" 2>/dev/null || echo unknown; }
# Both jobs as the VM sees them, or nothing when the VM did not answer (SSH through IAP drops now and then).
query() {
  local s
  s=$(on "bash ~/RiftEye-embed/ml/scripts/m1-vm.sh status" 2>/dev/null | tr -d '\r') || true
  if [ "$(printf '%s\n' "$s" | tail -n 1)" = ok ]; then printf '%s\n' "$s"; fi
}
get() { printf '%s\n' "$1" | grep "^$2|" | head -n 1 | cut -d'|' -f"$3"; }  # state job field (2, 3 or 4-)
read_state() {
  det_state=$(get "$1" detector 2); det_ok=$(get "$1" detector 3); det_last=$(get "$1" detector 4-)
  emb_state=$(get "$1" embedder 2); emb_ok=$(get "$1" embedder 3); emb_last=$(get "$1" embedder 4-)
}
running() { case "$1" in active|activating|deactivating|reloading) return 0 ;; esac; return 1; }
finished() {  # state last-step: not running, or only waiting to power the machine off
  case "$2" in "== ended"*) return 0 ;; esac
  ! running "$1"
}
fetch() {  # job: its results and log into $OUT
  mkdir -p "$OUT"
  on "bash ~/RiftEye-embed/ml/scripts/m1-vm.sh pack $1" >/dev/null 2>&1 || return 1
  copy "$VM:rifteye-m1-out/$1.tgz" "$OUT/.$1.tgz" >/dev/null 2>&1 || return 1
  tar -xzf "$OUT/.$1.tgz" -C "$OUT" && rm -f "$OUT/.$1.tgz"
}
extract() { on "rm -rf ~/RiftEye-embed && mkdir -p ~/RiftEye-embed && tar -xzf rifteye-embed-code.tgz -C ~/RiftEye-embed"; }
REALENV=""; RUNENV=""  # settings for the VM side: the real crops for both runs, the sizes for the real one only
if [ -n "${REAL:-}" ]; then
  [ -f "$REAL" ] || { echo "REAL=$REAL is not a file here"; exit 1; }
  REALENV="REAL=\$HOME/rifteye-real.tgz"
fi
for v in REAL_TEST REAL_REPEAT REAL_SCALES; do eval "x=\${$v:-}"; if [ -n "$x" ]; then REALENV="$REALENV $v=$x"; fi; done
for v in SEEDS EPOCHS CLEAN BATCH; do eval "x=\${$v:-}"; if [ -n "$x" ]; then RUNENV="$RUNENV $v=$x"; fi; done
smoke() { on "cd ~ && $REALENV RIFTEYE_DIR=~/RiftEye-embed SMOKE=1 DEVICE=$1 nice -n 10 bash ~/RiftEye-embed/ml/scripts/m1-embedder.sh"; }

if git -C "$ROOT" rev-parse --verify -q HEAD >/dev/null; then
  git -C "$ROOT" pull -q --ff-only 2>/dev/null || echo "(could not update $ROOT; using it as it is)"
  git -C "$ROOT" archive --format=tar.gz -o "$PKG" HEAD
  REV=$(git -C "$ROOT" log -1 --format='%h %s')
else
  COPYFILE_DISABLE=1 tar -czf "$PKG" -C "$ROOT" --exclude .venv --exclude node_modules --exclude .git .  # no macOS ._ files
  REV="a copy of $ROOT"
fi
echo "$(at) code: $REV"

st=$(vm_state)
if [ "$st" = unknown ]; then echo "cannot find $VM in project $PROJECT, zone $ZONE"; exit 1; fi
if [ "$st" != RUNNING ]; then
  echo "$(at) $VM is $st; starting it"
  while :; do
    if out=$(gcloud compute instances start "$VM" --project "$PROJECT" --zone "$ZONE" 2>&1); then break; fi
    case "$out" in
      *RESOURCE_POOL_EXHAUSTED*|*"enough resources"*|*STOCKOUT*) echo "$(at) no GPU free in $ZONE yet; trying again in 2 minutes"; sleep 120 ;;
      *) echo "$out"; exit 1 ;;
    esac
  done
fi
n=0
until on true >/dev/null 2>&1; do
  n=$((n + 1)); [ "$n" -lt 40 ] || { echo "no SSH after 10 minutes; is the VM still running?"; exit 1; }
  sleep 15
done

emb_now=$(on "systemctl is-active rifteye-m1-embed" 2>/dev/null | tr -d '\r' || true)
if running "$emb_now"; then
  echo "$(at) the embedder is already running on the VM; its code stays as it is"
else
  retry copy "$PKG" "$VM:rifteye-embed-code.tgz" >/dev/null || { echo "could not copy the code to the VM; try again"; exit 1; }
  retry extract || { echo "could not unpack the code on the VM; try again"; exit 1; }
  if [ -n "${REAL:-}" ]; then
    echo "$(at) copying the real crops to the VM ($(du -h "$REAL" | cut -f1))"
    retry copy "$REAL" "$VM:rifteye-real.tgz" >/dev/null || { echo "could not copy $REAL to the VM; try again"; exit 1; }
  fi
fi
rm -f "$PKG"
s=""
for _ in 1 2 3 4; do s=$(query); [ -z "$s" ] || break; sleep 15; done
[ -n "$s" ] || { echo "$(at) could not read the state of the runs on the VM; try again"; exit 1; }
read_state "$s"
echo "$(at) detector: ${det_last:-not started} ($det_state)"
echo "$(at) embedder: ${emb_last:-not started} ($emb_state)"

# A detector run that stopped half way (the VM was stopped) resumes; one that finished stays finished.
# Never while the embedder has the GPU: the detector would find it busy.
if [ -n "$det_last" ] && ! running "$det_state" && [ "$det_ok" != yes ] && ! running "$emb_state"; then
  echo "$(at) the detector run stopped before it finished; starting it again (it resumes)"
  on "bash ~/RiftEye-embed/ml/scripts/m1-vm.sh resume-detector" || echo "$(at) could not restart it"
fi

if running "$emb_state"; then
  echo "$(at) the embedder is running"
elif [ "$emb_ok" = yes ]; then
  echo "$(at) the embedder has already finished"
else
  dev=cpu; [ -n "$det_last" ] || dev=cuda; dev=${SMOKE_DEVICE:-$dev}
  if [ "$dev" = cpu ]; then echo "$(at) short check of the embedder first, on the CPU (about 10 minutes; the detector keeps the GPU)"
  else echo "$(at) short check of the embedder first (a few minutes)"; fi
  rc=0; smoke "$dev" || rc=$?
  if [ "$rc" = 255 ]; then echo "$(at) the connection dropped during the check; once more"; rc=0; smoke "$dev" || rc=$?; fi
  if [ "$rc" = 0 ]; then
    on "$REALENV $RUNENV bash ~/RiftEye-embed/ml/scripts/m1-vm.sh start-embedder" || true
    s=""
    for _ in 1 2 3 4; do s=$(query); [ -z "$s" ] || break; sleep 15; done
    read_state "$s"
  fi
  if running "$emb_state" && [ -z "$det_last" ]; then
    echo "$(at) the embedder is running on the GPU"
  elif running "$emb_state"; then
    echo "$(at) the embedder is queued: it starts on the GPU when the detector finishes"
  elif [ "$rc" = 0 ]; then
    echo "$(at) the embedder passed its check but did not start (see above). The detector is not affected:"
    echo "  this window still waits for it and copies its results."
  else
    echo "$(at) the embedder's check failed (see above), so it was not started. The detector is not affected:"
    echo "  this window still waits for it and copies its results."
  fi
fi

echo "$(at) checking every $((POLL / 60)) minutes; results go to $OUT"
det_got=no; emb_got=no; det_result=no; emb_result=no; fails=0; shown_det=""; shown_emb=""
while :; do
  if [ -n "$det_last" ] && [ "$det_got" = no ] && finished "$det_state" "$det_last"; then
    if fetch detector; then
      det_got=yes; det_result=$det_ok
      echo "$(at) detector $([ "$det_ok" = yes ] && echo finished || echo "stopped without finishing"); its results are in $OUT"
    else
      echo "$(at) could not copy the detector's results yet; trying again later"
    fi
  fi
  if [ -n "$emb_last" ] && [ "$emb_got" = no ] && finished "$emb_state" "$emb_last" && finished "$det_state" "$det_last"; then
    if fetch embedder; then
      emb_got=yes; emb_result=$emb_ok
      echo "$(at) embedder $([ "$emb_ok" = yes ] && echo finished || echo "stopped without finishing"); its results are in $OUT"
    else
      echo "$(at) could not copy the embedder's results yet; trying again later"
    fi
  fi
  # Done when neither runs and whatever ran has its results here.
  if finished "$det_state" "$det_last" && finished "$emb_state" "$emb_last" &&
     { [ -z "$det_last" ] || [ "$det_got" = yes ]; } && { [ -z "$emb_last" ] || [ "$emb_got" = yes ]; }; then
    break
  fi
  sleep "$POLL"
  st=$(vm_state)
  if [ "$st" != RUNNING ]; then echo "$(at) the VM is $st; run this again to pick up from there"; break; fi
  s=$(query)
  if [ -z "$s" ]; then
    fails=$((fails + 1))
    if [ $((fails % 6)) -eq 0 ]; then echo "$(at) no answer from the VM for $((fails * POLL / 60)) minutes; still trying (the runs go on without this window)"; fi
    continue
  fi
  fails=0
  read_state "$s"
  if [ -n "$det_last" ] && [ "$det_last" != "$shown_det" ]; then echo "$(at) detector: $det_last"; shown_det=$det_last; fi
  if [ -n "$emb_last" ] && [ "$emb_last" != "$shown_emb" ]; then echo "$(at) embedder: $emb_last"; shown_emb=$emb_last; fi
done

if [ "$(vm_state)" = RUNNING ]; then
  if [ "$DELETE" = yes ] && { [ -z "$det_last" ] || [ "$det_result" = yes ]; } && [ "$emb_result" = yes ]; then
    gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$ZONE" --quiet
    echo "$(at) deleted $VM"
  else
    gcloud compute instances stop "$VM" --project "$PROJECT" --zone "$ZONE"
    echo "$(at) stopped $VM; its disk keeps everything. When you no longer need it:"
    echo "  gcloud compute instances delete $VM --project $PROJECT --zone $ZONE"
  fi
fi
echo "$(at) results in $OUT (the CSVs are safe to share; the .pth weights are private):"
ls -la "$OUT" 2>/dev/null || true
