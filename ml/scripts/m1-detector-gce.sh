#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
#
# Run m1-detector.sh on an existing Google Cloud GPU VM, from your own machine (needs gcloud).
#
#   bash m1-detector-gce.sh VM PROJECT ZONE
#
# GPUs are often all taken in a zone, so this retries the start every 2 minutes until one is free.
# Then it runs the short check (SMOKE=1) while you watch, and starts the real run in the background
# with STOP_WHEN_DONE=1: the VM powers itself off when the run ends, finished or failed.
# Keep this machine awake while it waits (macOS: caffeinate -i bash m1-detector-gce.sh ...).
set -eu
VM=${1:?usage: m1-detector-gce.sh VM PROJECT ZONE}; PROJECT=${2:?project}; ZONE=${3:?zone}
JOB=https://raw.githubusercontent.com/effe-exe/RiftEye/main/ml/scripts/m1-detector.sh
at() { date +%H:%M; }

while :; do
  if out=$(gcloud compute instances start "$VM" --project "$PROJECT" --zone "$ZONE" 2>&1); then break; fi
  case "$out" in
    *RESOURCE_POOL_EXHAUSTED*|*"enough resources"*|*STOCKOUT*) echo "$(at) no GPU free in $ZONE yet; trying again in 2 minutes"; sleep 120 ;;
    *) echo "$out"; exit 1 ;;
  esac
done
echo "$(at) started $VM; waiting for SSH"
n=0
until gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap -- true >/dev/null 2>&1; do
  n=$((n + 1)); [ "$n" -lt 40 ] || { echo "no SSH after 10 minutes; is the VM still running?"; exit 1; }
  sleep 15
done
echo "$(at) short check first"
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap -- \
  "curl -fsSL $JOB -o m1-detector.sh && SMOKE=1 bash m1-detector.sh" || {
  echo "$(at) the short check failed (see above), so the real run was not started. The VM is still on; stop it with:"
  echo "  gcloud compute instances stop $VM --project $PROJECT --zone $ZONE"
  exit 1
}
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap -- \
  "STOP_WHEN_DONE=1 nohup setsid bash m1-detector.sh >/dev/null 2>&1 </dev/null &"
echo "$(at) the real run is going on $VM, and the VM powers off when it ends. Progress:"
echo "  gcloud compute ssh $VM --project $PROJECT --zone $ZONE --tunnel-through-iap -- \"grep '^==' ~/rifteye-m1/logs/run.log | tail -n 2\""
