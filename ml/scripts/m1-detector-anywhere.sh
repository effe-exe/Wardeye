#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
#
# Run the M1 detector job on a temporary Google Cloud GPU VM, in whichever zone has a GPU free.
#
#   bash ml/scripts/m1-detector-anywhere.sh PROJECT
#   JOB=embedder REAL=real.tgz bash ml/scripts/m1-detector-anywhere.sh PROJECT   # the embedder instead (v1 with REAL)
#
# Tries T4 zones around the world, then L4 zones, and creates the VM in the first that has a GPU
# free and quota for it (a Deep Learning VM image, NVIDIA driver included). Then it runs
# m1-detector-gce.sh on it (short check, real run, results copied to ~/rifteye-m1-results) and
# deletes the VM. Nothing else in the project is touched. Keep this machine awake (caffeinate -i).
# JOB=embedder: L4 zones only (it trains in bfloat16), then m1-embedder-gce.sh --delete, which deletes the VM
# once the embedder's results are here and keeps it stopped otherwise.
set -eu
PROJECT=${1:?usage: m1-detector-anywhere.sh PROJECT}
VM=${VM_NAME:-rifteye-m1-tmp}
HERE=$(cd "$(dirname "$0")" && pwd)
JOB=${JOB:-detector}
case "$JOB" in detector|embedder) ;; *) echo "JOB is detector or embedder"; exit 1 ;; esac
if [ "$JOB" = embedder ] && [ -n "${REAL:-}" ] && [ ! -f "$REAL" ]; then echo "REAL=$REAL is not a file here"; exit 1; fi
at() { date +%H:%M; }

T4_ZONES="us-central1-a us-central1-b us-central1-c us-central1-f us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c
us-west1-a us-west1-b us-west2-b us-west2-c us-west4-a us-west4-b northamerica-northeast1-c southamerica-east1-c
europe-west1-c europe-west1-d europe-west2-a europe-west2-b europe-west3-b europe-west4-a europe-west4-b europe-west4-c
asia-east1-a asia-east1-c asia-northeast1-a asia-northeast1-c asia-south1-a asia-south1-b asia-southeast1-a asia-southeast1-b
asia-southeast1-c australia-southeast1-a"
L4_ZONES="us-central1-a us-central1-b us-central1-c us-east1-b us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b
us-west1-c europe-west1-b europe-west1-c europe-west4-a europe-west4-b europe-west4-c asia-southeast1-a asia-southeast1-b"

IMAGE=$(gcloud compute images list --project deeplearning-platform-release --no-standard-images \
  --filter="family~^pytorch-.*-ubuntu-2204-nvidia" --sort-by=~creationTimestamp --limit=1 --format="value(name)")
[ -n "$IMAGE" ] || { echo "could not find a Deep Learning VM image"; exit 1; }
echo "$(at) image: $IMAGE"

try() {  # zone machine-type accelerator (empty for machine types with the GPU built in)
  local out  # no arrays: macOS bash 3.2 treats an empty one as unset under set -u
  if out=$(gcloud compute instances create "$VM" --project "$PROJECT" --zone "$1" --machine-type "$2" ${3:+--accelerator "type=$3,count=1"} \
      --maintenance-policy TERMINATE --image "$IMAGE" --image-project deeplearning-platform-release \
      --boot-disk-size 150GB --boot-disk-type pd-balanced --metadata install-nvidia-driver=True 2>&1); then
    return 0
  fi
  echo "  $1: $(echo "$out" | grep -o -m1 -E 'ZONE_RESOURCE_POOL_EXHAUSTED|QUOTA_EXCEEDED|Quota [^.]*|does not have enough resources|not found|not supported|[A-Z_]{8,}' || echo "$out" | tail -n 1)"
  return 1
}

ZONE=""
while [ -z "$ZONE" ]; do
  if [ "$JOB" = detector ]; then for z in $T4_ZONES; do try "$z" n1-standard-8 nvidia-tesla-t4 && { ZONE=$z; break; }; done; fi
  if [ -z "$ZONE" ]; then for z in $L4_ZONES; do try "$z" g2-standard-8 "" && { ZONE=$z; break; }; done; fi
  if [ -z "$ZONE" ]; then echo "$(at) no zone had a GPU free (or quota); trying them all again in 5 minutes"; sleep 300; fi
done
echo "$(at) created $VM in $ZONE"

if [ "$JOB" = embedder ]; then
  if bash "$HERE/m1-embedder-gce.sh" "$VM" "$PROJECT" "$ZONE" --delete; then exit 0; fi
  echo "$(at) the run did not finish cleanly; $VM is kept so nothing is lost. To pick up where it was:"
  echo "  REAL=${REAL:-} bash $HERE/m1-embedder-gce.sh $VM $PROJECT $ZONE --delete"
  echo "or, when done with it:  gcloud compute instances delete $VM --project $PROJECT --zone $ZONE"
  exit 1
fi

if bash "$HERE/m1-detector-gce.sh" "$VM" "$PROJECT" "$ZONE"; then
  gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$ZONE" --quiet
  echo "$(at) deleted $VM. Results are in ~/rifteye-m1-results"
else
  echo "$(at) the run did not finish cleanly; $VM is kept so nothing is lost. When done with it:"
  echo "  gcloud compute instances delete $VM --project $PROJECT --zone $ZONE"
  exit 1
fi
