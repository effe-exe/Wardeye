# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Federico Vietti and RiftEye contributors
"""The card embedder (M1): DINOv2-S/14 fine-tuned with Sub-center ArcFace on synthetic stream crops.

* `bank`: the training crops, every printing degraded many times through the H.264 stream simulator
  (torch-free, so it runs on every CPU core).
* `model`: the network, the loss, training, packing and the `embedder:<weights>` encoder.
* `evaluate`: fresh synthetic crops of held-out and training sets, frozen backbone against fine-tuned.

    python -m rifteye_ml.embed crops --catalog catalog.jsonl --cache ~/rifteye-data/art --out bank/
    python -m rifteye_ml.embed train --catalog ... --cache ... --bank bank/ --out runs/heldout --train-sets OGN,OGS,SFD
    python -m rifteye_ml.embed evaluate --catalog ... --cache ... --eval-bank bank-eval/ --model runs/heldout/final.pt ...
"""
