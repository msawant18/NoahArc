#!/bin/bash
python -m src.models.infer --ckpt artifacts/checkpoints/best.pt --in_dir data/processed --out_dir data/outputs --threshold 0.5

