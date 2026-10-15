#!/bin/bash
# Launch script for single-GPU training
# Usage: bash launch_single_gpu.sh

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    src/train.py \
    --model_name state-spaces/mamba-130m-hf \
    --dataset_name SargeDev/jev-distill-corpus-v3 \
    --epochs 2 \
    --micro_batch_size 8 \
    --grad_accum_steps 16 \
    --lr 3e-4 \
    --warmup_ratio 0.05 \
    --lora_r 16 \
    --lora_alpha 32 \
    --save_dir ./checkpoints \
    --wandb_project fork-mamba-chooser
