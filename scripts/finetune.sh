#!/bin/bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29501}
export NNODES=${NNODES:-1}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs_acc/4gpu.yaml}

export LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "

export PROGRAM="\
finetune.py \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v \
    --model_type wan-i2v \
    --training_type lora \
    --rank 256 \
    --lora_alpha 256 \
    --output_dir ./results/EgoX \
    --report_to tensorboard \
    --data_root ./dataset/train \
    --meta_data_file ./dataset/train/meta_with_uid.json \
    --train_resolution 49x448x1232 \
    --train_epochs 150 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_workers 16 \
    --pin_memory True \
    --nccl_timeout 1800 \
    --checkpointing_steps 250 \
    --checkpointing_limit 54 \
    --gen_fps 30 \
    --cos_sim_scaling_factor 1.0 \
"
# --resume_from_checkpoint ./results/EgoX/checkpoint-10000 \

export CMD="$LAUNCHER $PROGRAM"

# Use eval so the composed string is parsed into words/args correctly
eval "$CMD"

echo "END TIME: $(date)"
