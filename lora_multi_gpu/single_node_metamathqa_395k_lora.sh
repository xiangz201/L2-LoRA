#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    --main_process_port=1221 \
    ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /data2/zx/huggingface-llama2/llama2-7b-hf/ \
    --dataset metamathqa_395k \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../../saves/LLaMA2-7B/lora-metamath395k/lora/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 200 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --max_samples 40000 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16 > train_log2.out 2>&1 &
