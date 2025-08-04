#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 nohup accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    --main_process_port=1223 \
    ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /data2/zx/huggingface-llama2/llama2-7b-hf/ \
    --dataset commonsense_170k \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../../saves/LLaMA2-7B/commonsense170k/lora/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 170000 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16 > train_log.out 2>&1 &
