#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4 nohup accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    --main_process_port=1231 \
    ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /data2/zx/huggingface-llama2/llama2-7b-hf/ \
    --dataset codefeedback \
    --dataset_dir ../../data \
    --template alpaca \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../../saves/LLaMA2-7B/lora-codefeedback/0-15/ \
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
    --save_steps 300 \
    --eval_steps 300 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --max_samples 110000 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16 > train_log3.out 2>&1 &
