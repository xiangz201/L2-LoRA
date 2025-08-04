#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 python ../../src/export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B/commonsense170k/l2-lora/ \
    --template default \
    --finetuning_type lora \
    --export_dir ./llama2/merged_trained_models/cs-l2-lora/ \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
