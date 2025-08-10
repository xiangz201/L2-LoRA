# 🧠 L2-LoRA: Improving Low-Rank Adaptation with Layer-Specific Regularization

This repository provides the implementation of **L2-LoRA** (L2-LoRA: Layer-Specific Regularized Low-Rank Adaptation), which leverages **task-specific knowledge localization results** for parameter-efficient fine-tuning of large language models.

---

## 📁 Repository Structure
├── src/ \
│ ├── peft/ # Modified PEFT implementation (selectively apply LoRA updates to specified layers)  \
│ └── transformers/ # Modified Transformers (LLaMA layer-wise L2 Regularization) \
├── NAIE/ # Layer-wise task-specific Knowledge Localization \
├── lora_multi_gpu/ # L2-LoRA .sh code

### Key Components

- `src/peft`: Implements `peft.tuners.tuners_utils.BaseTuner.inject_adapter` (Lines 344–353) to enable **selective LoRA updates to specified layers**.
- `src/transformers`: Includes modified `transformers.trainer.py` ( Line 3266-3349) for layer-specific L2 regularization.
- `NAIE`: Contains code for **Layer-wise Task-Specific Knowledge Localization** analysis, adapted from [lm-eval v0.4.0](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.0) and [Rank-One Model Editing](https://github.com/kmeng01/rome).

---

## 🚀 Quickstart: Running L2-LoRA

### Step 1: Localize layer-wise task-specific knowledge
Our localization results is based on the [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) framework.
Make sure you have installed [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). Then copy the NAIE scripts into the corresponding locations:

```bash
cp -r ./NAIE/evaluator.py         ./lm-evaluation-harness/lm_eval/
cp -r ./NAIE/huggingface.py       ./lm-evaluation-harness/lm_eval/models/
cp -r ./NAIE/instance.py          ./lm-evaluation-harness/lm_eval/api/
cp -r ./NAIE/model.py             ./lm-evaluation-harness/lm_eval/api/
cp -r ./NAIE/rome_main.py         ./lm-evaluation-harness/lm_eval/api/
```

Get NAIE on Task-Specific Datasets:
```python
CUDA_VISIBLE_DEVICES=0 nohup lm_eval \
  --model hf \
  --model_args pretrained=/huggingface/llama2-7b-hf/,dtype=float16 \
  --tasks arc_challenge \
  --device cuda:0 \
  --batch_size 2 > myout.txt 2>&1 &
```

```bash
python ./NAIE/generate_results.py
```
### Step 2: Fine-tune Using L2-LoRA
Our fine-tuning code is built on top of [LLaMA-Factory v0.7.0](https://github.com/hiyouga/LLaMA-Factory/tree/v0.7.0). L2-LoRA modifications are integrated into src/peft and src/transformers.

To run training:
```bash
cd ./l2_lora/lora_multi_gpu/
bash sing_node_metamathqa_365k_l2_lora.sh
```
### Step3 :Evaluation
Our evaluation is based on the [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) framework.
```bash
cd lora_multi_gpu
bash merge.sh
CUDA_VISIBLE_DEVICES=0 nohup lm_eval \
  --model hf \
  --model_args pretrained=/llama2/merged_trained_models/l2_lora/,dtype=float16 \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size 2 > myout.txt 2>&1 & 
```

---
## Acknowledgements
This repo benefits from \
[PEFT](https://github.com/huggingface/peft) \
[Transformers](https://github.com/huggingface/transformers) \
[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/) \
[llama factory](https://github.com/hiyouga/LLaMA-Factory). \
Thanks for their wonderful works.
