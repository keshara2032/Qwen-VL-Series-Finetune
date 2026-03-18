#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Qwen-VL Video Finetuning (FULL)"
#SBATCH --error="./logs/job-%j-qwen_video_finetuner.error"
#SBATCH --output="./logs/job-%j-qwen_video_finetuner.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80gb
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"


module purge &&
module load miniforge  &&
module load gcc/11.4.0 &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda deactivate &&
conda deactivate &&
conda activate qwen_vl &&

export PYTHONNOUSERSITE=1

echo "Environment Ready: Qwen with GCC $(gcc -dumpversion) on $HOSTNAME"



# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen3.5-4B"

MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
# MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct"

export PYTHONPATH=/standard/UVA-DSA/Keshara/EgoVLM/models/Qwen-VL-Series-Finetune/src:$PYTHONPATH

TRAIN_DATA_PATH="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/qwen_video_train_data.json"
VAL_DATA_PATH="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/qwen_video_val_data.json"
DATA_ROOT="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/"

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
VIDEO_TOKEN_BUDGET=128
SAVE_STEPS=5
EVAL_STEPS=$SAVE_STEPS

# If your dataset is mixed with images and videos, you need to use zero2.
# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)
# If you switch MODEL_NAME to a Qwen3.5 model, set `--disable_flash_attn2 True`.
# Flash Attention 2 raised CUDA errors for the Qwen3.5 series in local tests, so SDPA is the stable path for now.


deepspeed /standard/UVA-DSA/Keshara/EgoVLM/models/Qwen-VL-Series-Finetune/src/train/train_sft.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$TRAIN_DATA_PATH" \
    --eval_path "$VAL_DATA_PATH" \
    --image_folder "$DATA_ROOT" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/test_train_30B \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((VIDEO_TOKEN_BUDGET * 32 * 32)) \
    --fps 10 \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --eval_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --per_device_eval_batch_size 1 \
    --generation_max_new_tokens 256 \
    --prediction_loss_only False \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_keystep_accuracy" \
    --greater_is_better True \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 10 \
    --dataloader_num_workers 1
