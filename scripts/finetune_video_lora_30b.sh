#!/bin/bash
#SBATCH --job-name="Qwen3-VL-30B LoRA Video FT"
#SBATCH --error="/standard/UVA-DSA/Keshara/EgoVLM/models/Qwen-VL-Series-Finetune/scripts/logs/job-%j-qwen30b_video_lora.error"
#SBATCH --output="/standard/UVA-DSA/Keshara/EgoVLM/models/Qwen-VL-Series-Finetune/scripts/logs/job-%j-qwen30b_video_lora.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:8
#SBATCH --constraint=a100_80gb
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"


PROJECT_ROOT="/standard/UVA-DSA/Keshara/EgoVLM/models/Qwen-VL-Series-Finetune"
DATA_ROOT="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego"
LOG_DIR="${PROJECT_ROOT}/scripts/logs"

MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct"
DEEPSPEED_CONFIG="scripts/zero2.json"
OUTPUT_DIR="output/qwen3_vl_30b_lora_ems_video"

GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

VIDEO_TOKEN_BUDGET=64
FPS=1
NUM_EPOCHS=3
SAVE_STEPS=50

module purge
module load miniforge
module load gcc/11.4.0
source /home/cjh9fw/.bashrc
echo "$HOSTNAME"
conda deactivate || true
conda deactivate || true
conda activate qwen_vl

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "Environment Ready: Qwen with GCC $(gcc -dumpversion) on $HOSTNAME"
echo "Using model: ${MODEL_NAME}"
echo "Using DeepSpeed config: ${DEEPSPEED_CONFIG}"
echo "Using ${NUM_DEVICES} GPUs with grad accumulation ${GRAD_ACCUM_STEPS}"

# Starting point for a small (~2k video) EMS dataset:
# - LoRA on the LLM to keep 30B training feasible
# - full vision/projector tuning so the video side can still adapt
# - lower fps/token budget to avoid the cross-entropy OOM seen in prior runs
# If GPU memory is still tight, lower VIDEO_TOKEN_BUDGET further before raising fps.

deepspeed src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_id "${MODEL_NAME}" \
    --data_path "${DATA_ROOT}/qwen_video_train_data.json" \
    --image_folder "${DATA_ROOT}/" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --max_seq_length 4096 \
    --video_max_pixels $((VIDEO_TOKEN_BUDGET * 32 * 32)) \
    --fps "${FPS}" \
    --learning_rate 5e-5 \
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
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 6 \
    --dataloader_num_workers 2
