#!/bin/bash

# Model Configuration
BASE_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
#LORA_PATH="saves/qwen2_5vl-3b/lora/sft"
LORA_PATH="/data/shared/jjseol/LLaMA-Factory/saves/qwen2_5vl-3b/lora/train_lora_32_5ep_24b/checkpoint-5000"
HOST="211.47.48.147"
PORT="8000"
GPU_MEMORY_UTILIZATION="0.5"
MAX_MODEL_LEN="16384"
SERVED_MODEL_NAME="webrl-llama3-8b"
TENSOR_PARALLEL_SIZE="1"
DTYPE="auto"
PROCESS_TITLE="MIRIDIH-JYJANG"
GPU_ID="7"  # Updated to match your CUDA_VISIBLE_DEVICES

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_USE_MODELSCOPE="False"

# Activate virtual environment
source /workspace/Structured-Content/vllm/bin/activate

# Display config
echo "Starting VLLM with LoRA using LLM class..."
echo "Base Model: $BASE_MODEL_PATH"
echo "LoRA Path: $LORA_PATH"
echo "GPU: $GPU_ID"
echo "Process Title: $PROCESS_TITLE"

# Check if LoRA path exists
if [ ! -d "$LORA_PATH" ]; then
    echo "Warning: LoRA path $LORA_PATH not found!"
    echo "Please verify the path or use base model only."
fi

# Export environment variables for Python script
export BASE_MODEL_PATH="$BASE_MODEL_PATH"
export LORA_PATH="$LORA_PATH"
export GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
export MAX_MODEL_LEN="$MAX_MODEL_LEN"
export TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE"
export DTYPE="$DTYPE"

# Launch the Python script
python vllm_inference.py
