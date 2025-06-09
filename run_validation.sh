#!/bin/bash

# End-to-end validation script for FineWeb fine-tuning

echo "Starting FineWeb fine-tuning validation..."

# Configuration
MODEL_NAME="Qwen/Qwen3-32B"  # More stable base model for validation
OUTPUT_DIR="./fineweb_validation"
MAX_STEPS=50  # Reduced steps to prevent overfitting
BATCH_SIZE=1
LEARNING_RATE=1e-6  # Smaller learning rate for stability
ZO_EPS=1e-3

# Create output directory
mkdir -p $OUTPUT_DIR

# Run validation
echo "Running validation with the following configuration:"
echo "Model: $MODEL_NAME"
echo "Max steps: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "ZO epsilon: $ZO_EPS"

python validate_fineweb.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --zo_eps $ZO_EPS

echo "Validation complete!"

# Test with different configurations
# echo -e "\n=== Testing with conservative hyperparameters ==="
# echo "Running validation with GPT-2 and conservative settings..."
# python validate_fineweb.py \
#     --model_name "gpt2" \
#     --output_dir "${OUTPUT_DIR}_gpt2_conservative" \
#     --max_steps 30 \
#     --batch_size 4 \
#     --learning_rate 5e-7 \
#     --zo_eps 1e-3

# echo -e "\n=== Testing with smaller model ==="
# echo "Running validation with DistilGPT-2..."
# python validate_fineweb.py \
#     --model_name "distilgpt2" \
#     --output_dir "${OUTPUT_DIR}_distilgpt2" \
#     --max_steps 50 \
#     --batch_size 8 \
#     --learning_rate 1e-6 \
#     --zo_eps 1e-3
