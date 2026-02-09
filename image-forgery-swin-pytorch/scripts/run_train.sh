#!/bin/bash

# Activate the virtual environment if needed
# source path_to_your_virtualenv/bin/activate

# Set the CUDA device to use (if you have multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Run the training script
python src/train.py --config configs/default.yaml --data_dir data --save_dir models --num_epochs 30 --batch_size 32 --learning_rate 3e-4

# Optionally, you can add logging or output redirection
# python src/train.py --config configs/default.yaml --data_dir data --save_dir models --num_epochs 30 --batch_size 32 --learning_rate 3e-4 > training_log.txt 2>&1

echo "Training completed!"