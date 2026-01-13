#!/bin/bash

MODEL_PATH="outputs/segzero_plus_7b/checkpoint-300"
SAM_PATH="pretrained_models/sam2.1_hiera_large.pt"
DATA_PATH="data/ReasonSeg/test.json"
OUTPUT_PATH="results/reasonseg_test_results.json"

python evaluation_scripts/eval_segzero_plus.py \
    --model_path $MODEL_PATH \
    --sam_model_path $SAM_PATH \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH