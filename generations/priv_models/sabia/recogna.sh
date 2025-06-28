#!/bin/bash

# chmod +x recogna.sh
# Usage: ./recogna.sh <save_interval> <intermediate_save_path> <output_file_path>

# Example:
# ./recogna.sh 10 "sabia_recogna_intermediate.json" "sabia_recogna_final_output.json" 

SAVE_INTERVAL=$1
INTERMEDIATE_SAVE_PATH=$2
OUTPUT_FILE_PATH=$3

python3 recogna.py $SAVE_INTERVAL $INTERMEDIATE_SAVE_PATH $OUTPUT_FILE_PATH
