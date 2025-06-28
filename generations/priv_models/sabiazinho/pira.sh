#!/bin/bash

# chmod +x pira.sh
# Usage: ./pira.sh <save_interval> <intermediate_save_path> <output_file_path>

# Example:
# ./pira.sh 10 "sabiazinho_pira_intermediate_r1.json" "sabiazinho_pira_final_output_r1.json" 

SAVE_INTERVAL=$1
INTERMEDIATE_SAVE_PATH=$2
OUTPUT_FILE_PATH=$3

python3 pira.py $SAVE_INTERVAL $INTERMEDIATE_SAVE_PATH $OUTPUT_FILE_PATH
