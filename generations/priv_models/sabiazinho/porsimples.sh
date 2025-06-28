#!/bin/bash

# chmod +x porsimples.sh
# Usage: ./porsimples.sh <save_interval> <intermediate_save_path> <output_file_path>

# Example:
# ./porsimples.sh 10 "sabiazinho_porsimples_intermediate_r2.json" "sabiazinho_porsimples_final_output_r2.json" 

SAVE_INTERVAL=$1
INTERMEDIATE_SAVE_PATH=$2
OUTPUT_FILE_PATH=$3

python3 porsimples.py $SAVE_INTERVAL $INTERMEDIATE_SAVE_PATH $OUTPUT_FILE_PATH
