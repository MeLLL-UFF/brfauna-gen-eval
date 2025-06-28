#!/bin/bash

# chmod +x porsimples.sh
# Usage: ./porsimples.sh <save_interval> <intermediate_save_path> <output_file_path>

# Example:
# ./porsimples.sh 10 "sabia_porsimples_intermediate_r1.json" "sabia_porsimples_final_output_r1.json" 

# SAVE_INTERVAL=$1
# INTERMEDIATE_SAVE_PATH=$2
# OUTPUT_FILE_PATH=$3

# python3 porsimples.py $SAVE_INTERVAL $INTERMEDIATE_SAVE_PATH $OUTPUT_FILE_PATH

python3 porsimples.py 10 "sabia_porsimples_intermediate_r1.json" "sabia_porsimples_final_output_r1.json" 
python3 porsimples.py 10 "sabia_porsimples_intermediate_r2.json" "sabia_porsimples_final_output_r2.json" 
python3 porsimples.py 10 "sabia_porsimples_intermediate_r3.json" "sabia_porsimples_final_output_r3.json" 
