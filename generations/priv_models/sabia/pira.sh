#!/bin/bash

# chmod +x pira.sh
# Usage: ./pira.sh <save_interval> <intermediate_save_path> <output_file_path>

# Example:
# ./pira.sh 10 "sabia_pira_intermediate_r1.json" "sabia_pira_final_output_r1.json" 

# SAVE_INTERVAL=$1
# INTERMEDIATE_SAVE_PATH=$2
# OUTPUT_FILE_PATH=$3

# python3 pira.py $SAVE_INTERVAL $INTERMEDIATE_SAVE_PATH $OUTPUT_FILE_PATH

python3 pira.py 10 "sabia_pira_intermediate_r1.json" "sabia_pira_final_output_r1.json" 
python3 pira.py 10 "sabia_pira_intermediate_r2.json" "sabia_pira_final_output_r2.json" 
python3 pira.py 10 "sabia_pira_intermediate_r3.json" "sabia_pira_final_output_r3.json" 
