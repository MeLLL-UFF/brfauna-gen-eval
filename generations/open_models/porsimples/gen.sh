#!/bin/bash
# chmod +x gen.sh

# Define arrays for models, tokenizer IDs, and output file names
declare -a model_ids=("TucanoBR/Tucano-2b4-Instruct" "recogna-nlp/Bode-3.1-8B-Instruct-full" "botbot-ai/CabraLlama3-8b" "lucianosb/boto-9B-it" "wandgibaut/periquito-3B")

declare -a tokenizer_ids=("TucanoBR/Tucano-2b4-Instruct" "recogna-nlp/Bode-3.1-8B-Instruct-full" "botbot-ai/CabraLlama3-8b" "lucianosb/boto-9B-it" "wandgibaut/periquito-3B")

declare -a output_files=("tucano_porsimples_r3.json" "bode_318B_porsimples_r3.json" "cabra_llama8b_porsimples_r3.json" "boto_gemma_porsimples_r3.json" "piriquito_ollama_porsimples_r3.json")


# Hugging Face API token
hf_token="" # Replace with your token

# GPU device index
gpu_device=0

# Loop through arrays and run the Python script with each configuration
for i in "${!model_ids[@]}"; do
    model_id="${model_ids[$i]}"
    tokenizer_id="${tokenizer_ids[$i]}"
    output_file="${output_files[$i]}"
    
    log_file="${output_file%.json}_log.txt"

    echo "Running inference with model: ${model_id}, tokenizer: ${tokenizer_id}, output file: ${output_file}"

    # Run the Python script and save logs
    CUDA_VISIBLE_DEVICES=0 python gen_porsimples_chat.py \
        --model_id "$model_id" \
        --tokenizer_id "$tokenizer_id" \
        --output_file "$output_file" \
        --hf_token "$hf_token" \
        --gpu "$gpu_device" > "$log_file" 2>&1

    echo "Finished run for ${model_id}"
done

echo "All runs completed."
