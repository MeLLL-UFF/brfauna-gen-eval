import transformers
import torch
from huggingface_hub import login
import gc
from datasets import load_dataset
import json
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import eco2ai

# Disable NCCL features for compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Summarize texts using a specified model and tokenizer.")
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for summarization.')
    parser.add_argument('--tokenizer_id', type=str, required=True, help='Tokenizer ID to use.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path to save the summaries.')
    parser.add_argument('--hf_token', type=str, required=False, help='Hugging Face API token for authentication.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for summarization.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use (e.g., 0 or 1).')

    args = parser.parse_args()

    # Set the environment variable to specify which GPU to use
    torch.cuda.set_device(args.gpu)
    
    def add_noticia_and_reference_columns(example):
        example['noticia'] = example['Noticia']
        example['reference'] = example['Sumario']
        return example

    # Hugging Face login
    if args.hf_token:
        login(args.hf_token)

    tracker_file = args.output_file.replace("json", "csv")
    # CO2 tracker setup
    tracker = eco2ai.Tracker(
        project_name=f'{args.model_id} inference', 
        experiment_description="Inference with multiple models in RecognaSumm",
        file_name=tracker_file
    )

    # Load dataset
    dataset = load_dataset("recogna-nlp/recognasumm", trust_remote_code=True)
    
    dataset = dataset.map(add_noticia_and_reference_columns)

    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['noticia', 'reference']]
    dataset = dataset.map(remove_columns=columns_to_remove)
        
    test_set = dataset['test'].select(range(2000))

    if 'cabrita' in args.model_id:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
        
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).to(f"cuda")

    pipeline = transformers.pipeline(
        task="text-generation",
        trust_remote_code=True,
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    
    
    def get_user_prompt(text):
        return f'''Sumarize de forma breve e direta a notícia a seguir.
    Notícia: {text}
    Sumário: '''

    final_data = []
    
    if 'periquito' in args.model_id:
        
        tracker.start()
        with torch.no_grad():  # Optimize by disabling gradient calculations
            for i in range(len(test_set)):
                original_text = test_set['noticia'][i]
                prompt = get_user_prompt(original_text)
                
                messages = [prompt]
                
                outputs = pipeline(
                    messages,
                    max_new_tokens=85,
                    temperature=0.3,
                    top_p=0.95,
                    repetition_penalty=2.5,
                )
                
                generated_text= outputs[0][0]["generated_text"][len(prompt):]

                # Collect the result
                article_data = {
                    'id': i,
                    'original': original_text,
                    'reference': test_set['reference'][i],
                    'generated': generated_text
                }
                final_data.append(article_data)

                # Clean up to free memory
                torch.cuda.empty_cache()
                gc.collect()
                
        tracker.stop()
    else:
        tracker.start()
        with torch.no_grad():  # Optimize by disabling gradient calculations
            for i in range(len(test_set)):
                original_text = test_set['noticia'][i]
                prompt = get_user_prompt(original_text)

                messages = [{
                        "role": "user", "content": prompt
                    }]
                
                outputs = pipeline(
                    messages,
                    max_new_tokens=85,
                    temperature=0.3,
                    top_p=0.95,
                    repetition_penalty=2.5,
                )

                # Decode the generated output
                generated_text = outputs[0]["generated_text"][1]['content']

                # Collect the result
                article_data = {
                    'id': i,
                    'original': original_text,
                    'reference': test_set['reference'][i],
                    'generated': generated_text
                }
                final_data.append(article_data)

                # Clean up to free memory
                torch.cuda.empty_cache()
                gc.collect()
                
        tracker.stop()

    # Save the summarized data to the specified output file
    with open(args.output_file, 'w') as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)

    print(f"Answers have been saved to '{args.output_file}'.")

if __name__ == '__main__':
    main()
