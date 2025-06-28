from dotenv import load_dotenv
import openai
from datasets import load_dataset
import os
import pandas as pd
import json
import sys

# Load environment variables
load_dotenv()
openai.api_key = os.environ['MARITACA_API_KEY']
client = openai.OpenAI(base_url="https://chat.maritaca.ai/api", api_key=os.environ['MARITACA_API_KEY'])

# Parse arguments from the shell script
DATASET_NAME = "ruanchaves/porsimplessent"
SAVE_INTERVAL = int(sys.argv[1])
INTERMEDIATE_SAVE_PATH = sys.argv[2]
OUTPUT_FILE_PATH = sys.argv[3]

# Define functions
def add_simpler_and_complexer_columns(example):
    if example['label'] == 0:
        example['simpler'] = example['sentence1']
        example['complexer'] = example['sentence2']
    elif example['label'] == 2:
        example['simpler'] = example['sentence2']
        example['complexer'] = example['sentence1']
    return example

def get_prompt(text):
    return f"""Substitua a frase complexa por uma frase simples. Mantenha o mesmo significado, mas torne-a mais simples.
Frase complexa: {text}
Frase Simples:"""

# Load and filter the dataset
dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

# Apply filters and map to add 'simpler' and 'complexer' columns
dataset = (dataset
           .filter(lambda example: example['split'] == 'N')
           .filter(lambda example: example['changed'] == 'S')
           .filter(lambda example: example['level'] != 'NAT->STR')
           .filter(lambda example: example['label'] != 1)
           .map(add_simpler_and_complexer_columns))

columns_to_remove = [col for col in dataset['train'].column_names if col not in ['simpler', 'complexer']]
dataset = dataset.map(remove_columns=columns_to_remove)

test_set = dataset['test']

# Check if intermediate file exists and resume if needed
instances = []
resume_index = 0

# Ensure the instances list matches the length of the test_set
instances = [{} for _ in range(len(test_set))]

# Process the test_set for translation
for idx in range(resume_index, len(test_set)):
    row = test_set[idx]
    complexer = row['complexer']
    simpler = row['simpler']

    prompt = get_prompt(complexer)

    try:
        response = client.chat.completions.create(
            model="sabia-3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.8,
            top_p=0.95
        )
        generated_text = response.choices[0].message.content
        instances[idx] = {
            "complexer": complexer,
            "reference": simpler,
            "generated": generated_text
        }
    except Exception as e:
        print(f"Error at index {idx}: {e}")
        instances[idx] = {
            "complexer": complexer,
            "reference": simpler,
            "generated": "Error: API Failed"
        }

    # Save intermediate results periodically
    if (idx + 1) % SAVE_INTERVAL == 0:
        with open(INTERMEDIATE_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(instances[:idx + 1], f, ensure_ascii=False, indent=4)
        print(f"Saved intermediate progress at row {idx + 1}")

# Save the final results
with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(instances, f, ensure_ascii=False, indent=4)

print(f"Program completed. The updated JSON is saved at {OUTPUT_FILE_PATH}.")
