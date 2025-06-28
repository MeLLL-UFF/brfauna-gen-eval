from dotenv import load_dotenv
import openai
from datasets import load_dataset
import os
import pandas as pd
import json
import sys

# Load environment variables
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()

# Parse arguments from the shell script
DATASET_NAME = "recogna-nlp/recognasumm"
SAVE_INTERVAL = int(sys.argv[1])
INTERMEDIATE_SAVE_PATH = sys.argv[2]
OUTPUT_FILE_PATH = sys.argv[3]

# Define functions
def add_noticia_and_reference_columns(example):
    example['noticia'] = example['Noticia']
    example['reference'] = example['Sumario']
    return example

def get_prompt(noticia):
    return f"""Sumarize de forma breve e direta a notícia a seguir.
Notícia: {noticia}
Sumário: """


# Load and filter the dataset
dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

dataset = dataset.map(add_noticia_and_reference_columns)

columns_to_remove = [col for col in dataset['train'].column_names if col not in ['noticia', 'reference']]
dataset = dataset.map(remove_columns=columns_to_remove)

test_set = dataset['test']

# Check if intermediate file exists and resume if needed
instances = []
resume_index = 0

# Ensure the instances list matches the length of the test_set
instances = [{} for _ in range(len(test_set))]

# Generate answers and save results
results = []
for idx, item in enumerate(test_set):
    noticia = item['noticia']
    reference = item['reference']
    prompt = get_prompt(noticia)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=85,
            temperature=0.3,
            top_p=0.95
        )
        generated_text = response.choices[0].message.content
        instances[idx] = {
            "noticia": noticia,
            "reference": reference,
            "generated": generated_text
        }
    except Exception as e:
        print(f"Error at index {idx}: {e}")
        instances[idx] = {
            "noticia": noticia,
            "reference": reference,
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

print(f"Results saved to {OUTPUT_FILE_PATH}")
