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

print(openai.api_key)

# Parse arguments from the shell script
DATASET_NAME = "paulopirozelli/pira"
SAVE_INTERVAL = int(sys.argv[1])
INTERMEDIATE_SAVE_PATH = sys.argv[2]
OUTPUT_FILE_PATH = sys.argv[3]

# Define functions
def add_question_and_reference_columns(example):
    example['question'] = example['question_pt_origin']
    example['reference'] = example['answer_pt_origin']
    return example

def get_prompt(question):
    return f"""Responda à seguinte pergunta com base em seu conhecimento geral sobre oceanos, a costa brasileira e as mudanças climáticas.
Seja objetivo.
Pergunta: {question}
Resposta: """

# Load and filter the dataset
dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

dataset = dataset.map(add_question_and_reference_columns)

columns_to_remove = [col for col in dataset['train'].column_names if col not in ['question', 'reference']]
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
    question = item['question']
    reference = item['reference']
    prompt = get_prompt(question)

    try:
        response = client.chat.completions.create(
            model="sabiazinho-3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
            top_p=0.95
        )
        generated_text = response.choices[0].message.content
        instances[idx] = {
            "question": question,
            "reference": reference,
            "generated": generated_text
        }
    except Exception as e:
        print(f"Error at index {idx}: {e}")
        instances[idx] = {
            "question": question,
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
