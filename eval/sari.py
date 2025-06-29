import os
import json
from easse.sari import corpus_sari

# Define your runs & models
RUNS = ["r1", "r2", "r3"]
MODELS = [
    "bode_318B_porsimples",
    "boto_gemma_porsimples",
    "cabra_llama8b_porsimples",
    "piriquito_ollama_porsimples",
    "tucano_porsimples",
    "gptmini_porsimples_final_output",
    "gpt_porsimples_final_output",
    "sabia_porsimples_final_output",
    "sabiazinho_porsimples_final_output"
]

BASE_INPUT_DIR = ""
BASE_OUTPUT_DIR = ""

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def evaluate_sari(json_path):
    """Compute sentence-level and corpus-level SARI."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_complex = [d["complexer"] for d in data]
    all_generated = [d["generated"] for d in data]
    all_refs = [[d["reference"]] for d in data]  

    per_sample_list = []
    for i, (complex_sent, generated_sent, ref_list) in enumerate(zip(all_complex, all_generated, all_refs)):
        sari_score = corpus_sari([complex_sent], [generated_sent], [ref_list])
        per_sample_list.append({
            "index": i,
            "complexer": complex_sent,
            "generated": generated_sent,
            "reference": ref_list[0],
            "SARI": float(sari_score)
        })

    corpus_sari_score = corpus_sari(all_complex, all_generated, list(map(list, zip(*all_refs))))

    return corpus_sari_score, per_sample_list


# Process models and runs
for run in RUNS:
    run_output_dir = os.path.join(BASE_OUTPUT_DIR, run)
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"\n=== Processing {run} ===")
    for model in MODELS:
        json_path = os.path.join(BASE_INPUT_DIR, run, f"{model}_{run}.json")
        if not os.path.isfile(json_path):
            print(f"  [WARNING] Missing file for model '{model}', run '{run}': {json_path}")
            continue

        print(f"  -> Evaluating SARI: {json_path}")
        corpus_sari_score, per_sample_list = evaluate_sari(json_path)

        # Prepare JSON output
        output_data = {
            "corpus_sari": corpus_sari_score,
            "entries": per_sample_list
        }
        output_filename = f"{model}_{run}_sari.json"
        output_path = os.path.join(run_output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(output_data, out_f, ensure_ascii=False, indent=2)

        print(f"     -> Saved to: {output_path}")

print("\nAll runs processed. SARI results saved.")
