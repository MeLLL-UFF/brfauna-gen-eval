import os
import json
from statistics import mean
import evaluate
from scipy.stats import ttest_rel
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

# Base path where your JSON files are located
BASE_INPUT_DIR = ""

# Where to store the evaluation JSON output (per run)
BASE_OUTPUT_DIR = ""

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Define metric sets
# Per-sample metrics
SAMPLE_METRICS = [
    "BERTScore_precision",
    "BERTScore_recall",
    "BERTScore_f1",
    "METEOR",
    "BLEU",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "ROUGE-Lsum"
]

CORPUS_METRIC = "SARI"

data_store = {
    model: {run: None for run in RUNS}
    for model in MODELS
}


bertscore_metric = evaluate.load("bertscore")
meteor_metric    = evaluate.load("meteor")
bleu_metric      = evaluate.load("bleu")
rouge_metric     = evaluate.load("rouge")

def evaluate_file(json_path):
    """
    Compute per-sample metrics:
      BERTScore, METEOR, BLEU, ROUGE
    And corpus-level SARI.
    Return:
        overall_metrics (dict): aggregated for entire corpus (inc. SARI)
        per_sample_list (list): one entry per sample, with texts + sample-level metrics
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract texts
    all_complex   = [d["complexer"] for d in data]
    all_generated = [d["generated"] for d in data]
    all_refs      = [d["reference"] for d in data]

    # BERTScore in batch
    bert_results = bertscore_metric.compute(
        predictions=all_generated,
        references=all_refs,
        lang="pt" 
    )

    meteor_scores = []
    bleu_scores   = []
    rouge_scores  = []

    for gen_text, ref_text in zip(all_generated, all_refs):
        # METEOR
        meteor_res = meteor_metric.compute(
            predictions=[gen_text],
            references=[ref_text]
        )
        meteor_scores.append(meteor_res["meteor"])

        # BLEU
        bleu_res = bleu_metric.compute(
            predictions=[gen_text],
            references=[[ref_text]]
        )
        bleu_scores.append(bleu_res["bleu"])

        # ROUGE
        rouge_res = rouge_metric.compute(
            predictions=[gen_text],
            references=[ref_text]
        )
        # => dict: "rouge1","rouge2","rougeL","rougeLsum"
        rouge_scores.append(rouge_res)

    # Build per-sample list
    per_sample_list = []
    for i, entry in enumerate(data):
        sample_metrics = {
            "BERTScore_precision": float(bert_results["precision"][i]),
            "BERTScore_recall":    float(bert_results["recall"][i]),
            "BERTScore_f1":        float(bert_results["f1"][i]),
            "METEOR":              float(meteor_scores[i]),
            "BLEU":                float(bleu_scores[i]),
            "ROUGE-1":             float(rouge_scores[i]["rouge1"]),
            "ROUGE-2":             float(rouge_scores[i]["rouge2"]),
            "ROUGE-L":             float(rouge_scores[i]["rougeL"]),
            "ROUGE-Lsum":          float(rouge_scores[i]["rougeLsum"])
        }
        sample_info = {
            "index": i,
            "complexer": entry["complexer"],
            "generated": entry["generated"],
            "reference": entry["reference"],
            "metrics":   sample_metrics
        }
        per_sample_list.append(sample_info)

    N = len(data)
    overall_metrics = {}
    if N > 0:
        overall_metrics["BERTScore_precision"] = float(sum(bert_results["precision"]) / N)
        overall_metrics["BERTScore_recall"]    = float(sum(bert_results["recall"]) / N)
        overall_metrics["BERTScore_f1"]        = float(sum(bert_results["f1"]) / N)
        overall_metrics["METEOR"]             = float(sum(meteor_scores) / N)
        overall_metrics["BLEU"]               = float(sum(bleu_scores) / N)
        overall_metrics["ROUGE-1"]            = float(sum(r["rouge1"]    for r in rouge_scores) / N)
        overall_metrics["ROUGE-2"]            = float(sum(r["rouge2"]    for r in rouge_scores) / N)
        overall_metrics["ROUGE-L"]            = float(sum(r["rougeL"]    for r in rouge_scores) / N)
        overall_metrics["ROUGE-Lsum"]         = float(sum(r["rougeLsum"] for r in rouge_scores) / N)


    references_sari = [[r] for r in all_refs]
    references_sari = list(map(list, zip(*references_sari))) 
    sari_score = corpus_sari(all_complex, all_generated, references_sari)
    overall_metrics["SARI"] = float(sari_score)

    total_len = sum(len(g.split()) for g in all_generated)
    overall_metrics["AvgGeneratedLength"] = float(total_len / N if N else 0.0)

    return overall_metrics, per_sample_list


for run in RUNS:
    run_output_dir = os.path.join(BASE_OUTPUT_DIR, run)
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"\n=== Processing {run} ===")
    for model in MODELS:
        json_path = os.path.join(BASE_INPUT_DIR, run, f"{model}_{run}.json")
        if not os.path.isfile(json_path):
            print(f"  [WARNING] Missing file for model '{model}', run '{run}': {json_path}")
            continue

        print(f"  -> Evaluate: {json_path}")
        overall_metrics, per_sample_list = evaluate_file(json_path)


        data_store[model][run] = {
            "per_sample": per_sample_list,
            "corpus_sari": overall_metrics["SARI"]  
        }


        output_data = {
            "overall_metrics": overall_metrics,
            "entries": per_sample_list
        }
        output_filename = f"{model}_{run}_evaluate_ALLMETRICS.json"
        output_path = os.path.join(run_output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(output_data, out_f, ensure_ascii=False, indent=2)

        print(f"     -> Saved to: {output_path}")


summary_txt_path = os.path.join(BASE_OUTPUT_DIR, "summary_results.txt")


SAMPLE_METRICS_TO_REPORT = SAMPLE_METRICS  

with open(summary_txt_path, "w", encoding="utf-8") as f_out:
    f_out.write("# Summary of Results (Averaged Across Runs)\n\n")

    average_results_across_runs = {}

    for model in MODELS:

        run_averages = []
        run_sari_scores = []
        for run in RUNS:
            data_this_run = data_store[model][run]
            if data_this_run is None:
                continue
            per_sample = data_this_run["per_sample"]  
            if len(per_sample) > 0:
                run_avg = {}
                for met in SAMPLE_METRICS_TO_REPORT:
                    vals = [sample["metrics"][met] for sample in per_sample]
                    run_avg[met] = mean(vals)
                run_averages.append(run_avg)

            run_sari_scores.append(data_this_run["corpus_sari"])

        if not run_averages:  
            continue

        final_avg = {}
        for met in SAMPLE_METRICS_TO_REPORT:
            vals = [ra[met] for ra in run_averages if met in ra]
            final_avg[met] = sum(vals) / len(vals) if len(vals) > 0 else 0.0

        if len(run_sari_scores) > 0:
            final_avg["SARI"] = sum(run_sari_scores) / len(run_sari_scores)
        else:
            final_avg["SARI"] = 0.0

        average_results_across_runs[model] = final_avg

    for model, avg_dict in average_results_across_runs.items():
        f_out.write(f"## Model: {model}\n")
        for met in SAMPLE_METRICS_TO_REPORT:
            val = avg_dict[met]
            f_out.write(f"  {met}: {val:.4f}\n")
        # also print corpus-level SARI
        f_out.write(f"  SARI: {avg_dict['SARI']:.4f}\n")
        f_out.write("\n")

    f_out.write("# Statistical Significance (Paired t-test)\n")
    f_out.write("We do pairwise comparisons among models on each metric.\n\n")

    valid_models = sorted(average_results_across_runs.keys())

    for metric in SAMPLE_METRICS_TO_REPORT:
        f_out.write(f"## Metric: {metric}\n")
        for i in range(len(valid_models)):
            for j in range(i+1, len(valid_models)):
                m1 = valid_models[i]
                m2 = valid_models[j]

                vals_m1 = []
                vals_m2 = []
                for run in RUNS:
                    info_m1 = data_store[m1][run]
                    info_m2 = data_store[m2][run]
                    if info_m1 is None or info_m2 is None:
                        continue
                    per_samp_m1 = info_m1["per_sample"]
                    per_samp_m2 = info_m2["per_sample"]
                    if len(per_samp_m1) != len(per_samp_m2):
                        continue
                    for k in range(len(per_samp_m1)):
                        vals_m1.append(per_samp_m1[k]["metrics"][metric])
                        vals_m2.append(per_samp_m2[k]["metrics"][metric])

                if len(vals_m1) < 2:
                    f_out.write(f"  {m1} vs {m2}: Not enough data\n")
                    continue

                tstat, pval = ttest_rel(vals_m1, vals_m2)

                if pval < 0.05:
                    interpretation = "SIGNIFICANT"
                else:
                    interpretation = "NOT-SIGNIFICANT"

                f_out.write(f"  {m1} vs {m2}: p-value={pval:.6f} => interpret:{interpretation}\n")
        f_out.write("\n")

    f_out.write("## Metric: SARI (Corpus-Level)\n")
    for i in range(len(valid_models)):
        for j in range(i+1, len(valid_models)):
            m1 = valid_models[i]
            m2 = valid_models[j]

            sari_m1 = []
            sari_m2 = []
            for run in RUNS:
                info_m1 = data_store[m1][run]
                info_m2 = data_store[m2][run]
                if info_m1 is None or info_m2 is None:
                    continue
                # each info_mX has "corpus_sari"
                sari_m1.append(info_m1["corpus_sari"])
                sari_m2.append(info_m2["corpus_sari"])

            if len(sari_m1) < 2:
                f_out.write(f"  {m1} vs {m2}: Not enough data\n")
                continue

            tstat, pval = ttest_rel(sari_m1, sari_m2)

            if pval < 0.05:
                interpretation = "SIGNIFICANT"
            else:
                interpretation = "NOT-SIGNIFICANT"

            f_out.write(f"  {m1} vs {m2}: p-value={pval:.6f} => interpret:{interpretation}\n")

    f_out.write("\nDONE.\n")


print(f"\nAll runs processed. Summary of results (including significance) saved to:\n{summary_txt_path}\n")
