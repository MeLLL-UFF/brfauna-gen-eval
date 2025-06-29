import os
import json
from statistics import mean
import evaluate
from scipy.stats import ttest_rel


# Define your runs & models
RUNS = ["r1", "r2", "r3"]
MODELS = [
    "bode_318B_pira",
    "boto_gemma_pira",
    "cabra_llama8b_pira",
    "piriquito_ollama_pira",
    "tucano_pira",
    "gptmini_pira_final_output",
    "gpt_pira_final_output",
    "sabia_pira_final_output",
    "sabiazinho_pira_final_output"
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
    "ROUGE-Lsum",
    "OverlapPrecision",
    "OverlapRecall",
    "OverlapF1"
]


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
      - BERTScore, METEOR, BLEU, ROUGE, Overlap-based F1
    Return:
        overall_metrics (dict): aggregated for entire corpus
        per_sample_list (list): one entry per sample, with all original data + sample-level metrics
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_generated = [d["generated"] for d in data]
    all_refs      = [d["reference"] for d in data]

    bert_results = bertscore_metric.compute(
        predictions=all_generated,
        references=all_refs,
        lang="pt"  
    )

    meteor_scores = []
    bleu_scores   = []
    rouge_scores  = []


    overlap_precisions = []
    overlap_recalls    = []
    overlap_f1s        = []

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
        # => dict with "rouge1","rouge2","rougeL","rougeLsum"
        rouge_scores.append(rouge_res)

        # Overlap-based Precision/Recall/F1
        gen_tokens = gen_text.split()
        ref_tokens = ref_text.split()
        common_tokens = set(gen_tokens).intersection(ref_tokens)

        precision = len(common_tokens) / len(gen_tokens) if gen_tokens else 0.0
        recall    = len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0
        f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        overlap_precisions.append(precision)
        overlap_recalls.append(recall)
        overlap_f1s.append(f1_score)

    per_sample_list = []
    for i, entry in enumerate(data):
        sample_info = dict(entry)  
        
        sample_metrics = {
            "BERTScore_precision": float(bert_results["precision"][i]),
            "BERTScore_recall":    float(bert_results["recall"][i]),
            "BERTScore_f1":        float(bert_results["f1"][i]),

            "METEOR":              float(meteor_scores[i]),
            "BLEU":                float(bleu_scores[i]),
            "ROUGE-1":             float(rouge_scores[i]["rouge1"]),
            "ROUGE-2":             float(rouge_scores[i]["rouge2"]),
            "ROUGE-L":             float(rouge_scores[i]["rougeL"]),
            "ROUGE-Lsum":          float(rouge_scores[i]["rougeLsum"]),

            # Overlap-based
            "OverlapPrecision": float(overlap_precisions[i]),
            "OverlapRecall":    float(overlap_recalls[i]),
            "OverlapF1":        float(overlap_f1s[i])
        }

        # Attach metrics to this sample
        sample_info["metrics"] = sample_metrics
        per_sample_list.append(sample_info)

    N = len(data)
    overall_metrics = {}
    if N > 0:
        overall_metrics["BERTScore_precision"] = float(sum(bert_results["precision"]) / N)
        overall_metrics["BERTScore_recall"]    = float(sum(bert_results["recall"]) / N)
        overall_metrics["BERTScore_f1"]        = float(sum(bert_results["f1"]) / N)

        overall_metrics["METEOR"] = float(sum(meteor_scores) / N)
        overall_metrics["BLEU"]   = float(sum(bleu_scores) / N)

        avg_rouge1    = sum(r["rouge1"]    for r in rouge_scores) / N
        avg_rouge2    = sum(r["rouge2"]    for r in rouge_scores) / N
        avg_rougeL    = sum(r["rougeL"]    for r in rouge_scores) / N
        avg_rougeLsum = sum(r["rougeLsum"] for r in rouge_scores) / N

        overall_metrics["ROUGE-1"]    = float(avg_rouge1)
        overall_metrics["ROUGE-2"]    = float(avg_rouge2)
        overall_metrics["ROUGE-L"]    = float(avg_rougeL)
        overall_metrics["ROUGE-Lsum"] = float(avg_rougeLsum)

        # Overlap-based
        sum_precision = sum(overlap_precisions)
        sum_recall    = sum(overlap_recalls)
        sum_f1        = sum(overlap_f1s)
        overall_metrics["OverlapPrecision"] = float(sum_precision / N)
        overall_metrics["OverlapRecall"]    = float(sum_recall    / N)
        overall_metrics["OverlapF1"]        = float(sum_f1        / N)

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
        for run in RUNS:
            data_this_run = data_store[model][run]
            if data_this_run is None:
                continue
            per_sample = data_this_run["per_sample"] 
            if len(per_sample) == 0:
                continue

            run_avg = {}
            for met in SAMPLE_METRICS_TO_REPORT:
                vals = [sample["metrics"][met] for sample in per_sample]
                run_avg[met] = mean(vals)
            run_averages.append(run_avg)

        if not run_averages:
            continue

        # Average across runs for sample-based metrics
        final_avg = {}
        for met in SAMPLE_METRICS_TO_REPORT:
            vals = [ra[met] for ra in run_averages if met in ra]
            final_avg[met] = sum(vals) / len(vals) if len(vals) > 0 else 0.0

        average_results_across_runs[model] = final_avg

    for model, avg_dict in average_results_across_runs.items():
        f_out.write(f"## Model: {model}\n")
        for met in SAMPLE_METRICS_TO_REPORT:
            val = avg_dict[met]
            f_out.write(f"  {met}: {val:.4f}\n")
        f_out.write("\n")


    f_out.write("# Statistical Significance (Paired t-test)\n\n")
    valid_models = sorted(average_results_across_runs.keys())

    for metric in SAMPLE_METRICS_TO_REPORT:
        f_out.write(f"## Metric: {metric}\n")
        for i in range(len(valid_models)):
            for j in range(i+1, len(valid_models)):
                m1 = valid_models[i]
                m2 = valid_models[j]

                # Gather the per-sample metric for all runs
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
                interpretation = "SIGNIFICANT" if pval < 0.05 else "NOT-SIGNIFICANT"
                f_out.write(f"  {m1} vs {m2}: p={pval:.6f} => {interpretation}\n")
        f_out.write("\n")

    f_out.write("\nDONE.\n")

print(f"\nAll runs processed. Summary of results (including significance) saved to:\n{summary_txt_path}\n")
