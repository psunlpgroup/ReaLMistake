import random

from tqdm import tqdm
from easy_io import dump_json, read_jsonl, read_json, dump_str_list_to_txt_file
import numpy as np

from src.path import annotated_dataset_dir, baseline_performance_dir, baseline_analysis_dir
from src.config import new_datasets_names, covnert_dataset_name_dict
from src.baseline.simple_prompt_baseline import get_baseline_output_dir
from src.error_detection_analysis.calculate_performance import get_performance


self_consistency_analysis_output_dir = baseline_analysis_dir / "self_consistency_analysis_results"

if __name__ == "__main__":
    self_consistency_analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    number_of_samples = 50
    prompts_list = ["baseline_errordetection_prompt_1"]
    initial_models_list = ["gpt-4-0613"]
    baseline_models_list = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "gpt-3.5-turbo-0125", "gpt-4-0125-preview"]
    
    # simple prompts performance on number_of_samples cases
    baseline_name = "simple_prompt_baseline"
    performance_dict: dict[str, dict[str, dict[str, dict[str, dict[str, dict]]]]] = {}
    for dataset_name in new_datasets_names:
        for initial_model in initial_models_list:
            initial_model_key = f"initial_model={initial_model}"
            # load dataset (ground truth)
            dataset_path = annotated_dataset_dir / dataset_name / f"{initial_model.split('/')[-1]}.jsonl"
            ground_truth = read_jsonl(dataset_path)[:number_of_samples]
            for baseline_model in baseline_models_list:
                baseline_model_key = f"baseline_model={baseline_model}"
                baseline_output_dir = get_baseline_output_dir(baseline_name, dataset_name, initial_model, baseline_model)
                for prompt_name in prompts_list:
                    baseline_path = baseline_output_dir / f"{prompt_name}.jsonl"
                    if baseline_path.exists():
                        error_detection_results = read_jsonl(baseline_path)[:number_of_samples]
                        ground_truth_truncated = ground_truth[:len(error_detection_results)]
                        performance = get_performance(dataset=ground_truth_truncated, predictions=error_detection_results)
                    else:
                        performance = {}
                    performance_dict.setdefault(dataset_name, {}).setdefault(initial_model_key, {}).setdefault(baseline_model_key, {})[f"prompt={prompt_name}"] = performance
    
    dump_json(performance_dict, self_consistency_analysis_output_dir / "simple_prompt_performance_first_k_samples.json")
    simple_baseline_performance = performance_dict

    # self consistency performance
    self_consistency_performance = read_json(baseline_performance_dir / "self_consistency" / "performance.json")
    
    # bootstrap pairwise comparison
    bootstraph_path = self_consistency_analysis_output_dir / "bootstrap_results.json"
    if False:  # bootstraph_path.exists():
        bootstrap_results = read_json(bootstraph_path)
    else:
        bootstrap_num = 1000
        bootstrap_results: dict[str, dict[str, dict[str, dict[str, dict[str, dict]]]]] = {}
        for dataset_name in new_datasets_names:
            for initial_model in initial_models_list:
                initial_model_key = f"initial_model={initial_model}"
                # load dataset (ground truth)
                dataset_path = annotated_dataset_dir / dataset_name / f"{initial_model.split('/')[-1]}.jsonl"
                ground_truth = read_jsonl(dataset_path)[:number_of_samples]
                ground_truth_truncated = ground_truth[:number_of_samples]
                for baseline_model in baseline_models_list:
                    baseline_model_key = f"baseline_model={baseline_model}"
                    
                    baseline_output_dict = {}
                    for baseline_name in ["simple_prompt_baseline", "self_consistency"]:
                        baseline_output_dir = get_baseline_output_dir(baseline_name, dataset_name, initial_model, baseline_model)
                        prompt_name = prompts_list[0]
                        baseline_path = baseline_output_dir / f"{prompt_name}.jsonl"
                        if baseline_path.exists():
                            baseline_output_dict[baseline_name] = read_jsonl(baseline_path)[:number_of_samples]
                    
                    b_performance: dict[str, dict[str, dict[str, dict]]] = {}
                    if len(baseline_output_dict) == 2:
                        b_performance = {"p-value": {}, "metrics": {}}
                        for bn in tqdm(range(bootstrap_num)):
                            b_ground_truth = random.Random(bn).choices(ground_truth_truncated, k=number_of_samples)
                            for baseline_name in ["simple_prompt_baseline", "self_consistency"]:
                                b_baseline_output = random.Random(bn).choices(baseline_output_dict[baseline_name], k=number_of_samples)

                                performance = get_performance(dataset=b_ground_truth, predictions=b_baseline_output)
                                for metric in ["f1", "precision", "recall"]:
                                    b_performance.setdefault("metrics", {}).setdefault(baseline_name, {}).setdefault(metric, []).append(performance["metrics"][metric])
                        
                        for metric in ["f1", "precision", "recall"]:
                            sc_worse: list[bool] = []
                            for bn in range(bootstrap_num):
                                if b_performance["metrics"]["simple_prompt_baseline"][metric][bn] >= b_performance["metrics"]["self_consistency"][metric][bn]:
                                    sc_worse.append(True)
                                else:
                                    sc_worse.append(False)
                            p_value = np.mean(sc_worse)
                            b_performance["p-value"][metric] = p_value

                    bootstrap_results.setdefault(dataset_name, {}).setdefault(initial_model_key, {}).setdefault(baseline_model_key, {})[f"prompt={prompt_name}"] = b_performance
        dump_json(bootstrap_results, self_consistency_analysis_output_dir / "bootstrap_results.json")

    # generate table
    tables_list: list[list[str]] = []
    prompt_key = f"prompt={prompts_list[0]}"
    for metric in ["f1", "precision", "recall"]:
        tables_list.append(metric)
        for dataset_name in new_datasets_names:
            row_list = [covnert_dataset_name_dict[dataset_name]]
            for baseline_model in baseline_models_list:
                baseline_model_key = f"baseline_model={baseline_model}"
                for baseline_name, performance_dict in [["simple_prompt_baseline", simple_baseline_performance], ["self_consistency", self_consistency_performance]]:
                    performance_d = performance_dict[dataset_name][f"initial_model={initial_models_list[0]}"][baseline_model_key][prompt_key]
                    if len(performance_d) == 0:
                        row_list.append("  ")
                    else:
                        performance = performance_d["metrics"][metric]
                        
                        # put mark if p < 0.1
                        p_value_dict = bootstrap_results[dataset_name][f"initial_model={initial_models_list[0]}"][baseline_model_key][prompt_key]
                        
                        cell = f"{performance*100:.1f} \phantom{{.}}"
                        if baseline_name == "self_consistency":
                            if len(p_value_dict) > 0:
                                if p_value_dict["p-value"][metric] < 0.1:
                                    cell = "{{\\bf " + cell.split()[0] + "$^*$}}"
                        
                        row_list.append(cell)
            tables_list.append(" & ".join(row_list) + " \\\\")
    dump_str_list_to_txt_file(tables_list, self_consistency_analysis_output_dir / "self_consistency_performance_table.txt")
        
