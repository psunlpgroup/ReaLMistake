from pathlib import Path
import re

from easy_io import read_json, dump_str_list_to_txt_file

from src.path import baseline_performance_dir, baseline_table_dir
from src.config import new_datasets_names, covnert_dataset_name_dict, new_datasets_initial_models


majority_vote_table_dir = baseline_table_dir / "majority_vote_tables"


if __name__ == "__main__":
    baseline_models = ["meta-llama/Llama-2-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1", "Qwen/Qwen1.5-72B-Chat"]
    
    for baseline_name in ["majority_vote"]:
        performance_dict: dict[str, dict[str, dict[str, dict]]] = read_json(baseline_performance_dir / baseline_name / "performance.json")
        original_performance_dict: dict[str, dict[str, dict[str, dict]]] = read_json(baseline_performance_dir / "simple_prompt_baseline" / "performance.json")
        
        prompt_names_list = ["majority_vote"]
        
        for prompt_name in prompt_names_list:
            prompt_key = f"prompt={prompt_name}"
            
            for metric in ["accuracy", "f1", "precision", "recall"]:
                output_dir = majority_vote_table_dir / metric
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # build latex table
                first_row =  f"{'initial model':>30s} & {'dataset':>30s} & " + " & ".join([f"{s.split('/')[-1]:>30s}" for s in baseline_models]) + "  \\\\"
                latex_table_lines = [first_row]
                
                for initial_model_name in new_datasets_initial_models:
                    datasets_list = new_datasets_names
                    for dataset_name in datasets_list:
                        converted_dataset_name = covnert_dataset_name_dict[dataset_name]
                        dataset_name_str = f"\scalebox{{0.9}}[1]{{{converted_dataset_name}}}"
                        row_list = [f"{'':>30s}", f"{dataset_name_str:>30s}"]
                        
                        performance_list: list[float] = []
                        # simple prompt
                        for baseline_model_name in baseline_models:
                            original_performance = original_performance_dict[dataset_name][f"initial_model={initial_model_name}"].get(f"baseline_model={baseline_model_name}")
                            if original_performance is None or len(original_performance["average"]) == 0:
                                performance_list.append(-1)
                            else:
                                value = original_performance["average"]["metrics"][metric]["average"]
                                performance_list.append(value * 100)
                        
                        # majority vote
                        baseline_model_performance = performance_dict[dataset_name][f"initial_model={initial_model_name}"].get(f"baseline_model={'__'.join([b.split('/')[-1] for b in baseline_models])}")
                        if baseline_model_performance is None or len(baseline_model_performance[prompt_key]) == 0:
                            performance_list.append(-1)
                        else:
                            value = baseline_model_performance["average"]["metrics"][metric]["average"]
                            performance_list.append(value * 100)

                        for idx, val in enumerate(performance_list):
                            if idx < len(performance_list) - 1 and val > performance_list[-1]:
                                row_list.append(f"\\cellcolor[RGB]{{240,230,140}}{{{val:.1f}}}")
                            elif val == -1:
                                row_list.append(f"{'':>30s}")
                            else:
                                row_list.append(f"{val:30.1f}")
                        
                        # update table
                        latex_table_lines.append(" & ".join(row_list) + "  \\\\")
                
                dump_str_list_to_txt_file(latex_table_lines, output_dir / f"{metric}_{prompt_name}_table.txt")
                
                # remove multiple spaces
                updated_lines = [re.sub(' +', ' ', line) for line in latex_table_lines]
                dump_str_list_to_txt_file(updated_lines, output_dir / f"{metric}_{prompt_name}_table_single_space.txt")
