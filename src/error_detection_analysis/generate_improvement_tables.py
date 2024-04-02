from pathlib import Path
import re

from easy_io import read_json, dump_str_list_to_txt_file

from src.path import baseline_performance_dir, baseline_table_dir
from src.config import new_datasets_names, covnert_dataset_name_dict, new_datasets_initial_models, baseline_models_open, baseline_models_closed


diff_tables_dir = baseline_table_dir / "improvement_tables"


if __name__ == "__main__":
    baseline_models = baseline_models_open + baseline_models_closed
    
    for baseline_name in ["advanced_prompt_baseline"]:
        performance_dict: dict[str, dict[str, dict[str, dict]]] = read_json(baseline_performance_dir / baseline_name / "performance.json")
        original_performance_dict: dict[str, dict[str, dict[str, dict]]] = read_json(baseline_performance_dir / "simple_prompt_baseline" / "performance.json")
        
        prompt_names_list = ["average"]
        
        for prompt_name in prompt_names_list:
            prompt_key = "average"
            
            for metric in ["accuracy", "f1", "precision", "recall"]:
                output_dir = diff_tables_dir / baseline_name / metric
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # build latex table
                first_row =  f"{'initial model':>30s} & {'dataset':>30s} & " + " & ".join([f"{s.split('/')[-1]:>30s}" for s in baseline_models + ["random", "human"]]) + "  \\\\"
                latex_table_lines = [first_row]
                
                for initial_model_name in new_datasets_initial_models:
                    datasets_list = new_datasets_names
                    for dataset_name in datasets_list:
                        converted_dataset_name = covnert_dataset_name_dict[dataset_name]
                        dataset_name_str = f"\scalebox{{0.9}}[1]{{{converted_dataset_name}}}"
                        row_list = [f"{'':>30s}", f"{dataset_name_str:>30s}"]
                        
                        performance_list: list[float] = []
                        for baseline_model_name in baseline_models:
                            baseline_model_performance = performance_dict[dataset_name][f"initial_model={initial_model_name}"].get(f"baseline_model={baseline_model_name}")
                            original_performance = original_performance_dict[dataset_name][f"initial_model={initial_model_name}"].get(f"baseline_model={baseline_model_name}")
                            
                            if baseline_model_performance is None or len(baseline_model_performance[prompt_key]) == 0:
                                performance_list.append(-1)
                            else:
                                value = baseline_model_performance[prompt_key]["metrics"][metric]["average"] - original_performance["prompt=baseline_errordetection_prompt_1"]["metrics"][metric]
                                performance_list.append(value * 100)
                        
                        # make top n bold
                        top_n = 1
                        top_n_indices = sorted(range(len(performance_list)), key=lambda i: performance_list[i], reverse=True)[:top_n]
                        for idx, val in enumerate(performance_list):
                            if val < 0:
                                row_list.append(f"\\cellcolor[RGB]{{211,211,211}}{{{val:.1f}}}")
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
