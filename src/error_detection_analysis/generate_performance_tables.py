from pathlib import Path
import re

from easy_io import read_json, dump_str_list_to_txt_file

from src.path import baseline_performance_dir, baseline_table_dir
from src.config import new_datasets_names, covnert_dataset_name_dict, new_datasets_initial_models, baseline_models_open, baseline_models_closed, convert_category_name_dict
from src.baseline.prompt import simple_prompt_baseline_prompts_dict, advanced_prompt_dict


category_table_dir = baseline_table_dir / "category_results_tables"


if __name__ == "__main__":
    baseline_table_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_models = baseline_models_open + baseline_models_closed
    easy_baseline_performance = read_json(baseline_performance_dir / "easy_baseline" / "performance.json")
    human_performance = read_json(baseline_performance_dir / "human_performance.json")
    
    for baseline_name in ["simple_prompt_baseline"]:
        performance_dict: dict[str, dict[str, dict[str, dict]]] = read_json(baseline_performance_dir / baseline_name / "performance.json")

        if baseline_name == "simple_prompt_baseline":
            prompt_names_list = list(simple_prompt_baseline_prompts_dict.keys()) + ["average"]
        else:
            prompt_names_list = list(advanced_prompt_dict.keys())
        
        for prompt_name in prompt_names_list:
            prompt_key = "average" if prompt_name == "average" else f"prompt={prompt_name}"
            
            for metric in ["accuracy", "f1", "precision", "recall"]:
                output_dir = baseline_table_dir / baseline_name / metric
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # build latex table
                first_row =  f"{'initial model':>30s} & {'dataset':>30s} & " + " & ".join([f"{s.split('/')[-1]:>30s}" for s in baseline_models + ["random", "human"]]) + "  \\\\"
                latex_table_lines = [first_row]
                
                for initial_model_name in new_datasets_initial_models:
                    datasets_list = new_datasets_names
                    if baseline_name == "simple_prompt_baseline":
                        datasets_list = datasets_list # + ["average"]
                    
                    for dataset_name in datasets_list:
                        if dataset_name == "average" and prompt_name != "average":
                            continue
                        
                        converted_dataset_name = "Average" if dataset_name == "average" else covnert_dataset_name_dict[dataset_name]
                        dataset_name_str = f"\scalebox{{0.9}}[1]{{{converted_dataset_name}}}"
                        row_list = [f"{'':>30s}", f"{dataset_name_str:>30s}"]
                        
                        performance_list: list[float] = []
                        for baseline_model_name in baseline_models:
                            baseline_model_performance = performance_dict[dataset_name][f"initial_model={initial_model_name}"].get(f"baseline_model={baseline_model_name}")
                            
                            if baseline_model_performance is None or len(baseline_model_performance[prompt_key]) == 0:
                                performance_list.append(-1)
                            else:
                                value = baseline_model_performance[prompt_key]["metrics"][metric]
                                if prompt_key == "average":
                                    value = value["average"]
                                
                                performance_list.append(value * 100)
                        
                        # easy baseline
                        random_performance = easy_baseline_performance[dataset_name][f"initial_model={initial_model_name}"]["baseline_model=random"]["metrics"][metric] * 100
                             
                        # make top n bold
                        top_n = 1
                        top_n_indices = sorted(range(len(performance_list)), key=lambda i: performance_list[i], reverse=True)[:top_n]
                        for idx, val in enumerate(performance_list):
                            if idx in top_n_indices:
                                performance_str = f"\\textbf{{{val:.1f}}}"
                                row_list.append(f"{performance_str:>30s}")
                            elif val < random_performance:
                                row_list.append(f"\\cellcolor[RGB]{{211,211,211}}{{{val:.1f}}}")
                            elif val == -1:
                                row_list.append(f"{'':>30s}")
                            else:
                                row_list.append(f"{val:30.1f}")
                        
                        # # always error baseline
                        # always_error_performance = easy_baseline_performance[dataset_name][f"initial_model={initial_model_name}"]["baseline_model=always_error"]["metrics"][metric] * 100
                        # row_list.append(f"{always_error_performance:30.1f}")

                        # random baseline
                        row_list.append(f"{random_performance:30.1f}")
                        
                        # human performance
                        h = human_performance[dataset_name][initial_model_name.split("/")[-1]]["metrics"][metric]
                        if dataset_name == "average":
                            h = h["average"]
                        row_list.append(f"{h * 100:30.1f}")
                        
                        # update table
                        latex_table_lines.append(" & ".join(row_list) + "  \\\\")
                
                dump_str_list_to_txt_file(latex_table_lines, output_dir / f"{metric}_{prompt_name}_table.txt")
                
                # remove multiple spaces
                updated_lines = [re.sub(' +', ' ', line) for line in latex_table_lines]
                dump_str_list_to_txt_file(updated_lines, output_dir / f"{metric}_{prompt_name}_table_single_space.txt")
    
    category_table_dir.mkdir(parents=True, exist_ok=True)
    for baseline_name in ["simple_prompt_baseline"]:
        # category performance
        category_performance_dict: dict[str, dict[str, dict[str, dict[str, dict]]]] = read_json(baseline_performance_dir / baseline_name / "category_performance.json")
        for metric in ["recall"]:
            output_dir = category_table_dir / baseline_name / metric
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for baseline_model in baseline_models + ["average"]:
                first_row =  f""
                category_latex_table_lines = [first_row]
                for initial_model_name in new_datasets_initial_models:
                    for dataset_name in new_datasets_names:
                        row_list = [f"{initial_model_name.split('/')[-1]:>30s}", f"{dataset_name:>30s}"]
                        for category in convert_category_name_dict.values():
                            if category in category_performance_dict[dataset_name].keys():
                                if baseline_model == "average":
                                    performance = category_performance_dict[dataset_name][category][f"initial_model={initial_model_name}"]["average"]["metrics"][metric]["average"]
                                else:
                                    performance = category_performance_dict[dataset_name][category][f"initial_model={initial_model_name}"][f"baseline_model={baseline_model}"]["average"]["metrics"][metric]["average"]
                                    
                                row_list.append(f"{performance*100:30.1f}")
                            else:
                                row_list.append(f"{'--':>30s}")
                        category_latex_table_lines.append(" & ".join(row_list) + "  \\\\")
                
                dump_str_list_to_txt_file(category_latex_table_lines, output_dir / f"{metric}_{baseline_model.split('/')[-1]}.txt")
                
                # remove multiple spaces
                updated_lines = [re.sub(' +', ' ', line) for line in category_latex_table_lines]
                dump_str_list_to_txt_file(updated_lines, output_dir / f"{metric}_{baseline_model.split('/')[-1]}_single_space.txt")
