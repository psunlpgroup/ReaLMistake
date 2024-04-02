from easy_io import read_jsonl, dump_json
from pathlib import Path

from src.config import new_datasets_names, new_datasets_initial_models
from src.path import annotated_dataset_dir, dataset_stats_dir
from src.config import convert_category_name_dict


def get_average_token_num(dataset: list[dict], key: str) -> float:
    return sum(len(d[key].split()) for d in dataset) / len(dataset)


def get_error_percentage(dataset: list[dict]) -> float:
    return sum(d["error_label"] == "error" for d in dataset) / len(dataset) * 100


if __name__ == "__main__":
    output = ""
    error_categories_output = ""
    stat_dict: dict[str, dict[str, dict]] = {}
    for initial_model in new_datasets_initial_models:
        output += f"{initial_model}\n"
        error_categories_output += f"{initial_model}\n"
        
        for dataset_name in new_datasets_names:
            dataset = read_jsonl(annotated_dataset_dir / dataset_name / f"{initial_model.split('/')[-1]}.jsonl")
            prefix = f"{dataset_name:30s}"
            
            # general statistics
            error_percentage = get_error_percentage(dataset)
            row = f"{prefix} &  {len(dataset):4d} & {get_average_token_num(dataset, key='input'):5.0f} & {get_average_token_num(dataset, key='llm_response'):5.0f} & {error_percentage:.1f} \\\\"
            output += f"{row}\n"
            
            stat_dict.setdefault(dataset_name, {}).setdefault(initial_model, {})["error_percentage"] = error_percentage
            
            # error categories
            count_list: list[str] = []
            for d in dataset:
                count_list.extend(d["error_categories"])
            
            ec_row = f"{prefix} & "
            for ec in convert_category_name_dict.values():
                c = count_list.count(ec)
                ec_row += f"{c / len(dataset) * 100:4.1f} & " if c > 0 else "  -- & "
            ec_row += f"{error_percentage:.1f} \\\\"
            error_categories_output += f"{ec_row}\n"
            
        output += "\n"
        error_categories_output += "\n"
    
    with open(dataset_stats_dir / "stats_table.txt", "w") as f:
        f.write(output)
    
    with open(dataset_stats_dir / "error_categories_table.txt", "w") as f:
        f.write(error_categories_output)
    
    dump_json(stat_dict, dataset_stats_dir / "stats.json")
