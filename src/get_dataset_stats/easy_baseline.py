from pathlib import Path

from easy_io import read_jsonl, read_json, dump_json

from src.config import new_datasets_names, new_datasets_initial_models
from src.path import annotated_dataset_dir, dataset_stats_dir, baseline_performance_dir
from src.error_detection_analysis.calculate_performance import get_performance


if __name__ == "__main__":
    performance_dict: dict[str, dict[str, dict]] = {}
    dataset_stats = read_json(dataset_stats_dir / "stats.json")
    
    for dataset in new_datasets_names:
        # read csv file as dictionary
        for initial_model in new_datasets_initial_models:
            # retrieve a part of data corresponding to human performance annotation
            data = read_jsonl(annotated_dataset_dir / dataset / f"{initial_model.split('/')[-1]}.jsonl")
            baseline_prediction = [
                {"prediction": "error", "metadata": d["metadata"]} for d in data
            ]
            
            performance = get_performance(dataset=data, predictions=baseline_prediction)
            performance_dict.setdefault(dataset, {}).setdefault(f"initial_model={initial_model}", {})[
                "baseline_model=always_error"] = performance

            # calculate random baseline
            error_p = dataset_stats[dataset][initial_model]["error_percentage"] / 100
            
            tp_error_p = error_p * error_p
            fp_error_p = (1 - error_p) * error_p
            tn_error_p = (1 - error_p) * (1 - error_p)
            fn_error_p = error_p * (1 - error_p)

            accuracy_error_p_lc = (tp_error_p + tn_error_p) / (tp_error_p + fp_error_p + tn_error_p + fn_error_p)
            precision_error_p_lc = tp_error_p / (tp_error_p + fp_error_p) if tp_error_p + fp_error_p != 0 else 0
            recall_error_p_lc = tp_error_p / (tp_error_p + fn_error_p) if tp_error_p + fn_error_p != 0 else 0
            f1_score_error_p_lc = 2 * (precision_error_p_lc * recall_error_p_lc) / (precision_error_p_lc + recall_error_p_lc) if (precision_error_p_lc + recall_error_p_lc) != 0 else 0

            accuracy_error_p_lc, precision_error_p_lc, recall_error_p_lc, f1_score_error_p_lc
            
            performance_dict.setdefault(dataset, {}).setdefault(f"initial_model={initial_model}", {})[
                "baseline_model=random"] = {
                    "metrics": {
                        "accuracy": accuracy_error_p_lc,
                        "precision": precision_error_p_lc,
                        "recall": recall_error_p_lc,
                        "f1": f1_score_error_p_lc,
                        "true_negative_rate": tn_error_p,
                        "false_positive_rate": fp_error_p,
                        "false_negative_rate": fn_error_p,
                        "true_positive_rate": tp_error_p,
                    }
                }
    
    # save human human_performance_dict
    output_dir = baseline_performance_dir / "easy_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(performance_dict, output_dir / "performance.json")
