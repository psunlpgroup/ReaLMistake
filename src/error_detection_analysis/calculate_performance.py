from easy_io import dump_json, read_jsonl
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.path import annotated_dataset_dir, baseline_performance_dir
from src.config import new_datasets_names, new_datasets_initial_models, baseline_models_closed, baseline_models_open, convert_category_name_dict
from src.baseline.prompt import simple_prompt_baseline_prompts_dict
from src.baseline.simple_prompt_baseline import get_baseline_output_dir


def get_performance(dataset: list[dict], predictions: list[dict]) -> dict:
    """Calculate performance of error detection.
    Args:
        dataset: list of annotations.
        predictions: list of predictions. Each element of the list is a dict with keys "metadata" and "prediction".
    """
    
    if len(dataset) != len(predictions):
        return {}
    
    assert all([d["metadata"]["id"] == r["metadata"]["id"] for d, r in zip(dataset, predictions)])
    
    prediction_error_num = sum([1 for p in predictions if p["prediction"] == "error"])
    gold_error_num = sum([1 for d in dataset if d["error_label"] == "error"])

    y_true = [1 if d["error_label"] == "error" else 0 for d in dataset]
    y_pred = [1 if p["prediction"] == "error" else 0 for p in predictions]

    # accuracy
    correct = sum([1 for d, p in zip(dataset, predictions) if p["prediction"] == d["error_label"]])
    total = len(predictions)
    accuracy = correct / total
    
    # f1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    # confusion matrix (error = positive) 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() / total
    
    return {
        "total_num": total,
        "prediction_error_num": prediction_error_num,
        "gold_error_num": gold_error_num,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negative_rate": tn,
            "false_positive_rate": fp,
            "false_negative_rate": fn,
            "true_positive_rate": tp,
        }
    }


def get_average_of_list_of_dict(list_of_dict: list[dict]) -> dict:
    average_dict = {}

    if len(list_of_dict) == 0:
        return average_dict
    
    for key in list_of_dict[0].keys():
        list_of_value = [d[key] for d in list_of_dict]
        
        average_dict[key] = {
            "average": np.mean(list_of_value).item(),
            "stdev": np.std(list_of_value).item(),
        }
    return average_dict


def get_average_performance(performance_dict: dict) -> dict:
    # check
    for value in performance_dict.values():
        if len(value) == 0:
            return {}    
    
    average_performance = {}
    first_key = list(performance_dict.keys())[0]
    for key in ["total_num", "gold_error_num"]:
        average_performance[key] = performance_dict[first_key][key]

    # calculate average
    list_of_dict = [performance_dict[key]["metrics"] for key in performance_dict.keys()]
    average_performance["metrics"] = get_average_of_list_of_dict(list_of_dict)
    
    return average_performance


dataset_to_categories = {
    "math_word_problem_generation": ["reasoning", "constraints"],
    "finegrained_fact_verification": ["reasoning", "constraints", "context"],
    "answerability_classification": ["reasoning", "knowledge"],
}


if __name__ == "__main__":
    baseline_performance_dir.mkdir(parents=True, exist_ok=True)
    
    for baseline_name in ["simple_prompt_baseline", "advanced_prompt_baseline", "majority_vote", "self_consistency"]:
        print(baseline_name)
        
        # config
        initial_models_list = ["gpt-4-0613"] if baseline_name == "self_consistency" else new_datasets_initial_models
        
        baseline_models_list = {
            "simple_prompt_baseline": baseline_models_open + baseline_models_closed,
            "advanced_prompt_baseline": baseline_models_open + baseline_models_closed,
            "majority_vote": ["Llama-2-70b-chat-hf__Mixtral-8x7B-Instruct-v0.1__Qwen1.5-72B-Chat"],
            "self_consistency": ["mistralai/Mixtral-8x7B-Instruct-v0.1", "gpt-3.5-turbo-0125", "gpt-4-0125-preview"]
        }[baseline_name]
        
        prompts_list = {
            "simple_prompt_baseline": list(simple_prompt_baseline_prompts_dict.keys()),
            "advanced_prompt_baseline": ["cot_instruction_prompt"],
            "majority_vote": ["majority_vote"],
            "self_consistency": ["baseline_errordetection_prompt_1"]
        }[baseline_name]
        
        # calculate performance        
        performance_dict: dict[str, dict[str, dict[str, float]]] = {}
        category_performance_dict: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
        for dataset_name in new_datasets_names:
            print(dataset_name)
            
            for initial_model in initial_models_list:
                initial_model_key = f"initial_model={initial_model}"
                print(initial_model_key)
                
                # load dataset (ground truth)
                dataset_path = annotated_dataset_dir / dataset_name / f"{initial_model.split('/')[-1]}.jsonl"
                ground_truth = read_jsonl(dataset_path)
                
                for baseline_model in baseline_models_list:
                    baseline_model_key = f"baseline_model={baseline_model}"
                    baseline_output_dir = get_baseline_output_dir(baseline_name, dataset_name, initial_model, baseline_model)
                    
                    print(baseline_model_key)
                    for prompt_name in prompts_list:
                        baseline_path = baseline_output_dir / f"{prompt_name}.jsonl"
                        
                        if baseline_path.exists():
                            error_detection_results = read_jsonl(baseline_path)
                            ground_truth_truncated = ground_truth[:len(error_detection_results)]
                            
                            performance = get_performance(dataset=ground_truth_truncated, predictions=error_detection_results)
                            
                            # performance on each category
                            for category in dataset_to_categories[dataset_name]:
                                selected_data = [d for d in ground_truth_truncated if convert_category_name_dict[category] in d["error_categories"]]
                                selected_pred = [p for d, p in zip(ground_truth_truncated, error_detection_results) if convert_category_name_dict[category] in d["error_categories"]]
                                category_performance = get_performance(dataset=selected_data, predictions=selected_pred)
                                
                                category_performance_dict.setdefault(dataset_name, {}).setdefault(convert_category_name_dict[category], {}).setdefault(
                                    initial_model_key, {}).setdefault(baseline_model_key, {})[prompt_name] = category_performance
                        else:
                            performance = {}
                            for category in dataset_to_categories[dataset_name]:
                                category_performance_dict.setdefault(dataset_name, {}).setdefault(convert_category_name_dict[category], {}).setdefault(
                                    initial_model_key, {}).setdefault(baseline_model_key, {})[prompt_name] = {}
                        
                        performance_dict.setdefault(dataset_name, {}).setdefault(initial_model_key, {}).setdefault(baseline_model_key, {})[f"prompt={prompt_name}"] = performance

                    average_performance = get_average_performance(performance_dict[dataset_name][initial_model_key][baseline_model_key])
                    performance_dict[dataset_name][initial_model_key][baseline_model_key]["average"] = average_performance

                    for category in dataset_to_categories[dataset_name]:
                        category_average_performance = get_average_performance(category_performance_dict[dataset_name][convert_category_name_dict[category]][initial_model_key][baseline_model_key])
                        category_performance_dict[dataset_name][convert_category_name_dict[category]][initial_model_key][baseline_model_key]["average"] = category_average_performance
                
                # average category performance
                for category in dataset_to_categories[dataset_name]:
                    list_of_all_category_performance = []
                    for baseline_model in baseline_models_list:
                        baseline_model_key = f"baseline_model={baseline_model}"
                        raw_average: dict[str, dict] = category_performance_dict[dataset_name][convert_category_name_dict[category]][initial_model_key][baseline_model_key]["average"]
                        
                        if len(raw_average) > 0:
                            average = {r: v["average"] for r, v in raw_average["metrics"].items()}
                            list_of_all_category_performance.append(average)
                    category_performance_dict[dataset_name][convert_category_name_dict[category]][initial_model_key].setdefault("average", {})["metrics"] = get_average_of_list_of_dict(list_of_all_category_performance)

        # average performance on three datasets
        for initial_model in initial_models_list:
            for baseline_model in baseline_models_list:
                performance_list = []
                total_num = 0
                gold_error_num = 0
                for dataset_name in new_datasets_names:
                    raw_average_performance: dict[str, dict] = performance_dict[dataset_name][f"initial_model={initial_model}"][f"baseline_model={baseline_model}"]["average"]
                    if len(raw_average_performance) == 0:
                        continue
                    
                    average = {r: v["average"] for r, v in raw_average_performance["metrics"].items()}
                    performance_list.append(average)
                    
                    total_num += raw_average_performance["total_num"]
                    gold_error_num += raw_average_performance["gold_error_num"]
                
                average_metrics = get_average_of_list_of_dict(performance_list)
                average_performance = {
                    "total_num": total_num,
                    "gold_error_num": gold_error_num,
                    "metrics": average_metrics,
                }
                performance_dict.setdefault("average", {}).setdefault(f"initial_model={initial_model}", {}).setdefault(f"baseline_model={baseline_model}", {})["average"] = average_performance
        
        # output
        performance_dir = baseline_performance_dir / baseline_name
        performance_dir.mkdir(parents=True, exist_ok=True)
        dump_json(performance_dict, performance_dir / "performance.json")

        category_performance_dir = baseline_performance_dir / baseline_name
        category_performance_dir.mkdir(parents=True, exist_ok=True)
        dump_json(category_performance_dict, category_performance_dir / "category_performance.json")
