from pathlib import Path

from easy_io import read_json, dump_json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.markers
from PIL import Image

from src.path import baseline_performance_dir, baseline_analysis_dir
from src.config import new_datasets_names, new_datasets_initial_models, baseline_models_open, baseline_models_closed, convert_model_names_dict

bias_analysis_dir = baseline_analysis_dir / "bias_analysis_results"

dataset_names_convert = {
    "math_word_problem_generation": "MathGen",
    "finegrained_fact_verification": "FgFactV",
    "answerability_classification": "AnsCls",
}

metric_names_convert = {
    "accuracy": "Accuracy",
    "false_negative_rate": "False Negative",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
}


matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['patch.linewidth'] = 2
matplotlib.rcParams['legend.fontsize'] = 11

plot_params = {"linestyle": "", "markersize": 7}

def figures_config(initial_model_name: str, metric: str, bias_name: str):
    plt.xlim([-0.5, len(new_datasets_names) - 0.5])
    
    plt.title(convert_model_names_dict[initial_model_name], y=-.3)
    plt.xticks(list(range(len(new_datasets_names))), [dataset_names_convert[s] for s in new_datasets_names])
    
    if "gpt-4" in initial_model_name:
        plt.ylabel(f"Diff in {metric_names_convert[metric]}")
    
    if "llama" in initial_model_name:
        plt.gca().set_yticklabels([])
        plt.legend(bbox_to_anchor=(1, 1.05))


def join_figures_horizontally(figure_path_list: list[Path], output_path: Path):
    images = [Image.open(img) for img in figure_path_list]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste((255, 255, 255), (0, 0, total_width, max_height))  # whilte background
    
    x_offset = 0
    for im in images:
        # paste from bottom
        new_im.paste(im, (x_offset, max_height - im.size[1]))
        x_offset += im.size[0]
        
    new_im.save(output_path)


def get_x_for_model(baseline_model_name: str) -> float:
    if "gpt-4" in baseline_model_name:
        return 0.0
    elif "gpt-3" in baseline_model_name:
        return 0.1
    elif "claude" in baseline_model_name:
        return 0.1
    elif "gemini" in baseline_model_name or "gemma" in baseline_model_name:
        return -0.1
    elif "llama" in baseline_model_name:
        return 0.0
    else:
        return -0.2


def get_model_color(baseline_model_name: str) -> str:
    if "gpt-4" in baseline_model_name:
        return "red"
    elif "gpt-3" in baseline_model_name:
        return "magenta"
    elif "claude" in baseline_model_name:
        return "orange"
    elif "gemini" in baseline_model_name or "gemma" in baseline_model_name:
        return "green"
    elif "llama" in baseline_model_name:
        return "blue"
    elif "Qwen" in baseline_model_name:
        return "purple"
    else:
        return "gray"


markers_dict = {
    "google/gemma-7b-it": "$g$",
    "meta-llama/Llama-2-13b-chat-hf": "<",
    "meta-llama/Llama-2-70b-chat-hf": ">",
    "mistralai/Mistral-7B-Instruct-v0.1": "$m$",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "$M$",
    "Qwen/Qwen1.5-14B-Chat": "$q$",
    "Qwen/Qwen1.5-72B-Chat": "$Q$",
    "gpt-3.5-turbo-0125": '*',
    "models/gemini-1.0-pro-001": "$G$",
    "claude-3-opus-20240229": "$C$",
    "gpt-4-0613": "X",
    "gpt-4-0125-preview": "P",
}

metrics_list = ["accuracy", "false_negative_rate", "recall", "precision", "f1"]

if __name__ == "__main__":
    bias_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_models = baseline_models_open + baseline_models_closed
    
    for baseline_name in ["simple_prompt_baseline"]:
        # prompt bias
        performance_dict: dict[str, dict[str, dict[str, dict]]] = read_json(baseline_performance_dir / baseline_name / "performance.json")
        bias_results: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]] = {}
            
        for prompt_group_name, prompt_groups_list in [
                ["wording_bias", [[f"baseline_errordetection_prompt_{idx}" for idx in [3, 4]], [f"baseline_errordetection_prompt_{idx}" for idx in [1, 2]]]],
                ["order_bias_1", [["baseline_errordetection_prompt_2"], ["baseline_errordetection_prompt_1"]]],
                ["order_bias_2", [["baseline_errordetection_prompt_4"], ["baseline_errordetection_prompt_3"]]],
            ]:
            
            for metric in metrics_list:
                figure_path_list = []
                
                all_initial_model_diff: list[float] = []
                all_initial_model_diff_dict: dict[str, list[float]] = {}
                for initial_model_name in new_datasets_initial_models:
                    initial_model_key = f"initial_model={initial_model_name}"
                    
                    all_dataset_diff: list[float] = []
                    for dataset_name in new_datasets_names:
                        all_model_diff: list[float] = []
                        for baseline_model_name in baseline_models:
                            baseline_model_key = f"baseline_model={baseline_model_name}"
                            baseline_model_performance = performance_dict[dataset_name][initial_model_key].get(baseline_model_key)
                            if baseline_model_performance is None or len(baseline_model_performance["average"]) == 0:
                                continue
                            else:
                                bias_results.setdefault(prompt_group_name, {}).setdefault(dataset_name, {}).setdefault(initial_model_key, {}).setdefault(baseline_model_key, {}).setdefault(metric, {})
                                results_of_this_model = {}
                                for prompt_idx, prompt_group in enumerate(prompt_groups_list):
                                    values_list: list[float] = []
                                    for prompt_name in prompt_group:
                                        values_list.append(baseline_model_performance[f"prompt={prompt_name}"]["metrics"][metric] * 100)
                                    
                                    results_of_this_model[f"prompt_group={prompt_idx}"] = np.average(values_list).item()
                                
                                results_of_this_model["prompt_group=1 - prompt_group=0"] = results_of_this_model["prompt_group=1"] - results_of_this_model["prompt_group=0"]
                                bias_results[prompt_group_name][dataset_name][initial_model_key][baseline_model_key][metric] = results_of_this_model

                                diff = results_of_this_model["prompt_group=1 - prompt_group=0"]
                                all_model_diff.append(diff)
                                all_initial_model_diff_dict.setdefault(baseline_model_name, []).append(diff)
                        
                        bias_results[prompt_group_name][dataset_name][initial_model_key].setdefault("average", {}).setdefault(metric, {})["prompt_group=1 - prompt_group=0"] = {
                            "average": np.average(all_model_diff).item(), "std": np.std(all_model_diff).item()
                        }
                        all_dataset_diff.extend(all_model_diff)
                    
                    bias_results[prompt_group_name].setdefault("average", {}).setdefault(initial_model_key, {}).setdefault(metric, {})["prompt_group=1 - prompt_group=0"] = {
                        "average": np.average(all_dataset_diff).item(), "std": np.std(all_dataset_diff).item()
                    }
                    all_initial_model_diff.extend(all_dataset_diff)

                    # plot for prompt biases
                    fig = plt.figure(figsize=[4, 3])
                    plt.axhline(y=0, color="gray", linewidth=1.5)
                    for baseline_idx, baseline_model_name in enumerate(baseline_models):
                        diff_list: list[float] = []
                        
                        baseline_model_key = f"baseline_model={baseline_model_name}"
                        if baseline_model_key in bias_results[prompt_group_name][dataset_name][initial_model_key].keys():
                            for dataset_name in new_datasets_names:
                                diff_list.append(
                                    bias_results[prompt_group_name][dataset_name][initial_model_key][baseline_model_key][metric]["prompt_group=1 - prompt_group=0"]
                                )
                        
                        x_values = [get_x_for_model(baseline_model_name) + idx for idx in range(len(diff_list))]
                        plt.plot(x_values, diff_list, label=convert_model_names_dict[baseline_model_name], marker=markers_dict[baseline_model_name], color=get_model_color(baseline_model_name), **plot_params)
                    
                    if metric == "false_negative_rate":
                        plt.ylim([-80, 20])
                    if metric == "accuracy":
                        plt.ylim([-20, 50])
                    if metric == "recall":
                        plt.ylim([-25, 90])
                    if metric == "precision":
                        plt.ylim([-45, 45])
                    
                    figures_config(initial_model_name, metric, prompt_group_name)
                    
                    # save
                    output_dir = bias_analysis_dir / "figures"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    figure_path = output_dir / f"{prompt_group_name}_{metric}_{initial_model_name.split('/')[-1]}.png"
                    figure_path_list.append(figure_path)
                    plt.savefig(figure_path, bbox_inches='tight')
                    plt.close()
                
                # horizontally concatenate the figures
                join_figures_horizontally(figure_path_list, bias_analysis_dir / f"{prompt_group_name}_{metric}.png")
        
                bias_results[prompt_group_name].setdefault("average", {}).setdefault("average", {}).setdefault(metric, {})["prompt_group=1 - prompt_group=0"] = {
                    "average": np.average(all_initial_model_diff).item(), "std": np.std(all_initial_model_diff).item()
                }
                
                for model_group in [["gpt-4-0613", "gpt-4-0125-preview", "claude-3-opus-20240229"]]:
                    all_m_diff: list[float] = []
                    for m_name in model_group:
                        all_m_diff.extend(all_initial_model_diff_dict[m_name])
                    
                    bias_results[prompt_group_name]["average"].setdefault(f"baseline_model={'-'.join(sorted(model_group))}", {}).setdefault(metric, {})["prompt_group=1 - prompt_group=0"] = {
                        "average": np.average(all_m_diff).item(), "std": np.std(all_m_diff).item()
                    }
        
        dump_json(bias_results, bias_analysis_dir / "prompt_bias_analysis.json")
        
        
        # self enhancement bias
        for metric in metrics_list:
            figure_path_list = []
            for initial_model_name in new_datasets_initial_models:
                initial_model_key = f"initial_model={initial_model_name}"
                
                # plot for prompt biases
                fig = plt.figure(figsize=[4, 3])
                plt.axhline(y=0, color="gray", linewidth=1.5)
                for baseline_idx, baseline_model_name in enumerate(baseline_models):
                    diff_list: list[float] = []
                    
                    for dataset_name in new_datasets_names:
                        # self detection
                        self_detection_result = performance_dict[dataset_name][initial_model_key].get(f"baseline_model={initial_model_name}")
                        if self_detection_result is None or len(self_detection_result["average"]) == 0:
                            raise Exception(f"Self detection result not found for {initial_model_name} ({dataset_name})")

                        # detection by another model
                        baseline_model_performance = performance_dict[dataset_name][initial_model_key].get(f"baseline_model={baseline_model_name}")
                        if baseline_model_performance is None or len(baseline_model_performance["average"]) == 0:
                            diff_list.append(None)
                        else:
                            diff_list.append(
                                (baseline_model_performance["average"]["metrics"][metric]["average"] - self_detection_result["average"]["metrics"][metric]["average"]) * 100
                            )
                    
                    x_values = [get_x_for_model(baseline_model_name) + idx for idx in range(len(diff_list))]
                    plt.plot(x_values, diff_list, label=convert_model_names_dict[baseline_model_name], marker=markers_dict[baseline_model_name], color=get_model_color(baseline_model_name), **plot_params)
                
                if metric == "false_negative_rate":
                    plt.ylim([-70, 70])
                if metric == "accuracy":
                    plt.ylim([-70, 70])
                
                figures_config(initial_model_name, metric, "self_enhancement")
                
                figure_path = output_dir / f"self_enhancement_{metric}_{initial_model_name.split('/')[-1]}.png"
                figure_path_list.append(figure_path)
                plt.savefig(figure_path, bbox_inches='tight')
                plt.close()
            
            # horizontally concatenate the figures
            join_figures_horizontally(figure_path_list, bias_analysis_dir / f"self_enhancement_{metric}.png")
