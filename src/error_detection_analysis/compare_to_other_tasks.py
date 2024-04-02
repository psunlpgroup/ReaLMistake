from easy_io import read_json, dump_json, dump_str_list_to_txt_file
import matplotlib.pyplot as plt
import scipy.stats
from adjustText import adjust_text

from src.path import baseline_performance_dir, baseline_analysis_dir
from src.config import new_datasets_names, new_datasets_initial_models, baseline_models_open, baseline_models_closed, \
    covnert_dataset_full_name_dict, convert_model_names_short, covnert_dataset_name_dict


makers_dict = {
    "google/gemma-7b-it": "Ge7B",
    "meta-llama/Llama-2-13b-chat-hf": "L13B", "meta-llama/Llama-2-70b-chat-hf": "L70B",
    "mistralai/Mistral-7B-Instruct-v0.1": "M7B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "M8x7B",
    "Qwen/Qwen1.5-14B-Chat": "Q14B", "Qwen/Qwen1.5-72B-Chat": "Q72B",
    "gpt-3.5-turbo-0125": "GPT3.5",
    "models/gemini-1.0-pro-001": "Ge1.0",
    "claude-3-opus-20240229": "Claude3",
    "gpt-4-0613": "GPT4$_{23}$", "gpt-4-0125-preview": "GPT4$_{24}$",
}

output_dir = baseline_analysis_dir / "other_tasks_analysis_results"

if __name__ == "__main__":
    # make matplotlib text larger
    plt.rcParams.update({'font.size': 15})
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_performance = read_json(baseline_performance_dir / "simple_prompt_baseline/performance.json")
    baseline_models_list = baseline_models_open + baseline_models_closed
    
    for other_task in ["elo_rating", "mmlu"]:
        other_task_performance: dict[str, dict] = read_json(baseline_analysis_dir / f"other_tasks_performance/{other_task}.json")["metric"]
        
        correlation_dict = {}
        for initial_model in new_datasets_initial_models:
            for dataset_name in new_datasets_names:
                fig = plt.figure(figsize=[6, 3.5])

                all_x = []
                all_y = []
                texts = []

                for metric in ["recall", "precision"]:
                    x = []
                    y = []
                    baseline_mdoels_src_for_this_metric = []
                    for baseline_model in baseline_models_list:
                        if baseline_model not in other_task_performance.keys():
                            continue
                        
                        baseline_model_key = f"baseline_model={baseline_model}"
                        if baseline_performance[dataset_name][f"initial_model={initial_model}"].get(baseline_model_key) is not None:
                            x.append(other_task_performance[baseline_model])
                            y.append(baseline_performance[dataset_name][f"initial_model={initial_model}"][baseline_model_key]["average"]["metrics"][metric]["average"])
                            baseline_mdoels_src_for_this_metric.append(convert_model_names_short[baseline_model])
                    
                        plt.plot(x[-1:], y[-1:], color={"precision": "red", "recall": "blue"}[metric], marker={"precision": "o", "recall": "^"}[metric],
                                markersize=9, linestyle="None")
                                # label={"precision": "Precision", "recall": "Recall"}[metric]
                        
                        texts.append(
                            plt.text(x[-1], y[-1], makers_dict[baseline_model], color={"precision": "red", "recall": "blue"}[metric], alpha=0.6,
                                 fontsize=11, horizontalalignment='center', verticalalignment='center')
                        )
                    
                    all_x += x
                    all_y += y

                    # pearson's correlation and spearman's correlation
                    corr = {}
                    pearson, p_p = scipy.stats.pearsonr(x, y)
                    corr["pearson"] = {"correlation": pearson, "p": p_p}
                    spearman, p_s = scipy.stats.spearmanr(x, y)
                    corr["spearman"] = {"correlation": spearman, "p": p_s}
                    
                    correlation_dict.setdefault(initial_model, {}).setdefault(dataset_name, {})[metric] = corr
                                
                if other_task == "elo_rating":
                    plt.xlim([985, 1280])
                if other_task == "mmlu":
                    plt.xlim([50, 90])
                plt.ylim([.03, 1.05])
                
                adjust_text(texts, all_x, all_y, expand=(1.3, 1.8), ensure_inside_axes=True)
                
                other_task_name = {"elo_rating": "LMSYS Chatbot Arena Elo Rating", "mmlu": "Accuracy on MMLU [%]"}[other_task]
                plt.xlabel(other_task_name)
                plt.title(f"{covnert_dataset_full_name_dict[dataset_name]}", y=1.05)

                # if dataset_name == "math_word_problem_generation":
                #     plt.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{other_task}_{initial_model.split('/')[-1]}_{dataset_name}.pdf")
    
        dump_json(correlation_dict, output_dir / f"{other_task}_correlation.json")
        
        # create table
        table_list: list[str] = []
        for initial_model in correlation_dict.keys():
            for dataset_name in correlation_dict[initial_model].keys():
                row = ["", covnert_dataset_name_dict[dataset_name]]
                for pr in ["precision", "recall"]:
                    for cr in ["pearson", "spearman"]:
                        correlation = correlation_dict[initial_model][dataset_name][pr][cr]
                        
                        val_str = f"${correlation['correlation']:.2f}$"
                        if pr == "recall" and correlation["correlation"] >= 0:
                            val_str = "\phantom{$-$}" + val_str
                        
                        if correlation["p"] < 0.05:
                            val_str += "$^*$"
                        else:
                            val_str += "\phantom{$^*$}"
                        row.append(val_str)
                table_list.append(" & ".join(row) + " \\\\")
        dump_str_list_to_txt_file(table_list, output_dir / f"{other_task}_correlation_table.txt")
