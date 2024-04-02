from easy_io import read_json

import numpy as np
import matplotlib.pyplot as plt

from src.path import baseline_analysis_dir
from src.config import new_datasets_names, convert_model_names_dict, covnert_dataset_full_name_dict


# Redefine the function to add value labels to put them in the center of each bar segment
def add_value_labels_center(ax, data):
    for i, (rect, val) in enumerate(zip(ax.patches, data.flatten())):
        # Get the center of the bar segment
        rect_center = rect.get_x() + rect.get_width() / 2
        y = rect.get_y() + rect.get_height() / 2

        # Only add text if the bar segment is large enough
        if rect.get_width() > 0:
            label = f"{val:.0f}"
            ax.text(rect_center, y, label, ha='center', va='center', fontsize=18)


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})
    
    manual_analysis_stat: dict[str, dict[str, dict[str, float]]] = read_json(baseline_analysis_dir / "manual_analysis" / "manual_analysis_stat.json")
    
    for dataset_name in new_datasets_names:
        # convert data
        stat_array = []
        for baseline_name, annotations in manual_analysis_stat[dataset_name].items():
            stat_row = []
            for key, value in annotations.items():
                stat_row.append(value / sum(annotations.values()) * 100)
            stat_array.append(stat_row)
        stat_array = np.array(stat_array)
        stat_array_transposed = stat_array.T

        # Recreate the figure and axes for the centered value labels
        fig, ax = plt.subplots(figsize=(10, 4))

        # Reset the starting point for the bars
        baseline_models_list = [convert_model_names_dict[name] for name in list(manual_analysis_stat[dataset_name].keys())]
        left = np.zeros(len(baseline_models_list))

        # Plotting the bars
        colors = ['dodgerblue', 'skyblue', 'palevioletred', "white", "white", 'lightgray', 'gray']
        for idx, row in enumerate(stat_array_transposed):
            bars = ax.barh(baseline_models_list, row, left=left, color=colors[idx], hatch='/' if idx == 4 else None)
            left += row

        # Adding value labels after all bars have been placed
        add_value_labels_center(ax, stat_array_transposed)
        
        ax.set_xticks([])
        ax.tick_params(axis='y', which='both', left=False)

        # remove lines around the figure
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.title(covnert_dataset_full_name_dict[dataset_name], fontsize=25)
        plt.tight_layout(pad=0)

        # Save the figure
        fig.savefig(baseline_analysis_dir / "manual_analysis" / f"manual_analysis_{dataset_name}.pdf")
