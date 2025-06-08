import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd
import numpy as np


class VisualizationServices:
    def __init__(self, dataframe, query_intent):
        self.dataframe = dataframe
        self.query_intent = query_intent

    def execute(self):
        action = self.query_intent.get("action")
        column = self.query_intent.get("column")

        print(action, column)
        if not action or not column:
            raise ValueError("Could not understand the intent or column.")

        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found.")

        # Numeric checks for mean and sum
        if action in ["mean", "sum"]:
            if not pd.api.types.is_numeric_dtype(self.dataframe[column]):
                raise ValueError(f"Column '{column}' must be numeric for {action} operation.")

        fig, file_message = None, None
        if action == "mean":
            fig, file_message = self.plot_mean(column, save=True)
        # commenting this as only KPI is good
        # elif action == "sum":
        #     fig, file_message = self.plot_sum(column, save=False)
        # commenting this as only KPI is good
        # elif action == "count":
        #     fig, file_message = self.plot_count(column, save=True)
        elif action == "filter":
            fig, file_message = self.plot_filter(column, save=True)
        # else:
        #     fig , file_message = None, None
        #     # raise ValueError(f"Action '{action}' is not supported yet")

        return fig, file_message

    def plot_mean(self, column, save=False):
        data = self.dataframe[column]
        mean_val = data.mean()
        chunk_size = self.get_chunk_size(column=column)
        data_agg = data.groupby(np.arange(len(data)) // chunk_size).mean()
        # print("Chunk size: ", chunk_size)

        # print(data_agg)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(data_agg.index, data_agg.values, linestyle='-')
        ax.axhline(mean_val, color='r', linestyle='--', label='Overall Mean')

        # ax.bar(len(data), mean_val, color='red', label='Mean')
        # ax.set_xticks(list(range(len(data))) + [len(data)], list(range(len(data))) + ['Mean'])

        yticks = list(ax.get_yticks())

        # Add mean_val to yticks if not already present
        if mean_val not in yticks:
            yticks.append(mean_val)
            yticks = sorted(yticks)

        ax.set_yticks(yticks)

        # Create custom ytick labels, mark the mean_val label differently
        # yticklabels = [f'{tick:.2f}' if tick != mean_val else f'{tick:.2f} ← Mean' for tick in yticks]
        # ax.set_yticklabels(yticklabels, color='black')
        ax.set_yticklabels([f'{tick:.2f}' for tick in yticks], fontsize=8)
        ax.set_title(f'Mean of {column} (Line Chart)')
        ax.set_ylabel(f'{column}')
        ax.set_xlabel('Chunk Index')
        ax.set_xticks([])
        ax.legend()

        file_message = None
        if save:
            filename = self.save_fig_with_timestamp(fig, prefix=f'mean_{column}')
            file_message = f"Chart is saved to {filename}"

        return fig, file_message

    def plot_sum(self, column, save=False):
        sum_val = self.dataframe[column].sum()

        fig, ax = plt.subplots(figsize=(8,4))
        # Simple bar with one bar showing sum
        ax.bar([0], [sum_val], color='purple')
        ax.set_xticks([0], [f'Sum of {column}'])
        ax.set_title(f'Sum of {column}')
        ax.set_ylabel('Sum')

        file_message = None
        if save:
            filename = self.save_fig_with_timestamp(fig, prefix=f'sum_{column}')
            file_message = f"Chart is saved to {filename}"

        return fig, file_message

    def plot_count(self, column, save=False):
        count_val = self.dataframe[column].count()

        fig, ax = plt.subplots(figsize=(8,4))
        # Count is a single number, show as bar
        ax.bar([0], [count_val], color='green')
        ax.set_xticks([0], [f'Count of {column}'])
        ax.set_title(f'Count of non-null {column} entries')
        ax.set_ylabel('Count')

        file_message = None
        if save:
            filename = self.save_fig_with_timestamp(fig, prefix=f'count_{column}')
            file_message = f"Chart is saved to {filename}"

        return fig, file_message

    def plot_filter(self, column, save=False):
        unique_vals, counts = self.dataframe[column].value_counts().index, self.dataframe[column].value_counts().values

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(len(unique_vals)), counts)
        ax.set_xticks(range(len(unique_vals)), unique_vals, rotation=45, ha='right')
        ax.set_title(f'Unique values count in {column}')
        ax.set_ylabel('Frequency')
        fig.tight_layout()

        file_message = None
        if save:
            filename = self.save_fig_with_timestamp(fig, prefix=f'filter_{column}')
            file_message = f"Chart is saved to {filename}"

        return fig, file_message

    def save_fig_with_timestamp(self, fig, prefix="chart"):
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(filename, bbox_inches='tight')
        return filename

    def freedman_daiconis_bin_width(self, column):
        q75, q25 = np.percentile(list(self.dataframe[column]), [75, 25])
        iqr = q75 - q25
        n = len(self.dataframe[column])
        bin_width = 2 * iqr / (n ** (1/3))
        return bin_width

    def get_chunk_size(self, column):
        bin_width = self.freedman_daiconis_bin_width(column=column)
        if bin_width == 0 :
            return 1
        data_range = self.dataframe[column].max() - self.dataframe[column].min()
        num_bins = max(1, int(np.ceil(data_range / bin_width)))
        chunk_size = max(1, len(self.dataframe[column]) // num_bins)
        return chunk_size
