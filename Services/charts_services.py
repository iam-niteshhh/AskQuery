import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd


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
        elif action == "sum":
            fig, file_message = self.plot_sum(column, save=True)
        elif action == "count":
            fig, file_message = self.plot_count(column, save=True)
        elif action == "filter":
            fig, file_message = self.plot_filter(column, save=True)
        else:
            raise f"Action '{action}' is not supported yet"

        return fig, file_message

    def plot_mean(self, column, save=False):
        data = self.dataframe[column]
        mean_val = data.mean()

        fig, ax = plt.subplots()
        ax.bar(range(len(data)), data, label='Data Points')
        ax.bar(len(data), mean_val, color='red', label='Mean')
        ax.set_xticks(list(range(len(data))) + [len(data)], list(range(len(data))) + ['Mean'])
        ax.set_title(f'Mean of {column}')
        ax.set_ylabel('Value')
        ax.legend()

        file_message = None
        if save:
            filename = self.save_fig_with_timestamp(fig, prefix=f'mean_{column}')
            file_message = f"Chart is saved to {filename}"

        return fig, file_message

    def plot_sum(self, column, save=False):
        sum_val = self.dataframe[column].sum()

        fig, ax = plt.subplots()
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

        fig, ax = plt.subplots()
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

        fig, ax = plt.subplots()
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