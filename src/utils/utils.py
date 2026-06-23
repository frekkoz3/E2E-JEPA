"""
Just some utility functions
"""

import argparse
import yaml
import math
import csv
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def flat_config(config : dict) -> dict:
    """Flattens a nested config file"""
    flat_config = {}
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                flat_config[key] = value
        else:
            flat_config[section] = params

    return flat_config

class MetricsCollector:
    """A class to collect metrics during training"""
    def __init__(self):
        self.metrics = []

    def add_metric(self, value: dict):
        """Adds a metric to the collector"""
        self.metrics.append(value)

    def get_metrics(self) -> dict:
        """Returns the collected metrics"""
        return self.metrics

    def save_metrics(self, where: str):
        """Saves the collected metrics to a csv file"""
        with open(where, 'w') as f:

            writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)


def plot_metrics(csv_path: str, save: bool):
    """
    Reads a CSV of training metrics and generates individual Plotly graphs for each loss,
    as well as a master HTML dashboard containing all plots with a unified legend.
    """
    csv_file = Path(csv_path)

    # 1. Validate File
    if not csv_file.exists():
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    # 2. Load Data
    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"The file {csv_path} is empty.")

    # 3. Prepare Output Directory
    if save:
        out_dir = csv_file.parent / "loss_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to: {out_dir}")

    x_values = df.index
    columns = df.columns
    num_metrics = len(columns)

    # 4. Setup Master Dashboard Grid
    grid_cols = 2 if num_metrics > 1 else 1
    grid_rows = math.ceil(num_metrics / grid_cols)
    subplot_titles = [col.replace("_", " ").title() for col in columns]

    master_fig = make_subplots(
        rows=grid_rows,
        cols=grid_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1
    )

    # Colors
    raw_color = 'rgba(31, 119, 180, 0.3)'   # Faded Blue
    ma_color = 'rgba(255, 20, 147, 1.0)'    # Deep Pink (The Pinkest Line)

    # 5. Generate Plots
    for i, column in enumerate(columns):
        clean_title = column.replace("_", " ").title()

        # Calculate dynamic moving average
        window_size = max(2, min(20, len(df) // 10))
        moving_avg = df[column].rolling(window=window_size, min_periods=1).mean()

        # --- A. Build Individual Plot ---
        indiv_fig = go.Figure()

        indiv_fig.add_trace(go.Scatter(
            x=x_values, y=df[column], mode='lines',
            name='Raw Data', line=dict(color=raw_color, width=1)
        ))
        indiv_fig.add_trace(go.Scatter(
            x=x_values, y=moving_avg, mode='lines',
            name=f'Moving Average (MA {window_size})', line=dict(color=ma_color, width=2.5)
        ))

        indiv_fig.update_layout(
            title=f"<b>{clean_title} Progression</b>",
            xaxis_title="Epoch / Update Step",
            yaxis_title="Loss Value",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )

        # --- B. Add to Master Dashboard ---
        row_pos = (i // grid_cols) + 1
        col_pos = (i % grid_cols) + 1

        # Only show the legend for the very first subplot to create a "Global Unique Legend"
        is_first = (i == 0)

        master_fig.add_trace(
            go.Scatter(x=x_values, y=df[column], mode='lines',
                       name='Raw Data', legendgroup='raw', showlegend=is_first,
                       line=dict(color=raw_color, width=1), hoverinfo="name+y"),
            row=row_pos, col=col_pos
        )
        master_fig.add_trace(
            go.Scatter(x=x_values, y=moving_avg, mode='lines',
                       name=f'Moving Average (MA)', legendgroup='ma', showlegend=is_first,
                       line=dict(color=ma_color, width=2.5), hoverinfo="name+y"),
            row=row_pos, col=col_pos
        )

        # Update subplot axes labels
        master_fig.update_xaxes(title_text="Epoch / Step", row=row_pos, col=col_pos)
        master_fig.update_yaxes(title_text="Loss", row=row_pos, col=col_pos)

        # --- C. Save Individual File ---
        if save:
            indiv_path = out_dir / f"{column}_plot.html"
            indiv_fig.write_html(str(indiv_path))
            print(f" -> Saved Individual Plot: {indiv_path.name}")
        else:
            indiv_fig.show()

    # 6. Finalize and Save Master Dashboard
    master_fig.update_layout(
        title="<b>E2E-JEPA Global Metrics Dashboard</b>",
        height=400 * grid_rows,
        template="plotly_white",
        hovermode="x unified",
        # Position the global legend horizontally at the top right, just above the charts
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    if save:
        master_path = out_dir / "all_metrics_dashboard.html"
        master_fig.write_html(str(master_path))
        print(f"\n >>> Saved Master Dashboard: {master_path.name} <<<")
    else:
        master_fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive Plotly graphs from training metrics CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the metrics.csv file.")
    parser.add_argument("--save", action="store_true", help="If flagged, saves plots as HTML files in a 'loss_plots' subfolder.")

    args = parser.parse_args()

    plot_metrics(args.csv, args.save)