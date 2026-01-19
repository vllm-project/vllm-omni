import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_benchmark_results(csv_source, output_dir="."):
    """
    Reads CSV data and plots three comparison line charts:
    1. QPS vs Throughput
    2. QPS vs Avg TTFT (in Seconds)
    3. QPS vs P99 TTFT (in Seconds)

    Args:
        csv_source (str): CSV string content or file path.
        output_dir (str): Directory to save the generated images.
    """
    # 1. Load data
    if isinstance(csv_source, str) and "\n" in csv_source:
        # If input is a CSV string
        df = pd.read_csv(io.StringIO(csv_source))
    else:
        # If input is a file path
        df = pd.read_csv(csv_source)

    # 2. Data Preprocessing: Convert ms to s
    if "avg_ttft_ms" in df.columns:
        df["avg_ttft_s"] = df["avg_ttft_ms"] / 1000.0
    if "p99_ttft_ms" in df.columns:
        df["p99_ttft_s"] = df["p99_ttft_ms"] / 1000.0

    # 3. Set plotting style
    sns.set_theme(style="whitegrid", context="talk")
    # Custom palette: Baseline (Gray), ModalityAware (Red)
    custom_palette = {"Baseline": "#7f7f7f", "ModalityAware": "#d62728"}

    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Define metrics configuration
    # Format: (Y_column, Title, Y_label, Filename)
    metrics_config = [
        ("throughput_tokens_per_s", "Throughput vs QPS", "Throughput (tokens/s)", "throughput_vs_qps.png"),
        ("avg_ttft_s", "Average TTFT vs QPS", "Avg TTFT (s)", "avg_ttft_vs_qps.png"),
        ("p99_ttft_s", "P99 TTFT vs QPS", "P99 TTFT (s)", "p99_ttft_vs_qps.png"),
    ]

    print(f"{'=' * 30}\n Generating Plots...\n{'=' * 30}")

    # 4. Loop to generate plots
    for y_col, title, y_label, filename in metrics_config:
        plt.figure(figsize=(10, 6))

        # Check if column exists to avoid errors
        if y_col not in df.columns:
            print(f"Warning: Column {y_col} not found in data, skipping {filename}.")
            continue

        # Plot line chart
        sns.lineplot(
            data=df,
            x="target_qps",
            y=y_col,
            hue="scenario",
            style="scenario",
            markers=True,
            dashes=False,
            palette=custom_palette,
            linewidth=3,
            markersize=10,
        )

        # Plot details
        plt.title(title, fontsize=18, pad=15, fontweight="bold")
        plt.xlabel("Target QPS", fontsize=14)
        plt.ylabel(y_label, fontsize=14)

        # Set X-axis ticks to match data points exactly
        if "target_qps" in df.columns:
            plt.xticks(sorted(df["target_qps"].unique()))

        # Legend adjustment
        plt.legend(title="Scheduler Scenario", loc="best")

        # Layout adjustment
        plt.tight_layout()

        # Save file
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f" Saved chart to: {save_path}")

        # Show plot
        plt.show()


if __name__ == "__main__":
    # python3 vllm-omni/benchmarks/modality_aware_test/plot_benchmark.py
    csv_data = "vllm-omni/benchmarks/modality_aware_test/benchmark_results.csv"
    output_dir = "vllm-omni/benchmarks/modality_aware_test/benchmark_plots"
    plot_benchmark_results(csv_data, output_dir=output_dir)
