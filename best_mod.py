import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the folders
data_dir = "eval/"

# Initialize a list to collect all stats
all_stats = []

# Walk through the directory structure to find stats.csv files
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file == "stats.csv":
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            chunk_size, text_percentage, distance_metric = folder_name.split("_")

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Add metadata columns to the DataFrame
            df["chunk_size"] = int(chunk_size)
            df["text_percentage"] = int(text_percentage)
            df["distance_metric"] = distance_metric

            # Append to the list
            all_stats.append(df)

# Combine all data into a single DataFrame
combined_df = pd.concat(all_stats, ignore_index=True)

# Define metrics for analysis
metrics = {
    "success_rate": ("max", "Success Rate", "lightgreen"),
    "average_time": ("min", "Average Time (s)", "lightskyblue"),
    "average_mrr": ("max", "Mean Reciprocal Rank (MRR)", "orange"),
    "carbon_emissions(g_CO2eq)": ("min", "CO2 Emissions (g)", "salmon"),
}

# Initialize a dictionary to store the best models for each metric
best_models = {}

# Analyze and plot for each metric
for metric, (method, metric_label, color) in metrics.items():
    if method == "max":
        best_row = combined_df.loc[combined_df[metric].idxmax()]
    else:  # method == "min"
        best_row = combined_df.loc[combined_df[metric].idxmin()]

    best_models[metric] = best_row

    # Group by encoder and calculate mean and std deviation
    grouped = combined_df.groupby("encoder")[metric].agg(["mean", "std"]).sort_values("mean")

    # Plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(
        grouped.index, grouped["mean"], yerr=grouped["std"], capsize=5, color=color, alpha=0.8
    )
    plt.title(f"Best Model for {metric_label}")
    plt.ylabel(metric_label)
    plt.xlabel("Encoder")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Annotate each bar with the mean value
    for bar, value in zip(bars, grouped["mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="left",
            va="bottom",
            fontsize=10,
        )

    # Save the plot
    plot_path = os.path.join(data_dir, f"best_model_{metric}.png")
    plt.savefig(plot_path)
    plt.close()

# Output the best models
for metric, best_row in best_models.items():
    print(f"Best model for {metric}:\n{best_row}\n")
