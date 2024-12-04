import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your repository
repo_path = "eval/"

# Define the output directory for plots
output_dir = "eval/summary_plots/"
os.makedirs(output_dir, exist_ok=True)

# Initialize an empty DataFrame to store data from all folders
all_data = []

# Loop through the folders in the repository
for folder in os.listdir(repo_path):
    folder_path = os.path.join(repo_path, folder)
    if os.path.isdir(folder_path):
        # Extract metadata from the folder name
        try:
            chunk_size, text_percentage, distance_metric = folder.split("_")
            
            # Read the stats.csv file
            csv_path = os.path.join(folder_path, "stats.csv")
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                
                # Add metadata to the data
                data["chunk_size"] = int(chunk_size)
                data["text_percentage"] = int(text_percentage)
                data["distance_metric"] = distance_metric
                
                # Append to the all_data list
                all_data.append(data)
        except ValueError:
            print(f"Skipping folder with unexpected name format: {folder}")

# Concatenate all data into a single DataFrame
if all_data:
    df = pd.concat(all_data, ignore_index=True)
else:
    print("No data found. Check your repository structure and file paths.")
    exit()

# Convert categorical variables to appropriate types
df["distance_metric"] = df["distance_metric"].astype("category")

# Define the metrics to evaluate
metrics = ["average_time", "success_rate", "average_mrr", "carbon_emissions(g_CO2eq)"]

# Function to plot the impact of a variable on a given metric and save the plot
def plot_impact(variable, metric):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=variable, y=metric, palette="Set3")
    plt.title(f"Impact of {variable} on {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"impact_of_{variable}_on_{metric}.png")
    plt.savefig(plot_path)
    plt.close()

# Evaluate the impact of chunk_size, text_percentage, and distance_metric on each metric
for variable in ["chunk_size", "text_percentage", "distance_metric"]:
    for metric in metrics:
        plot_impact(variable, metric)

# Summary statistics grouped by the variables of interest
summary = {}
for variable in ["chunk_size", "text_percentage", "distance_metric"]:
    summary[variable] = df.groupby(variable)[metrics].mean()

# Print summary statistics
for variable, stats in summary.items():
    print(f"\nSummary statistics grouped by {variable}:")
    print(stats)

# Save summary statistics to CSV
for variable, stats in summary.items():
    stats.to_csv(f"summary_by_{variable}.csv")