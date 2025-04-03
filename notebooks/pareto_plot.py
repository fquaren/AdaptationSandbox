import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(metrics, base_path, n_clusters):

    # Initialize metrics dictionary
    

    # Iterate over clusters, each time excluding one
    for exclude_cluster in range(n_clusters):
        try:
            print(f"Processing Cluster {exclude_cluster} (excluded)")

            # Initialize the dictionary for this exclusion step
            metrics[exclude_cluster] = {}

            # Locate the excluded cluster's test loss file
            pattern = os.path.join(base_path, f"*cluster_{exclude_cluster}*")
            exp_paths = glob.glob(pattern)

            if not exp_paths:
                print(f"Warning: No data found for excluded cluster {exclude_cluster}. Skipping...")
                continue

            exp_path = exp_paths[0]
            mse_file = os.path.join(exp_path, f"cluster_{exclude_cluster}_test_losses.npy")

            if os.path.exists(mse_file):
                mean_mse = np.mean(np.load(mse_file))
                metrics[exclude_cluster]["mse"] = mean_mse
                print(f"Cluster {exclude_cluster} MSE: {mean_mse}")
            else:
                print(f"Warning: Test loss file missing for cluster {exclude_cluster}.")
                continue

            # Compute consistency error across all other clusters
            list_mean_mse = []
            for i in range(n_clusters):
                if i == exclude_cluster:
                    continue

                pattern = os.path.join(base_path, f"*cluster_{i}*")
                exp_paths = glob.glob(pattern)

                if not exp_paths:
                    print(f"Warning: No data found for cluster {i}. Skipping...")
                    continue

                cluster_path = exp_paths[0]
                loss_file = os.path.join(cluster_path, f"cluster_{i}_test_losses.npy")

                if os.path.exists(loss_file):
                    mean_loss_per_cluster = np.mean(np.load(loss_file))
                    list_mean_mse.append(mean_loss_per_cluster)
                else:
                    print(f"Warning: Test loss file missing for cluster {i}. Skipping...")

            # Compute and store the mean consistency error
            if list_mean_mse:
                metrics[exclude_cluster]["Error"] = np.mean(list_mean_mse)
                print(f"Cluster {exclude_cluster} Consistency Error: {metrics[exclude_cluster]['Error']}")

        except Exception as e:
            print(f"Skipping cluster {exclude_cluster} due to error: {e}")

    print(metrics)

    # # **Processing Overall Metrics**
    # try:
    #     overall_pattern = os.path.join(base_path, "*all*")
    #     overall_exp_paths = glob.glob(overall_pattern)

    #     if not overall_exp_paths:
    #         raise FileNotFoundError(f"No experiment folder matching pattern: {overall_pattern}")

    #     overall_exp_path = overall_exp_paths[0]  # Use first match
    #     overall_loss_file = os.path.join(overall_exp_path, "all_clusters_test_losses.npy")

    #     if not os.path.exists(overall_loss_file):
    #         raise FileNotFoundError(f"File not found: {overall_loss_file}")

    #     # Store overall metrics in a separate key
    #     metrics['overall'] = {
    #         "mse": np.mean(np.load(overall_loss_file)),
    #         "Error": None
    #     }

    #     exclude_cluster = None
    #     hist_mean_loss_per_cluster = []

    #     for i in range(n_clusters):
    #         if i == exclude_cluster:
    #             continue

    #         pattern = os.path.join(overall_exp_path, f"*cluster_{i}_test_losses.npy")
    #         loss_files = glob.glob(pattern)

    #         if not loss_files:
    #             print(f"Warning: No test loss file found for cluster {i}. Skipping...")
    #             continue

    #         loss_file_path = loss_files[0]
    #         mean_loss_per_cluster = np.mean(np.load(loss_file_path))
    #         hist_mean_loss_per_cluster.append(mean_loss_per_cluster)

    #     if hist_mean_loss_per_cluster:
    #         metrics['overall']["Error"] = np.mean(hist_mean_loss_per_cluster)

    #     # Print final metrics
    #     print("\nFinal Metrics:")
    #     print(metrics)
    # except:
    #     print("All not found.")

    return metrics

save_path = "/Users/fquareng/experiments"

metrics = {}
metrics = compute_metrics(metrics, base_path="/Users/fquareng/experiments", n_clusters=8)

metrics["finetuning"] = {}
metrics["finetuning"] = compute_metrics(metrics["finetuning"], base_path="/Users/fquareng/experiments/finetuning_frozen", n_clusters=8)

metrics["interpolation"] = {}
metrics["interpolation"]["linear"] = {}
metrics["interpolation"]["linear"]["mse"] = np.mean(np.load("/Users/fquareng/experiments/interpolation/linear/all_clusters_test_losses.npy"))
metrics["interpolation"]["quadratic"] = {}
metrics["interpolation"]["quadratic"]["mse"] = np.mean(np.load("/Users/fquareng/experiments/interpolation/quadratic/all_clusters_test_losses.npy"))
metrics["interpolation"]["cubic"] = {}
metrics["interpolation"]["cubic"]["mse"] = np.mean(np.load("/Users/fquareng/experiments/interpolation/cubic/all_clusters_test_losses.npy"))

# Extract MSE and Error values from the metrics dictionary
mse_values = {}
error_values = {}
clusters = {}
mse_values["on_single_cluster"] = [data["mse"] for data in metrics.values() if "mse" in data and "Error" in data]
error_values["on_single_cluster"] = [data["Error"] for data in metrics.values() if "mse" in data and "Error" in data]
clusters["on_single_cluster"] = [str(cluster) for cluster in metrics.keys() if "mse" in metrics[cluster] and "Error" in metrics[cluster]]

mse_values["finetuning"] = [data["mse"] for data in metrics["finetuning"].values() if "mse" in data and "Error" in data]
error_values["finetuning"] = [data["Error"] for data in metrics["finetuning"].values() if "mse" in data and "Error" in data]
clusters["finetuning"] = ["ft"+str(cluster) for cluster in metrics["finetuning"].keys() if "mse" in metrics["finetuning"][cluster] and "Error" in metrics["finetuning"][cluster]]
mean_mse_finetuning = np.mean(mse_values["finetuning"])
mean_error_finetuning = np.mean(error_values["finetuning"])
clusters["finetuning"].append("Finetuning")

mse_values["interpolation"] = [data["mse"] for data in metrics["interpolation"].values() if "mse" in data and "Error" in data]
error_values["interpolation"] = [data["Error"] for data in metrics["interpolation"].values() if "mse" in data and "Error" in data]
clusters["interpolation"] = [str(cluster) for cluster in metrics["interpolation"].keys() if "mse" in metrics["interpolation"][cluster] and "Error" in metrics["interpolation"][cluster]]

# Create scatter plot
plt.figure(figsize=(10, 10))
# Plot vertical line for overall MSE
overall_pattern = os.path.join(save_path, "*all*")
overall_exp_paths = glob.glob(overall_pattern)[0]
overall_loss_file = os.path.join(overall_exp_paths, "all_clusters_test_losses.npy")
plt.axvline(np.mean(np.load(overall_loss_file)), color='k', linestyle='--', label='Overall MSE')
plt.axvline(np.mean(metrics["interpolation"]["linear"]["mse"]), color='r', linestyle=':', label='Linear MSE')
plt.axvline(np.mean(metrics["interpolation"]["quadratic"]["mse"]), color='g', linestyle=':', label='Quadratic MSE')
plt.axvline(np.mean(metrics["interpolation"]["cubic"]["mse"]), color='b', linestyle=':', label='Cubic MSE')

plt.scatter(mse_values["on_single_cluster"], error_values["on_single_cluster"], color='b', label='On single cluster', alpha=1, marker="x", s=100)
plt.scatter(mse_values["finetuning"], error_values["finetuning"], color='g', label='Finetuning', alpha=0.25, marker="o", s=100)
plt.scatter(mean_mse_finetuning, mean_error_finetuning, color='g', label='Mean Finetuning', alpha=1, marker="o", s=100)
plt.scatter(mse_values["interpolation"], error_values["interpolation"], color='r', label='Interpolation', alpha=1, marker="*", s=100)

# # Plot trend line using NumPy for efficiency
# x_vals = np.linspace(min(mse_values), max(mse_values), 100)
# y_vals = -1/6 * x_vals + 1.179
# plt.plot(x_vals, y_vals, label="y = -1/6x + 1.179", linestyle="--", color="r", alpha=0.6)

# Add labels and title
plt.xlabel('Mean Squared Error (MSE) [K]')
plt.ylabel('Consistency Error [K]')
# plt.xlim((0,2))
# plt.ylim((0,2))
plt.grid(True)

# Annotate each cluster
for i, cluster in enumerate(clusters["finetuning"]):
    try:
        plt.annotate(cluster, (mse_values["finetuning"][i], error_values["finetuning"][i]), textcoords="offset points", xytext=(7, 7), ha='center')
    except Exception as e:
        print(f"Skipping annotation for cluster {cluster} due to error: {e}")
for i, cluster in enumerate(clusters["interpolation"]):
    try:
        plt.annotate(cluster, (mse_values["interpolation"][i], error_values["interpolation"][i]), textcoords="offset points", xytext=(7, 7), ha='center')
    except Exception as e:
        print(f"Skipping annotation for cluster {cluster} due to error: {e}")
for i, cluster in enumerate(clusters["on_single_cluster"]):
    try:
        plt.annotate(cluster, (mse_values["on_single_cluster"][i], error_values["on_single_cluster"][i]), textcoords="offset points", xytext=(7, 7), ha='center')
    except Exception as e:
        print(f"Skipping annotation for cluster {cluster} due to error: {e}")

plt.legend()
plt.savefig(os.path.join(save_path,"pareto_plot"))