import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    filename="file_copy.log",  # Log file name
    level=logging.INFO,  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the directories
old_clusters_dir = Path("/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_clustered_kmeans")  # e.g., "1h_2D_sel_cropped_blurred_x8_clustered_kmeans"
new_data_dir = Path("/Users/fquareng/data/1h_2D_sel_cropped_gridded")  # e.g., "1h_2D_sel_cropped_gridded"
new_clusters_dir = Path("/Users/fquareng/data/1h_2D_sel_cropped_clustered_kmeans")  # e.g., "1h_2D_sel_cropped_clustered_kmeans"

# Function to process each file
def process_file(cluster_number, ix_iy, timestamp, new_ix_iy_dir):
    try:
        # Find the file with the same timestamp (YYYYMMDDhhmmss)
        matching_files = [f for f in new_ix_iy_dir.iterdir() if f.suffix == ".nz" and timestamp in f.name]
        if matching_files:
            new_file = matching_files[0]  # Pick the first match
            cluster_subdir = new_clusters_dir / f"cluster_{cluster_number}"
            cluster_subdir.mkdir(parents=True, exist_ok=True)  # Create the cluster directory if it doesn't exist

            target_path = cluster_subdir / new_file.name
            # Copy the file to the correct cluster directory
            shutil.copy(new_file, target_path)
            logging.info(f"Copied {new_file} to {target_path}")
        else:
            logging.warning(f"Timestamp {timestamp} not found in {new_ix_iy_dir}")
    except PermissionError as e:
        logging.error(f"Permission error while processing {new_ix_iy_dir}: {e}")
    except FileNotFoundError as e:
        logging.error(f"File not found while processing {new_ix_iy_dir}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while processing {new_ix_iy_dir}: {e}")

# Function to process each cluster
def process_cluster(cluster_dir):
    if cluster_dir.is_dir() and cluster_dir.name.startswith("cluster_"):
        cluster_number = cluster_dir.name.split("_")[1]  # Extract X from cluster_X
        logging.info(f"Processing cluster {cluster_number}...")

        # Iterate over the files in each cluster subdirectory (e.g., 0_10_lffd20101110150000_blurred_x8.nz)
        futures = []
        with ThreadPoolExecutor() as executor:
            for old_file in cluster_dir.iterdir():
                if old_file.suffix == ".nz":  # Process only .nz files
                    filename = old_file.name
                    # Extract the ix_iy and timestamp (YYYYMMDDhhmmss) from the filename
                    ix_iy, timestamp = filename.split("_")[:2], filename.split("_")[2].replace("lffd", "")
                    ix_iy = "_".join(ix_iy)  # e.g., "0_10"

                    # Build the path for the corresponding ix_iy in the new data directory
                    new_ix_iy_dir = new_data_dir / ix_iy
                    if new_ix_iy_dir.exists():
                        # Submit the file copy task to the executor
                        future = executor.submit(process_file, cluster_number, ix_iy, timestamp, new_ix_iy_dir)
                        futures.append(future)

            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()  # This will re-raise any exceptions caught during processing

# Main execution
def main():
    # Iterate through each cluster directory in the old data (1h_2D_sel_cropped_blurred_x8_clustered_kmeans)
    with ThreadPoolExecutor() as executor:
        futures = []
        for cluster_dir in old_clusters_dir.iterdir():
            futures.append(executor.submit(process_cluster, cluster_dir))

        # Wait for all cluster processing to complete
        for future in as_completed(futures):
            future.result()  # This will re-raise any exceptions caught during processing

if __name__ == "__main__":
    main()