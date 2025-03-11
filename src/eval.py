import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from train import *

output_dir = "/scratch/fquareng/experiments/UNet8x-baseline-T2M/7_50713959_99ee564a"
model = UNet8x()
batch_size = 5
variable = "T_2M"
data_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data"
elevation_dir = os.path.join(data_path, "dem_squares")
input_dir = os.path.join(data_path, f"1h_2D_sel_cropped_gridded_clustered_threshold_blurred/cluster_7")
target_dir = os.path.join(data_path, f"1h_2D_sel_cropped_gridded_clustered_threshold/cluster_7")
_, _, test_loader = get_dataloaders(variable, input_dir, target_dir, elevation_dir, batch_size)

print("Loading best model...")
best_model_path = os.path.join(output_dir, f"best_model.pth")
model.load_state_dict(torch.load(best_model_path))
model.to("cuda")
print("Model loaded")
print("Evaluating model...")
evaluate_and_plot(model, test_loader, output_dir)