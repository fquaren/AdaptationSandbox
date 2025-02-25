import matplotlib.pyplot as plt

# Path to your loss data file
loss_file = "/scratch/fquareng/outputs/UNet-Baseline/50639351"

# Lists to store parsed losses
train_losses = []
val_losses = []

# Read and parse the loss file
with open(loss_file, "r") as file:
    for line in file:
        parts = line.strip().split(": ")
        if len(parts) == 2:
            key, value = parts
            loss_value = float(value)
            if "Batch Loss" in key:
                train_losses.append(loss_value)
            elif "Val Batch Loss" in key:
                val_losses.append(loss_value)

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s", linestyle="dashed")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()