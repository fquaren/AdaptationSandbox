import matplotlib.pyplot as plt
import re

# Path to your loss data file
log_file = "/Users/fquareng/phd/AdaptationSandbox/figures/50639351"

# Lists to store parsed losses
train_losses = []
val_losses = []

# Regular expressions for extracting loss values
train_loss_pattern = re.compile(r"Batch Loss: (\d+\.\d+)")
val_loss_pattern = re.compile(r"Val Batch Loss: (\d+\.\d+)")

# Read and parse the log file
with open(log_file, "r") as file:
    for line in file:
        train_match = train_loss_pattern.search(line)
        val_match = val_loss_pattern.search(line)

        if train_match:
            train_losses.append(float(train_match.group(1)))
        if val_match:
            val_losses.append(float(val_match.group(1)))

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linestyle="dashed")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()