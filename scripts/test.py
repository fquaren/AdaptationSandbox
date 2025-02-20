import time
import os

# Record the start time
start_time = time.time()

# Run a loop that continues until 5 seconds have passed
while time.time() - start_time < 5:
    print("Running...")  # You can replace this with any task
    time.sleep(1)  # Sleep for 1 second before printing again