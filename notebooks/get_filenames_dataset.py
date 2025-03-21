import os
import re
from collections import defaultdict

def get_selected_files(file_list):
    # Sort files to ensure chronological order
    file_list.sort()
    
    # Group files by date
    date_pattern = re.compile(r"(\d{8})(\d{2})\d{4}")
    files_by_date = defaultdict(list)
    
    for file in file_list:
        match = date_pattern.search(file)
        if match:
            date, hour = match.groups()
            files_by_date[date].append((hour, file))
    
    # Sort each day's files by hour
    for date in files_by_date:
        files_by_date[date].sort()
    
    selected_files = []
    shift = 0  # Initial shift value
    
    sorted_dates = sorted(files_by_date.keys())
    
    for date in sorted_dates:
        selected_hours = [
            (0 + shift) % 24,
            # (3 + shift) % 24, 
            # (6 + shift) % 24,
            (8 + shift) % 24,  # once every 9 days
            # (9 + shift) % 24,
            # (12 + shift) % 24,
            (15 + shift) % 24,
            # (16 + shift) % 24,
            # (18 + shift) % 24,
            # (21 + shift) % 24,
        ]
        day_files = files_by_date[date]
        
        selected_day_files = [f for h, f in day_files if int(h) in selected_hours]
        
        if len(selected_day_files) == 3:
            selected_files.extend(selected_day_files)
        
        shift = (shift + 1) % 24  # Increase shift for the next day
    
    return selected_files

# Example usage:
file_list = os.listdir("/users/fquareng/data/1h_2D_sel")  # Modify with the correct path
selected_files = get_selected_files(file_list)

# Print selected files
for f in selected_files:
    print(f)

