"""
Count number of anomalies from the saved results from previous step (detect_anomalies).
"""

import os

from das_anomaly import search_keyword_in_files

root = "/path/to/saved/results/from/detect_anomalies/"
result_folder_name = "result_folder"

keyword = "anomaly"

directory = os.path.join(root, result_folder_name)

total_count, lines = search_keyword_in_files(directory, keyword)

with open(root + "anomalies_" + result_folder_name + ".txt", "w") as file:
    for line in lines:
        file.write(f"{line}\n")

print(f"Total number of detected anomalies with '{keyword}' keyword:", total_count)
