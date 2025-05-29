"""
Count number of anomalies detected from previous step (detect_anomalies).
"""

import os

from das_anomaly import search_keyword_in_files
from das_anomaly.settings import SETTINGS


results_path = SETTINGS.RESULTS_PATH
results_folder_name = SETTINGS.RESULT_FOLDER_NAME

keyword = "anomaly"

directory = os.path.join(results_path, results_folder_name)

total_count, lines = search_keyword_in_files(directory, keyword)

with open(results_path + f"{keyword}_" + results_folder_name + ".txt", "w") as file:
    for line in lines:
        file.write(f"{line}\n")

print(f"Total number of detected anomalies with '{keyword}' keyword:", total_count)
