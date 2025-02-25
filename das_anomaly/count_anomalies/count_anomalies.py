"""
Count number of anomalies from the result (saved) text files.
"""

import os

from das_anomaly import search_keyword_in_files

root = "/globalscratch/ahmad9/caserm/spectrum_analysis/results/"
result_folder = "UTC-YMD20220609-HMS124917.291"

keyword = "anomaly"

directory = os.path.join(root, result_folder)

total_count, lines = search_keyword_in_files(directory, keyword)

with open(root + "anomalies_" + result_folder + ".txt", "w") as file:
    for line in lines:
        file.write(f"{line}\n")

print(f"Total occurrences of the '{keyword}':", total_count)
