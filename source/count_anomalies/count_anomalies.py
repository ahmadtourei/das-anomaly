"""
Count number of anomalies from the result (saved) text files.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '..')
sys.path.append(source_dir)
from utils import search_keyword_in_files


root = '/globalscratch/ahmad9/caserm/spectrum_analysis/results/' 
result_folder = 'UTC-YMD20220609-HMS124917.291'

keyword = 'anomaly'  

directory = os.path.join(root, result_folder)

total_count, lines = search_keyword_in_files(directory, keyword)

with open(root + 'anomalies_' + result_folder + '.txt', 'w') as file:
    for line in lines:
        file.write(f"{line}\n")

print(f"Total occurrences of the '{keyword}':", total_count)
