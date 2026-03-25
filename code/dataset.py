import os
import csv

import csv
from pathlib import Path

input_folder = Path('./../dataset/KITTI/rgb')
output_csv = 'dataset_files.csv'

files = sorted(input_folder.glob('*.png')) 

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['rgb', 'dense'])
    
    for file_path in files:
        writer.writerow(["rgb/"+file_path.name, "dense/gt_" + file_path.name])