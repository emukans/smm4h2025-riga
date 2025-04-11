import csv
import json
import os


data_dir = '../data/task1'
result_dir = '../data/task1/result'


with open(os.path.join(data_dir, 'test_stratified/ade_mining/processed.json'), 'r') as f:
    processed = json.load(f)

result = []

with open(os.path.join(result_dir, 'gpt.csv'), 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'predicted_label'])

    for id_, ade in processed.items():
        csv_writer.writerow([id_.replace('test_', ''), 1 if ade is not None else 0])
