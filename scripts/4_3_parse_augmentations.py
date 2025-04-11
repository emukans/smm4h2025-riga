import csv
import json
import os
import re

import pandas as pd
from rapidfuzz import process, fuzz
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def build_map(augmentation_type):
    augmentation_path = os.path.join(data_dir, f'test_stratified/{augmentation_type}/augmentation_result.jsonl')
    augmentation_result = {}

    with open(augmentation_path, 'r') as f:
        for line in tqdm(f.readlines()):
            result = json.loads(line)
            if result['error'] is not None or result['response']['status_code'] != 200:
                continue
            augmentation_result[result['custom_id']] = ', '.join([s.strip('-* ') for s in result['response']['body']['choices'][0]['message']['content'].strip().splitlines()])

            if augmentation_result[result['custom_id']].lower() == 'null':
                augmentation_result[result['custom_id']] = None

    with open(os.path.join(data_dir, f'test_stratified/{augmentation_type}/processed.json'), 'w') as f:
        json.dump(augmentation_result, f)


if __name__ == '__main__':
    data_dir = '../data/task1'

    # augmentation_type = 'translate_summarize_advanced'
    # augmentation_type = 'translate_advanced'
    # augmentation_type = 'paraphrase_advanced'
    augmentation_type = 'ade_mining'

    build_map(augmentation_type)
