import json
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def build_map(augmentation_type):
    augmentation_path = os.path.join(data_dir, f'stratified/{augmentation_type}/augmentation_result.jsonl')
    augmentation_map = {}
    with open(augmentation_path, 'r') as f:
        for line in f.readlines():
            result = json.loads(line)
            augmentation_map[result['custom_id']] = result['response']['body']['choices'][0]['message']['content']

    return augmentation_map


if __name__ == '__main__':
    data_dir = '../data/task1'

    test_df = pd.read_csv(os.path.join(data_dir, 'dev_preprocessed.csv'))
    train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))

    # augmentation_type = 'translation'
    augmentation_type = 'translate_summarize'

    translate_map = build_map('translation')
    translate_summarize_map = build_map(augmentation_type)

    for id_, content in tqdm(translate_map.items()):
        if id_ in translate_summarize_map:
            continue
        train_df.loc[train_df['id'] == id_, 'text'] = content
        test_df.loc[train_df['id'] == id_, 'text'] = content

    for id_, content in tqdm(translate_summarize_map.items()):
        train_df.loc[train_df['id'] == id_, 'text'] = content
        test_df.loc[train_df['id'] == id_, 'text'] = content

    train_df, val_df = train_test_split(train_df, test_size=0.2)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'dev': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df)
    })

    dataset.save_to_disk(f'../data/task1/ds_preprocessed_{augmentation_type}_full')
