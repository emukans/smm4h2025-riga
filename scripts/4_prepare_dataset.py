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

    with open(os.path.join(data_dir, 'stratified/drug_mining2/processed.json'), 'r') as f:
        normalized_drug_map = json.load(f)

    # augmentation_type = 'translation2'
    augmentation_type = 'translate_summarize2'

    translate_map = build_map('translation2')
    translate_summarize_map = build_map(augmentation_type)

    for id_, content in tqdm(translate_map.items()):
        if id_ in translate_summarize_map:
            continue
        train_df.loc[train_df['id'] == id_, 'text'] = content
        test_df.loc[test_df['id'] == id_, 'text'] = content

    for id_, content in tqdm(translate_summarize_map.items()):
        train_df.loc[train_df['id'] == id_, 'text'] = content
        test_df.loc[test_df['id'] == id_, 'text'] = content

    for id_, drug_list in tqdm(normalized_drug_map.items()):
        if not len(drug_list):
            continue
        train_df.loc[train_df['id'] == id_, 'text'] = ' [sep] ' + '[sep] '.join(drug_list) + train_df[train_df['id'] == id_]['text']
        test_df.loc[test_df['id'] == id_, 'text'] = ' [sep] ' + '[sep] '.join(drug_list) + test_df[test_df['id'] == id_]['text']

    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'dev': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df)
    })

    dataset.save_to_disk(f'../data/task1/ds_preprocessed_{augmentation_type}_with_drugs')
