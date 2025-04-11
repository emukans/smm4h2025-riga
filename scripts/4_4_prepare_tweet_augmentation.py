import json

import pandas as pd
import os

import tiktoken

if __name__ == '__main__':
    data_dir = '../data/task1'

    test_df = pd.read_csv(os.path.join(data_dir, 'dev_preprocessed.csv'))
    train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))
    encoder = tiktoken.encoding_for_model("gpt-4o")

    dataset_dir = os.path.join(data_dir, 'stratified', 'train_tweet_augmentation')
    os.makedirs(dataset_dir, exist_ok=True)

    print(len(test_df[test_df['label'] == 1]), len(test_df[test_df['label'] == 0]), len(test_df[test_df['label'] == 1]) / len(test_df[test_df['label'] == 0]))
    print(len(train_df[train_df['label'] == 1]), len(train_df[train_df['label'] == 0]), len(train_df[train_df['label'] == 1]) / len(train_df[train_df['label'] == 0]))

    with open(os.path.join(data_dir, 'stratified/drug_mining2/processed.json'), 'r') as f:
        normalized_drug_map = json.load(f)

    with open(os.path.join(data_dir, 'stratified/ade_mining_advanced/processed.json'), 'r') as f:
        gpt_ade_map = json.load(f)

    total_input_tokens = 0
    result = []
    for _ in range(1000):
        samples = []
        for i, tweet in train_df[train_df['label'] == 1].sample(10).iterrows():
            samples.append(f"Drugs: {', '.join(normalized_drug_map[tweet['id']]) if len(normalized_drug_map[tweet['id']]) else 'null'}\nSymptoms: {gpt_ade_map[tweet['id']]}\nText: {tweet['text']}\n---")

        content = '\n'.join(samples)
        total_input_tokens += len(encoder.encode(content))
        result.append(content)

    with open(os.path.join(dataset_dir, 'source.json'), 'w') as f:
        json.dump(result, f)

    print(result[0])
    print('Total input tokens: ', total_input_tokens)
