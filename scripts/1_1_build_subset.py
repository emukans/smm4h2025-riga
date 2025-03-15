import json
import os

import pandas as pd


def select_by_field(df, field_type, field_value):
    result = {}
    for index, row in df[df[field_type] == field_value].iterrows():
        if row['id'] in result:
            raise Exception(f"Duplicate: {row['id']}")
        result[row['id']] = row['text']

    return result


if __name__ == "__main__":
    split_list = ["train", "dev"]
    os.makedirs('../data/task1/stratified', exist_ok=True)

    for split in split_list:
        df = pd.read_csv(f'../data/task1/{split}_preprocessed.csv')

        for language in df['language'].unique():
            result = select_by_field(df, 'language', language)
            with open(f'../data/task1/stratified/{split}_language_{language}.json', 'w') as f:
                json.dump(result, f)

            print(f"Total {split}-language-{language}: {len(result)}")

        for language in df['type'].unique():
            result = select_by_field(df, 'type', language)
            with open(f'../data/task1/stratified/{split}_type_{language}.json', 'w') as f:
                json.dump(result, f)

            print(f"Total {split}-type-{language}: {len(result)}")
