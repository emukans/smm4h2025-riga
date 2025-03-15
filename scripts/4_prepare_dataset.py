import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


if __name__ == '__main__':
    data_dir = '../data/task1'

    test_df = pd.read_csv(os.path.join(data_dir, 'dev_preprocessed.csv'))
    train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))

    train_df, val_df = train_test_split(train_df, test_size=0.2)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'dev': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df)
    })

    dataset.save_to_disk('../data/task1/plain_ds_preprocessed')
