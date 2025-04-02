from random import seed

import os
import csv
import torch
from torch.utils.data import DataLoader

import numpy as np

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


seed(42)
np.random.seed(42)


MAX_LEN = 100
BATCH_SIZE = 16
os.environ["WANDB_DISABLED"] = "true"

# model_name = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest/ds_preprocessed_translate_summarize_full/lr-2e-05-downsample-1-max_len-100-4"
model_name = "distilbert/distilbert-base-uncased/plain/lr-2e-05-downsample-1-max_len-200-2"

# ds_path = f'data/task1/ds_preprocessed_translate_summarize_full'
ds_path = f'data/task1/plain_ds'

id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

tokenizer = AutoTokenizer.from_pretrained('model/' + model_name)
model = AutoModelForSequenceClassification.from_pretrained('model/' + model_name)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



dataset = DatasetDict.load_from_disk(ds_path)['test']
test_df = dataset.to_pandas().copy()
test_df.loc[:, 'id'] = test_df['id'].apply(lambda x: x.replace('dev_', ''))


def preprocess_function(examples):
    return tokenizer([s for s in examples["text"]], max_length=MAX_LEN, truncation=True, padding='max_length')


dataset = dataset.map(preprocess_function, batched=True)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

test_predictions = trainer.predict(dataset, metric_key_prefix='test')
test_df['predicted_label'] = np.argmax(test_predictions.predictions, axis=1)

test_df[['id', 'predicted_label']].to_csv(f'result/{model_name.replace("/", "_")}.csv', index=False)
