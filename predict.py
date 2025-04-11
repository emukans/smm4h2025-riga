from random import seed

import os
import csv
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

import numpy as np

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


seed(42)
np.random.seed(42)


output_text = False

MAX_LEN = 150
BATCH_SIZE = 16
cased = False
os.environ["WANDB_DISABLED"] = "true"

# model_name = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest/ds_preprocessed_translate_summarize_full/lr-2e-05-downsample-1-max_len-100-4"
# model_name = "distilbert/distilbert-base-uncased/plain/lr-2e-05-downsample-1-max_len-200-2"
model_name = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest/ds_preprocessed_translate_summarize/lr-5e-06-max_len-150-cased-False-decay-0.0001-12-join"

ds_path = f'data/task1/test/ds_preprocessed_translate_summarize2'
# ds_path = f'data/task1/ds_preprocessed_translate_summarize_full'
# ds_path = f'data/task1/plain_ds'

id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

tokenizer = AutoTokenizer.from_pretrained('model/' + model_name)
model = AutoModelForSequenceClassification.from_pretrained('model/' + model_name)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


dataset = DatasetDict.load_from_disk(ds_path)['test']
test_df = dataset.to_pandas().copy()
test_df.loc[:, 'id'] = test_df['id'].apply(lambda x: x.replace('test_', ''))


def preprocess_function(examples):
    return tokenizer([s if cased else s.lower() for s in examples["text"]], max_length=MAX_LEN, truncation=True, padding='max_length')


dataset = dataset.map(preprocess_function, batched=True)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

test_predictions = trainer.predict(dataset, metric_key_prefix='test')

# print(test_predictions.predictions)
# preds = torch.Tensor(test_predictions.predictions).sigmoid()
# print(preds.tolist())
# print(preds[:, 1].tolist())
# print(test_df['label'].loc[[0, 1, 2]].tolist())

# fpr, tpr, thresholds = roc_curve(test_df['label'].tolist(), preds[:, 1].tolist())

# precisions, recalls, thresholds = precision_recall_curve(test_df['label'].tolist(), preds[:, 1].tolist())

# f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

# # Find the index of the best F1 score
# best_idx = np.argmax(f1_scores)
# best_threshold = thresholds[best_idx]
# 
# print(f"Best Threshold: {best_threshold:.4f}")
# print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")
# 
# # wandb.plot.roc_curve()
# print('Threshold:', np.mean(thresholds))

test_df['predicted_label'] = np.argmax(test_predictions.predictions, axis=1)

output_columns = ['id', 'predicted_label']

if output_text:
    output_columns += ['text', 'label']

test_df[output_columns].to_csv(f'result/{model_name.replace("/", "_")}.csv', index=False)
