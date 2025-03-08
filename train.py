from random import shuffle, seed

import torch
import wandb
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback

import os

import numpy as np

from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


seed(42)
np.random.seed(42)


MAX_LEN = 200
batch_size = 64
epoch_count = 50
learning_rate = 2e-5
downsample_size = 1


# checkpoint = "distilbert/distilbert-base-uncased"
# checkpoint = "cardiffnlp/tweet-topic-21-multi"
# checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# checkpoint = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"
# checkpoint = "cardiffnlp/twitter-roberta-large-hate-latest"
# checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
# checkpoint = "microsoft/deberta-v2-xxlarge-mnli"
# checkpoint = "Azie88/COVID_Vaccine_Tweet_sentiment_analysis_roberta"
# checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
checkpoint = "papluca/xlm-roberta-base-language-detection"

train_dataset_dir = 'data/task1/train'
dev_dataset_dir = 'data/task1/dev'
dataset_type = 'plain'

os.environ["WANDB_PROJECT"] = "smm4h2025-task1-classification"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NAME"] = f"{checkpoint}/{dataset_type}/lr-{learning_rate}-downsample-{downsample_size}-max_len-{MAX_LEN}-1"
# os.environ["WANDB_NOTES"] = "Spans extracted by GPT3.5 from tweets, classification. Downample 0.2"


id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


with open(os.path.join(train_dataset_dir, dataset_type, 'positive.csv')) as f:
    train_positive_list = f.readlines()


with open(os.path.join(train_dataset_dir, dataset_type, 'negative.csv')) as f:
    train_negative_list = f.readlines()


shuffle(train_negative_list)
train_negative_list = train_negative_list[:round(len(train_negative_list) * downsample_size)]

with open(os.path.join(dev_dataset_dir, dataset_type, 'positive.csv')) as f:
    dev_positive_list = f.readlines()


with open(os.path.join(dev_dataset_dir, dataset_type, 'negative.csv')) as f:
    dev_negative_list = f.readlines()


wandb.init()
wandb.log({
    'train_size': len(train_positive_list + train_negative_list),
    'dev_size': len(dev_positive_list + dev_negative_list),
    'downsample_size': downsample_size,
    'train_positive_proportion': len(train_positive_list) / len(train_positive_list + train_negative_list),
    'dev_positive_proportion': len(dev_positive_list) / len(dev_positive_list + dev_negative_list),
    'model_size': model.num_parameters(),
    'max_len': MAX_LEN,
})

train_dataset = Dataset.from_dict({'text': train_positive_list + train_negative_list, 'label': [1] * len(train_positive_list) + [0] * len(train_negative_list)})
dev_dataset = Dataset.from_dict({'text': dev_positive_list + dev_negative_list, 'label': [1] * len(dev_positive_list) + [0] * len(dev_negative_list)})


def preprocess_function(examples):
    return tokenizer([s.lower() for s in examples["text"]], max_length=MAX_LEN, truncation=True, padding='max_length')


train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="model/" + os.environ["WANDB_NAME"],
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch_count,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

trainer.save_model("model/" + os.environ["WANDB_NAME"])


# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# predicted_class_id = logits.argmax().item()
# model.config.id2label[predicted_class_id]
