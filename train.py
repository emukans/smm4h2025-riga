from random import seed

import wandb

import os

import numpy as np

from datasets import DatasetDict
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


seed(42)
np.random.seed(42)


MAX_LEN = 100
batch_size = 32
# epoch_count = 1
epoch_count = 30
learning_rate = 5e-7
downsample_size = 1


# checkpoint = "distilbert/distilbert-base-uncased"
# checkpoint = "cardiffnlp/tweet-topic-21-multi"
# checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
checkpoint = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"
# checkpoint = "cardiffnlp/twitter-roberta-large-hate-latest"
# checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
# checkpoint = "microsoft/deberta-v2-xxlarge-mnli"
# checkpoint = "Azie88/COVID_Vaccine_Tweet_sentiment_analysis_roberta"
# checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# checkpoint = "papluca/xlm-roberta-base-language-detection"

# dataset_type = 'plain_ds'
# dataset_type = 'ds_preprocessed'
# dataset_type = 'ds_preprocessed_translation'
# dataset_type = 'ds_preprocessed_translate_summarize'
dataset_type = 'ds_preprocessed_translate_summarize_full'

os.environ["WANDB_PROJECT"] = "smm4h2025-task1-classification"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NAME"] = f"{checkpoint}/{dataset_type}/lr-{learning_rate}-downsample-{downsample_size}-max_len-{MAX_LEN}-3"
# os.environ["WANDB_NOTES"] = "Spans extracted by GPT3.5 from tweets, classification. Downample 0.2"


id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
# model = AutoModelForSequenceClassification.from_pretrained('model/' + os.environ["WANDB_NAME"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


wandb.init()

ds_path = f'data/task1/{dataset_type}'
dataset = DatasetDict.load_from_disk(ds_path)
dev_df = dataset['dev'].to_pandas().copy()
test_df = dataset['test'].to_pandas().copy()

wandb.log({
    'train_size': len(dataset['train']),
    'dev_size': len(dataset['dev']),
    'test_size': len(dataset['test']),
    'model_size': model.num_parameters(),
    'max_len': MAX_LEN,
})


def preprocess_function(examples):
    return tokenizer([s for s in examples["text"]], max_length=MAX_LEN, truncation=True, padding='max_length')


dataset = dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="model/" + os.environ["WANDB_NAME"],
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch_count,
    weight_decay=0.001,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['dev'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

trainer.save_model("model/" + os.environ["WANDB_NAME"])


test_predictions = trainer.predict(dataset['test'])
test_df['prediction'] = np.argmax(test_predictions.predictions, axis=1)

dev_predictions = trainer.predict(dataset['dev'])
dev_df['prediction'] = np.argmax(dev_predictions.predictions, axis=1)


def stratified_predictions(df, category, split):
    result = dict()
    for cat_type in df[category].unique():
        df_to_test = df[df[category] == cat_type]

        r = accuracy.compute(predictions=df_to_test['prediction'].tolist(), references=df_to_test['label'].tolist())
        for k, v in r.items():
            result[split + '/' + category + '/' + cat_type + '/' + k] = v

    return result


wandb.log(stratified_predictions(dev_df, 'language', 'eval'))
wandb.log(stratified_predictions(dev_df, 'type', 'eval'))

wandb.log(stratified_predictions(test_df, 'language', 'test'))
wandb.log(stratified_predictions(test_df, 'type', 'test'))
