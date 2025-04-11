import sys
from random import seed

import wandb

import os
import torch

import numpy as np
from sklearn.metrics import precision_recall_curve
import torch.nn.functional as F

from datasets import DatasetDict
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import concatenate_datasets


seed(42)
np.random.seed(42)

# iteration = int(sys.argv[1])
# gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])


# todo: try longer contexts
MAX_LEN = 150
batch_size = 16
# epoch_count = 1
epoch_count = 20
learning_rate = 5e-6
cased = False
weight_decay = 0.0001


# checkpoint = "distilbert/distilbert-base-uncased"
# checkpoint = "cardiffnlp/tweet-topic-21-multi"
# checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
checkpoint = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"  # this
# checkpoint = "cardiffnlp/twitter-roberta-large-hate-latest"
# checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
# checkpoint = "microsoft/deberta-v2-xxlarge-mnli"
# checkpoint = "Azie88/COVID_Vaccine_Tweet_sentiment_analysis_roberta"
# checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# checkpoint = "papluca/xlm-roberta-base-language-detection"
# checkpoint = "meta-llama/Llama-3.2-1B"
# checkpoint = "FacebookAI/xlm-roberta-large"
# checkpoint = "EuroBERT/EuroBERT-610m"  # this
# checkpoint = "EuroBERT/EuroBERT-2.1B"

# dataset_type = 'plain_ds'
# dataset_type = 'ds_preprocessed'
# dataset_type = 'ds_preprocessed_translation'
# dataset_type = 'ds_preprocessed_translate_summarize'
# dataset_type = 'ds_preprocessed_translate_summarize_full'
# dataset_type = 'ds_preprocessed_translate_summarize_with_drugs'
# dataset_type = 'ds_preprocessed_translate_summarize_with_drugs_description'
# dataset_type = 'ds_preprocessed_translate_summarize_with_drugbank_names'
# dataset_type = 'ds_preprocessed_translate_summarize_with_drugbank_names2'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_drugbank_description'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_drugbank_classification'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_classification2'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_classification2_food_interaction'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_classification2_drug_interaction'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_classification2_drug_food_interaction'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_drug_interaction'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_food_interaction'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_drug_food_interaction'
# dataset_type = 'ds_preprocessed_with_classification2_drug_food_interaction'  # try this
# dataset_type = 'ds_preprocessed_enru_with_classification2_drug_food_interaction'
# dataset_type = 'ds_preprocessed_en_with_classification2_drug_food_interaction'
# dataset_type = 'ds_preprocessed_defr_with_classification2_drug_food_interaction'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_ade'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_interaction_description'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_drugs'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_drugs_interaction'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_classification'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_classification_description'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_classification_description_ade'  # todo
dataset_type = 'ds_preprocessed_translate_summarize2_with_gpt'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_gpt'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_gpt_drug_ade'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_gpt_drug_classification'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_gpt_drug2'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_gpt_drug_aug_2'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_gpt_drug_2'
# dataset_type = 'ds_preprocessed_translate_summarize_advanced_with_gpt_drug_aug_2'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_gpt_drug_interaction_2'
# dataset_type = 'ds_preprocessed_translate_summarize2_with_gpt_drug_food_interaction_2'

# dataset_gpu_map = [
#     ['ds_preprocessed_translate_summarize2_with_gpt', 'ds_preprocessed_translate_summarize_advanced_with_drugs'],
#     ['ds_preprocessed_translate_summarize_advanced_with_gpt', 'ds_preprocessed_translate_summarize_advanced_with_classification'],
#     ['ds_preprocessed_translate_summarize_advanced_with_classification_description', 'ds_preprocessed_translate_summarize_advanced_with_classification_description_ade'],
#     ['ds_preprocessed_translate_summarize_advanced_with_ade_description', 'ds_preprocessed_translate_summarize_advanced_with_ade']
# ]
#
# dataset_type = dataset_gpu_map[gpu][iteration % 2]
#
# if 0 <= iteration < 2:
#     MAX_LEN = 100
# elif 2 <= iteration < 4:
#     MAX_LEN = 150
# elif 4 <= iteration < 6:
#     MAX_LEN = 200
# elif 6 <= iteration < 8:
#     MAX_LEN = 300
# elif 8 <= iteration < 10:
#     MAX_LEN = 200
#     weight_decay = 0.0001
#
# print(dataset_type, MAX_LEN, weight_decay, os.path.exists(f'data/task1/{dataset_type}'))

# exit()
os.environ["WANDB_PROJECT"] = "smm4h2025-task1-classification"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NAME"] = f"{checkpoint}/{dataset_type}/lr-{learning_rate}-max_len-{MAX_LEN}-cased-{cased}-decay-{weight_decay}-12-join"
# os.environ["WANDB_NOTES"] = "Spans extracted by GPT3.5 from tweets, classification. Downample 0.2"


id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True, trust_remote_code=True
)
# model = AutoModelForSequenceClassification.from_pretrained('model/' + os.environ["WANDB_NAME"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions_max = np.argmax(predictions, axis=1)

    results = accuracy.compute(predictions=predictions_max, references=labels)

    predictions_sigmoid = F.softmax(torch.Tensor(predictions))
    precision, recall, thresholds = precision_recall_curve(labels, predictions_sigmoid[:, 1])

    f1_scores = (2 * precision * recall) / (precision + recall)

    best_idx = np.nanargmax(f1_scores)

    results['best_threshold'] = thresholds[best_idx]
    results['best_f1'] = f1_scores[best_idx]

    # wandb.log({
    #     'eval/roc_curve': wandb.plot.roc_curve(labels, predictions),
    #     'eval/pr_curve': wandb.plot.pr_curve(labels, predictions)
    # })

    return results


wandb.init()

ds_path = f'data/task1/{dataset_type}'
dataset = DatasetDict.load_from_disk(ds_path)


# dataset['dev'] = dataset['dev'].select(range(20))
# dataset['test'] = dataset['test'].select(range(20))

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
    return tokenizer([s if cased else s.lower() for s in examples["text"]], max_length=MAX_LEN, truncation=True, padding='max_length')


dataset = dataset.map(preprocess_function, batched=True)

joined_dataset = concatenate_datasets([dataset['train'], dataset['dev']]).shuffle(seed=42)

training_args = TrainingArguments(
    output_dir="model/" + os.environ["WANDB_NAME"],
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch_count,
    weight_decay=weight_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=joined_dataset,
    # train_dataset=dataset['dev'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("model/" + os.environ["WANDB_NAME"])


def stratified_predictions(df, category, split):
    result = dict()
    for cat_type in df[category].unique():
        df_to_test = df[df[category] == cat_type]

        r = accuracy.compute(predictions=df_to_test['prediction'].tolist(), references=df_to_test['label'].tolist())
        for k, v in r.items():
            result[split + '/' + category + '/' + cat_type + '/' + k] = v

    return result

# dev_predictions = trainer.predict(dataset['dev'], metric_key_prefix='eval')
# dev_df['prediction'] = np.argmax(dev_predictions.predictions, axis=1)
# # dev_df.to_csv('data/task1/dev_result.csv', index=False)
# # exit()
# wandb.log(stratified_predictions(dev_df, 'language', 'eval'))
# wandb.log(stratified_predictions(dev_df, 'type', 'eval'))
#
# wandb.log({"eval/roc_curve": wandb.plot.roc_curve(dev_df['label'].tolist(), dev_predictions.predictions)})
# wandb.log({"eval/pr_curve": wandb.plot.pr_curve(dev_df['label'].tolist(), dev_predictions.predictions)})


test_predictions = trainer.predict(dataset['test'], metric_key_prefix='test')
test_df['prediction'] = np.argmax(test_predictions.predictions, axis=1)
# test_df.to_csv('data/task1/test_result.csv', index=False)

wandb.log({"test/roc_curve": wandb.plot.roc_curve(test_df['label'].tolist(), test_predictions.predictions)})
wandb.log({"test/pr_curve": wandb.plot.pr_curve(test_df['label'].tolist(), test_predictions.predictions)})

wandb.log(stratified_predictions(test_df, 'language', 'test'))
wandb.log(stratified_predictions(test_df, 'type', 'test'))
