from zipfile import ZipFile

import pandas as pd
import os
from tqdm import tqdm


data_dir = '../data/task1'
result_dir = '../data/task1/result'
result_file = 'gpt.csv'

test_df = pd.read_csv(os.path.join(data_dir, 'test_preprocessed.csv'))
test_df.loc[:, 'id'] = test_df['id'].apply(lambda x: x.replace('test_', ''))
result_df = pd.read_csv(os.path.join(result_dir, result_file))

missing_id = set(test_df['id'].unique()) - set(result_df['id'].unique())
to_add = []

for id_ in missing_id:
    to_add.append({'id': id_, 'predicted_label': 0})

result_df = pd.concat([result_df, pd.DataFrame(to_add)], ignore_index=True)

print(len(test_df['id'].unique()), len(result_df['id'].unique()))

assert set(test_df['id'].unique()) == set(result_df['id'].unique())

print("Total predicted:", len(result_df))
print("Total positive:", len(result_df.loc[result_df['predicted_label'] == 1]))
print("Total positive %:", len(result_df.loc[result_df['predicted_label'] == 1]) / len(result_df))

dup_df = pd.read_csv(os.path.join(data_dir, 'test_duplicated.csv'))

for _, dup in tqdm(dup_df.iterrows()):
    id_list = test_df[test_df['text'].str.lower() == dup['text'].lower()]['id'].unique()
    if result_df[result_df['id'].isin(id_list)]['predicted_label'].iloc[0] != dup['label']:
        print(dup['text'], result_df[result_df['id'].isin(id_list)]['predicted_label'].iloc[0])
    # result_df.loc[id_list['id'].isin(id_list), 'predicted_label'] = dup['label']

train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))
dev_df = pd.read_csv(os.path.join(data_dir, 'dev_preprocessed.csv'))
df = pd.concat([train_df, dev_df], ignore_index=True)

matching_texts = []
text_unique = df['text'].str.lower().unique()
for row in tqdm(test_df['text'].str.lower().unique().tolist()):
    if row in text_unique:
        matching_texts.append(row)

for text in matching_texts:
    result_df.loc[df['text'].str.lower() == text, 'predicted_label'] = df[df['text'].str.lower() == text]['label'].iloc[0]


name_map = {
'cardiffnlp_twitter-roberta-large-topic-sentiment-latest_ds_preprocessed_translate_summarize2_with_gpt_drug_food_interaction_2_lr-5e-06-max_len-150-cased-False-decay-0.0001-12-join.csv': 'r1.csv',
'cardiffnlp_twitter-roberta-large-topic-sentiment-latest_ds_preprocessed_translate_summarize2_with_gpt_drug_interaction_2_lr-5e-06-max_len-150-cased-True-decay-0.0001-12-join.csv': 'r2.csv',
'cardiffnlp_twitter-roberta-large-topic-sentiment-latest_ds_preprocessed_translate_summarize2_with_gpt_drug2_lr-5e-06-max_len-150-cased-False-decay-0.0001-12-join.csv': 'r3.csv',
'cardiffnlp_twitter-roberta-large-topic-sentiment-latest_ds_preprocessed_translate_summarize2_with_gpt_lr-5e-06-max_len-150-cased-True-decay-1e-05-9.csv': 'r4.csv',
'cardiffnlp_twitter-roberta-large-topic-sentiment-latest_ds_preprocessed_translate_summarize2_with_gpt_drug2_lr-5e-06-max_len-150-cased-True-decay-0.0001-12.csv': 'r5.csv',
'cardiffnlp_twitter-roberta-large-topic-sentiment-latest_ds_preprocessed_translate_summarize2_with_gpt_lr-5e-06-max_len-150-cased-False-decay-0.0001-12-join.csv': 'r6.csv',
'cardiffnlp_twitter-roberta-large-topic-sentiment-latest_ds_preprocessed_translate_summarize_lr-5e-06-max_len-150-cased-False-decay-0.0001-12-join.csv': 'r7.csv',
'gpt.csv': 'r8.csv',
}

mapped_name = name_map[result_file]
result_df.to_csv(os.path.join(result_dir, mapped_name), index=False)

with ZipFile(os.path.join(result_dir, mapped_name.replace('.csv', '.zip')), 'w') as z:
    z.write(os.path.join(result_dir, mapped_name), mapped_name)

print('Matched texts:', len(matching_texts))
