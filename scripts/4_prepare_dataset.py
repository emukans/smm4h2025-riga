import json
import os
import re

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datasets import Dataset, DatasetDict


def build_map(augmentation_type):
    augmentation_path = os.path.join(data_dir, f'test_stratified/{augmentation_type}/augmentation_result.jsonl')
    augmentation_map = {}
    with open(augmentation_path, 'r') as f:
        for line in f.readlines():
            result = json.loads(line)
            augmentation_map[result['custom_id']] = result['response']['body']['choices'][0]['message']['content']

    return augmentation_map


def parse_augmented_samples(augmentation_map):
    augmented_samples = []
    current_entry = {}
    for custom_id, text in augmentation_map.items():
        i = 0
        for line in text.strip().splitlines():
            line = line.strip(' -')
            if not line:
                continue

            if line.lower().startswith('drugs:'):
                current_entry['drugs'] = re.match(r'drugs: (.*)', line, re.I).groups()[0]
                current_entry['drugs'] = ' [drug] '.join([s.strip() for s in current_entry['drugs'].split(',')])
            elif line.lower().startswith('symptoms:'):
                current_entry['symptoms'] = re.match(r'symptoms: (.*)', line, re.I).groups()[0]
            elif line.lower().startswith('text:'):
                current_entry['text'] = re.match(r'text: (.*)', line, re.I).groups()[0]

            if len(current_entry) == 3:
                current_entry['id'] = f'{custom_id}_{i}'
                current_entry['label'] = 1
                i += 1
                augmented_samples.append(current_entry)
                current_entry = {}

    return pd.DataFrame(augmented_samples)

if __name__ == '__main__':
    data_dir = '../data/task1'

    # test_df = pd.read_csv(os.path.join(data_dir, 'dev_preprocessed.csv'))
    # train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_preprocessed.csv'))

    # train_df = train_df[train_df['language'].isin(['de', 'fr'])]
    # test_df = test_df[test_df['language'].isin(['de', 'fr'])]

    with open(os.path.join(data_dir, 'known_interactions.json'), 'r') as f:
        known_interactions_map = json.load(f)

    with open(os.path.join(data_dir, 'known_interaction_description.json'), 'r') as f:
        known_interaction_description_map = json.load(f)

    with open(os.path.join(data_dir, 'food_interactions.json'), 'r') as f:
        food_interactions_map = json.load(f)

    with open(os.path.join(data_dir, 'drug_classification.json'), 'r') as f:
        classification_map = json.load(f)

    with open(os.path.join(data_dir, 'drug_description.json'), 'r') as f:
        description_map = json.load(f)

    with open(os.path.join(data_dir, 'drug_ade.json'), 'r') as f:
        drug_ade = json.load(f)

    with open(os.path.join(data_dir, 'test_stratified/drug_mining2/processed.json'), 'r') as f:
    # with open(os.path.join(data_dir, 'stratified/drug_mining2/processed_description.json'), 'r') as f:
        normalized_drug_map = json.load(f)

    with open(os.path.join(data_dir, 'test_stratified/ade_mining/processed.json'), 'r') as f:
        gpt_ade_map = json.load(f)

    # augmentation_type = 'translation2'
    augmentation_type = 'translate_summarize2'
    # augmentation_type = 'translate_summarize_advanced'

    translate_map = build_map('translation2')
    # translate_map = build_map('translate_advanced')
    # paraphrase_map = build_map('paraphrase_advanced')
    translate_summarize_map = build_map(augmentation_type)

    # train_augmentation_result = build_map('train_tweet_augmentation')
    # train_aug_df = parse_augmented_samples(train_augmentation_result)

    for id_, content in tqdm(translate_map.items()):
        if id_ in translate_summarize_map:
            continue
        # train_df.loc[train_df['id'] == id_, 'text'] = content
        test_df.loc[test_df['id'] == id_, 'text'] = content

    # for id_, content in tqdm(paraphrase_map.items()):
    #     train_df.loc[train_df['id'] == id_, 'text'] = content
    #     test_df.loc[test_df['id'] == id_, 'text'] = content

    for id_, content in tqdm(translate_summarize_map.items()):
        # train_df.loc[train_df['id'] == id_, 'text'] = content
        test_df.loc[test_df['id'] == id_, 'text'] = content

    # for id_, content in tqdm(gpt_ade_map.items()):
    #     if content is None:
    #         continue
    #     # train_df.loc[train_df['id'] == id_, 'text'] = ' [gpt] ' + content + ' [sep] ' + train_df[train_df['id'] == id_]['text']
    #     test_df.loc[test_df['id'] == id_, 'text'] = ' [gpt] ' + content + ' [sep] ' + test_df[test_df['id'] == id_]['text']

    for id_, drug_list in tqdm(normalized_drug_map.items()):
        # continue
        if not len(drug_list):
            continue

        has_drug_interaction = False
        for to_check_drug in drug_list:
            if has_drug_interaction:
                break
            interaction_list = known_interactions_map[to_check_drug]
            for checking_drug in drug_list:
                if checking_drug == to_check_drug:
                    continue

                if checking_drug in interaction_list:
                    has_drug_interaction = True
                    break

        drug_interaction_text = ''
        if has_drug_interaction:
            drug_interaction_text = ' [sep] has known drug interaction'

        drug_interaction_description_text = []
        drug_to_skip = []
        for to_check_drug in drug_list:
            if to_check_drug in drug_to_skip:
                continue
            interaction_list = known_interaction_description_map[to_check_drug]
            for checking_drug in drug_list:
                if checking_drug == to_check_drug:
                    continue

                if checking_drug in interaction_list:
                    drug_to_skip.append(checking_drug)
                    drug_interaction_description_text.append(interaction_list[checking_drug])
                    break

        drug_interaction_description_text = ' [ade] '.join(drug_interaction_description_text)
        if len(drug_interaction_description_text):
            drug_interaction_description_text = ' [ade] ' + drug_interaction_description_text

        has_food_interaction = any(len(food_interactions_map[drug]) > 0 for drug in drug_list)

        food_interaction_text = ''
        if has_food_interaction:
            food_interaction_text = ' [sep] has known food interaction'

        classification_text = []
        for drug in drug_list:
            if drug.lower() not in classification_map:
                print(f"Missing drug classification: {drug.lower()}")
                # classification_text.append(f' [drug] {drug}')
                continue
            if not len(classification_map[drug.lower()]):
                classification_text.append(f' [drug] {drug}')
                continue

            classification_text.append(f' [drug] {drug} [desc] {classification_map[drug]}')
        classification_text = ''.join(classification_text).strip()

        ade_text = []
        for drug in drug_list:
            if drug.lower() not in drug_ade:
                ade_text.append(f' [drug] {drug}')
                continue

            ade_text.append(f' [drug] {drug} [ade] {" [ade] ".join(drug_ade[drug])}')
        ade_text = ''.join(ade_text).strip()

        ade_short_text = []
        for drug in drug_list:
            if drug.lower() not in drug_ade:
                continue

            ade_short_text.append(f' [ade] {" [ade] ".join(drug_ade[drug])}')
        ade_short_text = ''.join(ade_short_text).strip()

        drug_text = '[drug] ' + ' [drug] '.join(drug_list)

        add_sep = ''
        if not test_df[test_df['id'] == id_]['text'].str.startswith(' [sep] ').iloc[0] and not test_df[test_df['id'] == id_]['text'].str.startswith(' [gpt] ').iloc[0]:
            add_sep = ' [sep] '

        # train_df.loc[train_df['id'] == id_, 'text'] = drug_text + drug_interaction_text + food_interaction_text + train_df[train_df['id'] == id_]['text']
        test_df.loc[test_df['id'] == id_, 'text'] = drug_text + drug_interaction_text + food_interaction_text + add_sep + test_df[test_df['id'] == id_]['text']

    # train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True)

    # # todo: unify
    # train_aug_df['text'] = '[drug] ' + train_aug_df['drugs'] + ' [gpt] ' + train_aug_df['symptoms'] + ' [sep] ' + train_aug_df['text']
    # train_df = pd.concat([train_df, train_aug_df], ignore_index=True)
    #
    # train_df = shuffle(train_df)
    # train_df.reset_index(inplace=True, drop=True)

    dataset = DatasetDict({
        # 'train': Dataset.from_pandas(train_df, preserve_index=False),
        # 'dev': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df)
    })

    dataset.save_to_disk(f'../data/task1/test/ds_preprocessed_{augmentation_type}_with_gpt_drug_food_interaction_2')
    # dataset.save_to_disk(f'../data/task1/ds_preprocessed_defr_with_classification2_drug_food_interaction')
