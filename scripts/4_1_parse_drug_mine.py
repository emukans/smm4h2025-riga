import csv
import json
import os
import re

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def build_map(augmentation_type):

    with open(os.path.join(data_dir, 'drugbank_map.json')) as f:
        drug_map = json.load(f)

    with open(os.path.join(data_dir, 'ru_drug_map.json')) as f:
        ru_drug_map = json.load(f)

    with open(os.path.join(data_dir, 'unmapped_drug_map.json')) as f:
        unmapped_drug_map = json.load(f)

    with open(os.path.join(data_dir, 'drug_description.json')) as f:
        drug_description_map = json.load(f)

    augmentation_path = os.path.join(data_dir, f'stratified/{augmentation_type}/augmentation_result.jsonl')
    drug_mining_map = {}
    drug_mining_map_with_description = {}
    augmentation_map = set()
    multiple_mappings = []
    no_mappings = []
    no_mappings_unique = set()
    ru_no_mappings = set()
    total_not_found = 0
    with open(augmentation_path, 'r') as f:
        for line in f.readlines():
            result = json.loads(line)
            line_drugs = set()
            for result_line in result['response']['body']['choices'][0]['message']['content'].splitlines():
                result_line = result_line.strip(' -*').lower()

                if not result_line or result_line == 'null':
                    continue
                additional_drugs = set()
                if match := re.match(r'.*\((.*)\)', result_line):
                    if ',' in match.groups()[0]:
                        additional_drugs |= {s.strip() for s in match.groups()[0].split(',')}
                    else:
                        additional_drugs.add(match.groups()[0])

                    result_line = result_line.replace(match.groups()[0], '').strip('() ')

                if ',' in result_line:
                    additional_drugs |= {s.strip() for s in result_line.split(',')}
                elif len(result_line):
                    additional_drugs.add(result_line)

                additional_drugs = {ru_drug_map[d] if d in ru_drug_map else d for d in additional_drugs}

                remapped_drugs = set()
                has_remapped_drugs = False
                for drug in additional_drugs:
                    if drug in unmapped_drug_map and drug not in drug_map:
                        remapped_drugs |= set(unmapped_drug_map[drug])
                        has_remapped_drugs = True
                    else:
                        remapped_drugs.add(drug)

                mapped_drugs = set()
                for drug in remapped_drugs:
                    if drug in drug_map:
                        mapped_drugs.add(drug_map[drug])
                    elif has_remapped_drugs:
                        # todo: for now force remapped drugs by gpt
                        mapped_drugs.add(drug)

                # mapped_drugs = {drug_map[d] for d in remapped_drugs if d in drug_map}
                if len(mapped_drugs) > 1:
                    multiple_mappings.append([result['custom_id'], '|'.join(remapped_drugs), '|'.join(mapped_drugs)])
                if not len(mapped_drugs):
                    # if 'ru' in result['custom_id']:
                    #     ru_no_mappings |= additional_drugs
                    # else:
                    if has_remapped_drugs:
                        total_not_found += 1
                    no_mappings.append([result['custom_id'], '|'.join(remapped_drugs)])
                    no_mappings_unique |= remapped_drugs

                augmentation_map |= mapped_drugs
                line_drugs |= mapped_drugs

            drug_mining_map[result['custom_id']] = list(line_drugs)
            drug_mining_map_with_description[result['custom_id']] = [drug_description_map[d] for d in line_drugs if d in drug_description_map]

    with open(os.path.join(data_dir, 'multiple_mappings.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['custom_id', 'input_drugs', 'mapped_drugs'])
        for line in multiple_mappings:
            writer.writerow(line)

    with open(os.path.join(data_dir, 'no_mappings.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['custom_id', 'input_drugs'])
        for line in no_mappings:
            writer.writerow(line)

    with open(os.path.join(data_dir, 'no_mappings_unique.csv'), 'w') as f:
        f.writelines([l + '\n' for l in no_mappings_unique])

    with open(os.path.join(data_dir, 'ru_no_mappings_unique.csv'), 'w') as f:
        f.writelines([l + '\n' for l in ru_no_mappings])

    with open(os.path.join(data_dir, f'stratified/{augmentation_type}/processed_description.json'), 'w') as f:
        json.dump(drug_mining_map_with_description, f)

    with open(os.path.join(data_dir, f'stratified/{augmentation_type}/processed.json'), 'w') as f:
        json.dump(drug_mining_map, f)

    return augmentation_map


if __name__ == '__main__':
    data_dir = '../data/task1'

    test_df = pd.read_csv(os.path.join(data_dir, 'dev_preprocessed.csv'))
    train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))

    augmentation_type = 'drug_mining2'

    drug_set = build_map(augmentation_type)
    print(len(drug_set))
    with open(os.path.join(data_dir, 'drug_list.json'), 'w') as f:
        json.dump(list(drug_set), f, ensure_ascii=False)
