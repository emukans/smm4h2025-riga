import json
import os
import re


def build_map(augmentation_type):
    id_to_drug_map = {}
    with open(os.path.join(data_dir, f'{augmentation_type}/payload.jsonl')) as f:
        for line in f.readlines():
            payload = json.loads(line)
            custom_id = payload['custom_id']

            drug_name = re.findall(r'^Drug: (.*)$', payload['body']['messages'][0]['content'], flags=re.M)
            if len(drug_name) != 1:
                raise ValueError('check')
            id_to_drug_map[custom_id] = drug_name[0]


    augmentation_path = os.path.join(data_dir, f'{augmentation_type}/augmentation_result.jsonl')
    translation_map = {}
    with open(augmentation_path, 'r') as f:
        for line in f.readlines():
            ade_list = []
            result = json.loads(line)
            translated_drug = result['response']['body']['choices'][0]['message']['content'].strip(' -').lower()
            if translated_drug == 'null':
                continue

            for ade in translated_drug.splitlines():
                ade_list.append(ade.strip(' -'))

            translation_map[id_to_drug_map[result['custom_id']]] = ade_list

    with open(os.path.join(data_dir, 'drug_ade.json'), 'w') as f:
        json.dump(translation_map, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    data_dir = '../data/task1'

    augmentation_type = 'ade_mining'

    drug_map = build_map(augmentation_type)
