import json
import os


def build_map(augmentation_type):
    augmentation_path = os.path.join(data_dir, f'{augmentation_type}/augmentation_result.jsonl')
    translation_map = {}
    with open(augmentation_path, 'r') as f:
        for line in f.readlines():
            result = json.loads(line)
            translated_drug = result['response']['body']['choices'][0]['message']['content'].strip().lower()
            if translated_drug == 'null':
                continue

            translation_map[result['custom_id']] = translated_drug

    result_map = {}
    # with open(os.path.join(data_dir, f'{augmentation_type}/source.csv')) as f:
    with open(os.path.join(data_dir, f'{augmentation_type}/source.json')) as f:
        for i, line in enumerate(json.load(f)):
            if f'drug_{i}' not in translation_map:
                continue

            # result_map[line.strip().lower()] = [s.strip() for s in translation_map[f'drug_{i}'].lower().split(',')]
            result_map[line.strip().lower()] = translation_map[f'drug_{i}'].lower()

    # with open(os.path.join(data_dir, 'unmapped_drug_map.json'), 'w') as f:
    with open(os.path.join(data_dir, 'drug_description.json'), 'w') as f:
        json.dump(result_map, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    data_dir = '../data/task1'

    # augmentation_type = 'drug_mapping'
    augmentation_type = 'drug_description_mining'

    drug_map = build_map(augmentation_type)
