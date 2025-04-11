import json
import os


def build_map(augmentation_type):
    augmentation_path = os.path.join(data_dir, f'{augmentation_type}/augmentation_result.jsonl')
    translation_map = {}
    with open(augmentation_path, 'r') as f:
        for line in f.readlines():
            result = json.loads(line)
            translated_drug = result['response']['body']['choices'][0]['message']['content'].strip()
            if len(translated_drug.split()) > 4 or translated_drug.lower() == 'null':
                continue
            translation_map[result['custom_id']] = translated_drug

    result_map = {}
    with open(os.path.join(data_dir, f'{augmentation_type}/source.csv')) as f:
        for i, line in enumerate(f.readlines()):
            if f'test_{i}' not in translation_map:
                continue
            result_map[line.strip().lower()] = translation_map[f'test_{i}'].lower()

    with open(os.path.join(data_dir, 'ru_drug_map.json'), 'r') as f:
        drug_map = json.load(f)
    drug_map.update(result_map)

    with open(os.path.join(data_dir, 'ru_drug_map2.json'), 'w') as f:
        json.dump(drug_map, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    data_dir = '../data/task1'

    augmentation_type = 'test_ru_mapping_translate'

    drug_map = build_map(augmentation_type)
