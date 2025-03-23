import csv
import json
import os

if __name__ == '__main__':
    data_dir = '../data/task1'
    drug_map = {}
    with open(os.path.join(data_dir, 'drugbank vocabulary.csv')) as f:
        reader = csv.DictReader(f)
        for line in reader:
            line['Common name'] = line['Common name'].lower()
            line['Synonyms'] = line['Synonyms'].lower()
            drug_map[line['Common name']] = line['Common name']
            if not line['Synonyms']:
                continue
            for synonym in line['Synonyms'].split('|'):
                drug_map[synonym.strip()] = line['Common name']

    with open(os.path.join(data_dir, 'drugbank_map.json'), 'w') as f:
        json.dump(drug_map, f, indent=4)
