import json
import os
import io
import xml.etree.ElementTree as ET
from tqdm import tqdm


if __name__ == '__main__':
    data_dir = '../data/task1'

    with open(os.path.join(data_dir, 'drug_list.json'), 'r') as f:
        drug_list = json.load(f)

    # tree = ET.parse('../data/task1/drug.xml')
    tree = ET.parse('../data/task1/full database.xml')
    root = tree.getroot()

    classification_map = {}
    description_map = {}

    i = 0
    for drug in tqdm(root):
        drug_name = drug.find('name')

        formatted_name = drug_name.text.strip().lower()
        if formatted_name not in drug_list:
            continue

        # i += 1
        # if i == 2:
        #     exit()

        classification_desc = drug.find('classification/description')
        if classification_desc is None:
            classification_desc = ''
        else:
            classification_desc = classification_desc.text if classification_desc.text is not None else ''
        classification_map[formatted_name] = classification_desc.lower().strip()

        description = drug.find('description')
        if description is None:
            description = ''
        else:
            description = description.text if description.text is not None else ''
        description_map[formatted_name] = description.lower().strip()

        # print(ET.ElementTree(ET.Element('drugbank').append(drug)))


        # drugbank = ET.Element('drugbank')
        # drugbank.append(drug)
        # ET.ElementTree(drugbank).write('../data/task1/drug.xml', encoding="utf-8")
        # exit()

    # drug-interactions/drug-interaction/name
    # add dataset for interactions with other drugs

    # food-interactions/food-interaction
    # add dataset for interactions with food

    with open(os.path.join(data_dir, 'drug_description.json'), 'w') as f:
        json.dump(description_map, f, indent=4, ensure_ascii=False)

    with open(os.path.join(data_dir, 'drug_classification.json'), 'w') as f:
        json.dump(classification_map, f, indent=4, ensure_ascii=False)
