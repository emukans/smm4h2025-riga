import json

from tqdm import tqdm
import xmlschema
import xml.etree.ElementTree as ET

xsd = xmlschema.XMLSchema("../data/task1/drugbank.xsd")
# with open('../data/task1/full database.xml', 'r') as f:
#     xt = ET.fromstring(f.read())

# tree = ET.parse('../data/task1/single_drug.xml')
# tree = ET.parse('../data/task1/drug.xml')
tree = ET.parse('../data/task1/full database.xml')
root = tree.getroot()

drug_map = {}
known_interactions = {}
food_interactions = {}
known_interaction_description = {}
# drug-interaction/description

mechanism_of_action = {}
# mechanism-of-action

indication = {}
toxicity = {}
# toxicity **Overdose information**

for drug in tqdm(root):
    drug_name = drug.find('name')

    # if drug_name.text.lower().strip() != 'metformin':
    #     continue
    #
    # print('FOUND')
    # element = ET.Element('drugbank')
    # element.append(drug)
    # ET.ElementTree(element).write('../data/task1/drug.xml')
    # exit()
    # if drug_name.text.strip() in drug_map:
    #     raise Exception(f'Duplicate: {drug_name.text.strip()}')

    known_interactions[drug_name.text.strip()] = []
    for interactions in drug.findall('drug-interactions'):
        known_interaction_description[drug_name.text.strip()] = {}
        for interaction in interactions.findall('drug-interaction'):
            known_interactions[drug_name.text.strip()].append(interaction.find('name').text.strip())
            known_interaction_description[drug_name.text.strip()][interaction.find('name').text.strip()] = interaction.find('description').text.strip()

    food_interactions[drug_name.text.strip()] = []
    for interactions in drug.findall('food-interactions'):
        for interaction in interactions.findall('food-interaction'):
            food_interactions[drug_name.text.strip()].append(interaction.text.strip())

    indication[drug_name.text.strip()] = drug.find('indication').text.strip() if drug.find('indication').text else ''
    toxicity[drug_name.text.strip()] = drug.find('toxicity').text.strip() if drug.find('toxicity').text else ''
    mechanism_of_action[drug_name.text.strip()] = drug.find('mechanism-of-action').text.strip() if drug.find('mechanism-of-action').text else ''

    drug_map[drug_name.text.strip()] = drug_name.text.strip()

    for synonyms in drug.findall('synonyms'):
        for synonym in synonyms.findall('synonym'):
            if synonym.text.strip() == drug_name.text.strip():
                continue
            if synonym.text.strip() in drug_map:
                raise Exception(f'Duplicate synonym: {synonym.text.strip()}')
            drug_map[synonym.text.strip()] = drug_name.text.strip()

    product_name_list = []
    for products in drug.findall('products'):
        for product in products.findall('product'):
            if product.text.strip() == drug_name.text.strip() or product.text.strip() in product_name_list or not len(product.text.strip()):
                continue

            product_name_list.append(product.text.strip())
            if product.text.strip() in drug_map:
                raise Exception(f'Duplicate product: {product.text.strip()}')

            drug_map[product.text.strip()] = drug_name.text.strip()

with open('../data/task1/full_name_map.json', 'w') as f:
    f.write(json.dumps(drug_map, ensure_ascii=False, indent=4).lower())

with open('../data/task1/known_interactions.json', 'w') as f:
    f.write(json.dumps(known_interactions, ensure_ascii=False, indent=4).lower())

with open('../data/task1/food_interactions.json', 'w') as f:
    f.write(json.dumps(food_interactions, ensure_ascii=False, indent=4).lower())

with open('../data/task1/known_interaction_description.json', 'w') as f:
    f.write(json.dumps(known_interaction_description, ensure_ascii=False, indent=4).lower())

with open('../data/task1/mechanism_of_action.json', 'w') as f:
    f.write(json.dumps(mechanism_of_action, ensure_ascii=False, indent=4).lower())

with open('../data/task1/indication.json', 'w') as f:
    f.write(json.dumps(indication, ensure_ascii=False, indent=4).lower())

with open('../data/task1/toxicity.json', 'w') as f:
    f.write(json.dumps(toxicity, ensure_ascii=False, indent=4).lower())
