import json

from tqdm import tqdm
import xmlschema
import xml.etree.ElementTree as ET

xsd = xmlschema.XMLSchema("../data/task1/drugbank.xsd")
# with open('../data/task1/full database.xml', 'r') as f:
#     xt = ET.fromstring(f.read())

# tree = ET.parse('../data/task1/single_drug.xml')
tree = ET.parse('../data/task1/full database.xml')
root = tree.getroot()

drug_map = {}

for drug in tqdm(root):
    drug_name = drug.find('name')
    # if drug_name.text.strip() in drug_map:
    #     raise Exception(f'Duplicate: {drug_name.text.strip()}')
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
