import csv
import json
import os
from glob import glob
import dotenv
import tiktoken

from tqdm import tqdm


dotenv.load_dotenv()

import openai

translation_prompt = """Translate to English. Output just the result of the translation without any supplementary text. Keep the original semantic, orthography and punctuation.

Text for translation:
{text}
"""

translation_advanced_prompt = """Translate to English and paraphrase highlighting already happened adverse drug events. You are also provided with additional context of drug names mentioned in the text, corresponsing drug description and adverse drug effects that could cause the drug. Output just the result of the translation and paraphasing without any supplementary text. 
Context:
{context}

Text for translation:
{text}
"""

paraphrase_advanced_prompt = """Paraphrase highlighting already happened adverse drug events. You are also provided with additional context of drug names mentioned in the text, corresponsing drug description and adverse drug effects that could cause the drug. Output just the result of the paraphasing without any supplementary text. 
Context:
{context}

Text for paraphrasing:
{text}
"""

translation_summarization_prompt = """Summarize and translate to English. Output just the result of the translation without any supplementary text. Keep the original semantic, orthography and punctuation. The summarization should focus on detection of adverse drug events. Irrelevant  and formal information, such as greetings, closing, etc, could be omitted. Keep the named and nominal and named entities related to drugs, symptoms and drug effects. The output should be up to 5 sentences.

Text to process:
{text}
"""

translation_summarization_advanced_prompt = """You are provided with a text. Summarize and translate it to English. You are also provided with additional context of drug names mentioned in the text, corresponsing drug description and adverse drug effects that could cause the drug. Output just the result of the translation without any supplementary text. Keep the original semantic, orthography and punctuation. The summarization should focus on detection of adverse drug events. Irrelevant and formal information, such as greetings, closing, etc, could be omitted. Keep the named and nominal and named entities related to drugs, symptoms and drug effects. The output should be up to 5 sentences. The first sentence should list all already happened adverse drug events that the person reports if there are any of them.
Context:
{context}

Text to process:
{text}
"""

ade_mining_advanced_prompt = """You are provided with a text. List all already happened adverse drug effects caused by taking medications. Ignore symptoms that are cured by the mentioned drugs. You are also provided with additional context of drug names mentioned in the text, corresponding drug descriptions, and adverse drug effects that could cause the drug. Output just the list of adverse effects without any supplementary. The output text should be in English. Each line could contain only one adverse effect. If no adverse effects, then output null.
Context:
{context}

Text to process:
{text}
"""


drug_mining = """Extract a list of drugs that a mentioned in the text. If there are multiple options for a valid drug name, then provide them in brackets as comma-separated. If no drug in the text, then output null. The output should be provided as a list where each drug is on a new line. Every line starts with a bullet point *

Text to process:
{text}
"""

drug_description_mining = """You are provided with a drug name or active substance of the drug. Provide up to 3 sentences of description for the drug. In the description mention the type and class of the drug, the purpose of the drug (used to diagnose, cure, treat, or prevent disease), whether is it a prescription or over-the-counter drug, and any other relevant information. Additionally, provide up to 2 sentences of most known or common adverse drug events for the provided drug. If no adverse drug events known, then output just drug description. Both output values should be separated with a line-break.

Text to process:
{text}
"""

ru_drug_name_translate = """Translate drug name to English. Output just the name. If provided text is not a drug, then output null.

Text to process:
{text}
"""

drug_name_mapping = """You are provided with a drug name. Provide a proper medical name or the name of the active element if you are provided with specific brand. Output just the name. If the name unknown output null. If there are multiple elements, then output them comma-separated

Text to process:
{text}
"""

ade_mining = """You are provided with drug name and description information. Summarize possible adverse drug effects or symptoms that could be caused by taking this drug by a human. Take the information from the provided context. Each line in the output could contain only one symptom or effect description. The description should be brief and concise. If not adverse effects or symptoms, then output null.

{text}
"""

tweet_augmentation = """You are provided with data entries. The task is to augment the data and output 5 more data entries with supplemental information. The original source is social media resources, such as Twitter, forums, and reviews. The output should be in English and written in the similar manner as the provided samples. Text entry in the input could have different length, but the text entry in the output should not exceed 3 sentences. Output just the augmentation result without any supplementary text.
[Structure]
Drugs: [comma-separated list of drugs mentioned in the text or null]
Symptoms: [comma-separated list of adverse drug effects, that are caused by taking drug or null]
Text: [a sample of the data]
---
[Samples]
{text}
"""


if __name__ == '__main__':
    stratified_source_path = '../data/task1/test_stratified'
    source_path = '../data/task1'

    encoder = tiktoken.encoding_for_model("gpt-4o")

    # task_type = 'translation2'
    # task_type = 'translate_summarize2'
    # task_type = 'drug_mining2'
    # task_type = 'test_ru_mapping_translate'
    # task_type = 'drug_mapping'
    # task_type = 'drug_description_mining'
    task_type = 'ade_mining'
    # task_type = 'translate_summarize_advanced'
    # task_type = 'translate_advanced'
    # task_type = 'paraphrase_advanced'
    # task_type = 'ade_mining_advanced'
    # task_type = 'train_tweet_augmentation'
    # dataset_path = os.path.join(stratified_source_path, task_type)
    dataset_path = os.path.join(stratified_source_path, task_type)
    os.makedirs(dataset_path, exist_ok=True)
    full_json = {}
    # split_list = ['dev', 'train']
    split_list = ['test']
    # stratification_type_list = ['de', 'fr']
    stratification_type_list = ['de', 'fr', 'ru', 'en']
    # stratification_type_list = ['forum post', 'review']
    for split in split_list:
        for stratify_by in stratification_type_list:
            # with open(os.path.join(stratified_source_path, f'{split}_type_{stratify_by}.json'), 'r') as f:
            with open(os.path.join(stratified_source_path, f'{split}_language_{stratify_by}.json'), 'r') as f:
                full_json.update(json.load(f))

    # with open(os.path.join(source_path, 'no_mappings_unique.csv'), 'r') as f:
    #     full_dataset = [s.strip() for s in f.readlines()]
    with open(os.path.join(dataset_path, 'source.json'), 'w') as f:
        json.dump(full_json, f)

    with open(os.path.join(dataset_path, 'source.json')) as f:
        full_dataset = json.load(f)

    print(len(full_dataset))
    # exit()
    with open(os.path.join(source_path, 'drug_classification.json')) as f:
        drug_classification = json.load(f)

    with open(os.path.join(source_path, 'drug_description.json')) as f:
        drug_description = json.load(f)

    with open(os.path.join(source_path, 'food_interactions.json')) as f:
        food_interactions = json.load(f)

    with open(os.path.join(source_path, 'indication.json')) as f:
        indication = json.load(f)

    with open(os.path.join(source_path, 'mechanism_of_action.json')) as f:
        mechanism_of_action = json.load(f)

    with open(os.path.join(source_path, 'toxicity.json')) as f:
        toxicity = json.load(f)

    with open(os.path.join(source_path, 'known_interaction_description.json')) as f:
        known_interaction_description = json.load(f)

    with open(os.path.join(stratified_source_path, 'drug_mining2', 'processed.json')) as f:
        tweet_drug_map = json.load(f)

    i = 0
    total_input_tokens = 0
    with open(os.path.join(dataset_path, 'payload.jsonl'), 'w') as f:
        for tweet_id, text in tqdm(list(full_dataset.items())):
        # for tweet_id, text in tqdm(enumerate(full_dataset)):
            context = ''
            drug_to_skip = []
            drug_interactions = []

            for drug_name in tweet_drug_map[tweet_id]:
                drug_name = drug_name.lower()
                # if not len(drug_description[drug_name]) and not len(drug_classification[drug_name]):
                #     continue
                i += 1

                # context = f'Drug: {drug_name}\n\nDescription:\n{drug_description[drug_name]}\n\nClassification:\n{drug_classification[drug_name]}\n\n'
                if len(context):
                    context += '\n---\n\n'
                context += f'Drug: {drug_name}\n\n'
                if indication[drug_name]:
                    context += f'Indication:\n{indication[drug_name]}\n\n'

                if food_interactions[drug_name]:
                    context += f'Food Interactions:\n{"\n".join(food_interactions[drug_name])}\n\n'

                if mechanism_of_action[drug_name]:
                    context += f'Mechanism of action:\n{mechanism_of_action[drug_name]}\n\n'

                if toxicity[drug_name]:
                    context += f'Toxicity:\n{toxicity[drug_name]}\n\n'

                # if not len(known_interaction_description[drug_name]) or drug_name in drug_to_skip:
                #     continue

                # for drug_to_check in tweet_drug_map[tweet_id]:
                #     if drug_to_check == drug_name or drug_to_check not in known_interaction_description[drug_name]:
                #         continue
                #
                #     drug_interactions.append(known_interaction_description[drug_name][drug_to_check])
                #     drug_to_skip.append(drug_to_check)

            # if len(drug_interactions):
            #     context += f'\nInteractions:\n{"\n".join(drug_interactions)}\n\n'

            if not len(context):
                context = 'No context'
            content = ade_mining_advanced_prompt.format(context=context, text=text)
            # content = ru_drug_name_translate.format(text=text)

            total_input_tokens += len(encoder.encode(content))

            json.dump({"custom_id": tweet_id, "method": "POST", "url": "/v1/chat/completions",
                       "body": {
                           # "model": "gpt-4o-mini",
                           "model": "gpt-4o",
                           # "model": "gpt-3.5-turbo-0125",
                           "messages": [
                               {
                                   "role": "user",
                                   "content": content
                               }
                           ],
                           "max_tokens": 100,
                           "temperature": 0,
                           "top_p": 1,
                           "frequency_penalty": 0,
                           "presence_penalty": 0}}, f, ensure_ascii=False)
            f.write('\n')

    print(len(full_dataset), len(set(full_dataset)), i)
    print('Total input tokens: ', total_input_tokens)
    # exit()
    client = openai.OpenAI()

    batch_input_file = client.files.create(
        file=open(os.path.join(dataset_path, 'payload.jsonl'), "rb"),
        purpose="batch"
    )

    print(batch_input_file.id)
    response = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Test data ade mining"
        }
    )

    print(response)

    with open(os.path.join(dataset_path, 'response.json'), 'w') as f:
        f.write(response.json())
