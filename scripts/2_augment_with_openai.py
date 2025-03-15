import csv
import json
import os
from glob import glob
import dotenv

import openai
from tqdm import tqdm


dotenv.load_dotenv()


translation_prompt = """Translate to English. Output just the result of the translation without any supplementary text. Keep the original semantic, orthography and punctuation.

Text for translation:
{text}
"""

translation_summarization_prompt = """Summarize and translate to English. Output just the result of the translation without any supplementary text. Keep the original semantic, orthography and punctuation. The summarization should focus on detection of adverse drug events. Irrelevant  and formal information, such as greetings, closing, etc, could be omitted. Keep the named and nominal and named entities related to drugs, symptoms and drug effects. The output should be up to 5 sentences.

Text to process:
{text}
"""

drug_mining = """Extract a list of drugs that a mentioned in the text. If there are multiple options for a valid drug name, then provide them in brackets as comma-separated. If no drug in the text, then output null. The output should be provided as a list where each drug is on a new line. Every line starts with a bullet point *

Text to process:
{text}
"""

drug_description_mining = """Extract a list of drugs that a mentioned in the text. Provide up to 3 sentences of description for every drug found. In the description mention the type and class of the drug, the purpose of the drug (used to diagnose, cure, treat, or prevent disease), whether is it a prescription or over-the-counter drug, and any other relevant information. If no drug in the text, then output null. The output should be provided as a list where each drug is on a new line. Every line starts with a bullet point *

Text to process:
{text}
"""


if __name__ == '__main__':
    source_path = '../data/task1/stratified'

    # task_type = 'translation'
    # task_type = 'translate_summarize'
    task_type = 'drug_mining'
    dataset_path = os.path.join(source_path, task_type)
    os.makedirs(dataset_path, exist_ok=True)
    full_json = {}
    split_list = ['dev', 'train']
    stratification_type_list = ['de', 'fr', 'ru', 'en']
    # stratification_type_list = ['forum post', 'review']
    for split in split_list:
        for stratify_by in stratification_type_list:
            # with open(os.path.join(source_path, f'{split}_type_{stratify_by}.json'), 'r') as f:
            with open(os.path.join(source_path, f'{split}_language_{stratify_by}.json'), 'r') as f:
                full_json.update(json.load(f))

    with open(os.path.join(dataset_path, 'source.json'), 'w') as f:
        json.dump(full_json, f)

    print(len(full_json))

    with open(os.path.join(dataset_path, 'payload.jsonl'), 'w') as f:
        for tweet_id, text in tqdm(list(full_json.items())):
            json.dump({"custom_id": tweet_id, "method": "POST", "url": "/v1/chat/completions",
                       "body": {
                           # "model": "gpt-4o-mini",
                           "model": "gpt-4o",
                           # "model": "gpt-3.5-turbo-0125",
                           "messages": [
                               {
                                   "role": "user",
                                   "content": drug_mining.format(text=text)
                               }
                           ],
                           "max_tokens": 128,
                           "temperature": 0,
                           "top_p": 1,
                           "frequency_penalty": 0,
                           "presence_penalty": 0}}, f)
            f.write('\n')

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
            "description": "Drug mining"
        }
    )

    print(response)

    with open(os.path.join(dataset_path, 'response.json'), 'w') as f:
        f.write(response.json())
