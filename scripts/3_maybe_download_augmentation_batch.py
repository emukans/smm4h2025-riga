import json
import os

import openai
import dotenv


dotenv.load_dotenv()


if __name__ == '__main__':
    source_path = '../data/task1/test_stratified'
    # source_path = '../data/task1'
    # task_type = 'translation2'
    # task_type = 'translate_summarize2'
    # task_type = 'drug_mining2'
    # task_type = 'ru_mapping_translate'
    # task_type = 'test_ru_mapping_translate'
    # task_type = 'drug_mapping'
    # task_type = 'drug_description_mining'
    task_type = 'ade_mining'
    # task_type = 'translate_summarize_advanced'
    # task_type = 'translate_advanced'
    # task_type = 'paraphrase_advanced'
    # task_type = 'ade_mining_advanced'
    # task_type = 'train_tweet_augmentation'

    response_path = os.path.join(source_path, task_type, 'response.json')
    with open(response_path, 'r') as f:
        response = json.load(f)

    batch_id = response['id']

    client = openai.OpenAI()
    # response = client.batches.list()
    response = client.batches.retrieve(batch_id)

    print(response.model_dump())
    print('Status: ', response.status)
    print('Completed: ', response.request_counts.completed / response.request_counts.total)
    print('Output file: ', response.output_file_id)

    if response.output_file_id is not None:
        file_response = client.files.content(response.output_file_id)

        with open(os.path.join(os.path.dirname(response_path), 'augmentation_result.jsonl'), 'w') as f:
            f.write(file_response.text)
