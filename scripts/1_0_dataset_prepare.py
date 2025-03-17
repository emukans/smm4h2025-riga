import csv
import json
import os
import re
import emoji


def normalize_string(s):
    s = s.strip().lstrip('"').rstrip('"')
    s = re.sub(r'\s{2,}', ' ', s, flags=re.I)
    s = re.sub(r'@user_*', '[user]', s, flags=re.I)
    s = re.sub(r'httpurl_*', '[url]', s, flags=re.I)
    s = re.sub(r'(<user>)+', r'[user]', s, flags=re.I)
    s = re.sub(r'(<url>)+', r'[url]', s, flags=re.I)
    s = re.sub(r'(@\w+)+', r'[user]', s, flags=re.I)
    s = re.sub(r'https?://(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)', '[url]', s, flags=re.I)

    s = re.sub(r'(\[url]\s+)+', r'[url] ', s, flags=re.I)
    s = re.sub(r'(\[user]\s+)+', r'[user] ', s, flags=re.I)
    s = emoji.demojize(s)

    return s.strip()


def preprocess_data(data_path, id_prefix):
    id_list = []
    text_list = []
    file_path, ext = os.path.splitext(data_path)
    file_path += '_preprocessed' + ext
    duplicate_count = 0

    with open(data_path, 'r') as fr, open(file_path, 'w') as fw:
        reader = csv.DictReader(fr)
        writer = csv.DictWriter(fw, reader.fieldnames)
        writer.writeheader()

        for line in reader:
            if line['id'] in id_list:
                raise Exception(f"Duplicate tweet: {line['id']}")

            line['text'] = normalize_string(line['text'].strip().strip('"'))
            line['id'] = id_prefix + '_' + line['id']
            if not line['text']:
                print(f'Empty text: {line["id"]}')
                continue

            if line['text'] in text_list:
                print('Duplicate text: ', line['id'], line['text'])
                duplicate_count += 1
                continue

            writer.writerow(line)
            id_list.append(line['id'])
            text_list.append(line['text'])

    print(f'Duplicate cound for {id_prefix}: {duplicate_count}')


if __name__ == '__main__':
    train_tweets_data_path = '../data/task1/train.csv'
    dev_tweets_data_path = '../data/task1/dev.csv'

    preprocess_data(train_tweets_data_path, 'train')
    preprocess_data(dev_tweets_data_path, 'dev')
