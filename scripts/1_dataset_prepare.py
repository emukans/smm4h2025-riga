import csv
import json
import os
import re


def normalize_string(s):
    s = s.strip().lstrip('"').rstrip('"')
    s = re.sub(r'@user_*', '[user]', s, flags=re.I)
    s = re.sub(r'httpurl_*', '[url]', s, flags=re.I)
    s = re.sub(r'(\[user]\s)+', r'[user] ', s, flags=re.I)
    s = re.sub(r'(\[url]\s)+', r'[url] ', s, flags=re.I)
    s = re.sub(r'(<user>\s)+', r'[user] ', s, flags=re.I)
    s = re.sub(r'(<url>\s)+', r'[url] ', s, flags=re.I)
    s = re.sub(r'@\w+', r'[user] ', s, flags=re.I)

    return s


def build_tweet_dict(data_path):
    id_list = []
    file_path, ext = os.path.splitext(data_path)
    file_path += '_preprocessed' + ext

    with open(data_path, 'r') as fr, open(file_path, 'w') as fw:
        reader = csv.DictReader(fr)
        writer = csv.DictWriter(fw, reader.fieldnames)
        writer.writeheader()

        for line in reader:
            if line['id'] in id_list:
                raise Exception(f"Duplicate tweet: {line['id']}")

            line['text'] = normalize_string(line['text'].strip().strip('"'))
            writer.writerow(line)
            id_list.append(line['id'])


if __name__ == '__main__':
    train_tweets_data_path = '../data/task1/train.csv'
    dev_tweets_data_path = '../data/task1/dev.csv'

    build_tweet_dict(train_tweets_data_path)
    build_tweet_dict(dev_tweets_data_path)
