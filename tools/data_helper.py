import random
from datasets import Dataset
from collections import defaultdict
import os
import torch
import pickle
import numpy as np
from torch import nn
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, concatenate_datasets, Features, Value
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from transformers import AutoTokenizer
from datetime import datetime

def calculate_num_user_feats(input_type, num_tweets, cols_dict):
    type2 = input_type.split('-')[1]
    if type2 == 'in1':
        num_feats = 1
    elif type2 == 'in2':
        num_feats = 2
    elif type2 == 'inN':
        num_feats = 1 + num_tweets
    elif type2 == 'inuser+1':
        num_feats = len(cols_dict['cols_user']) + 1
    elif type2 == 'inuser+N':
        num_feats = len(cols_dict['cols_user']) + num_tweets
    elif type2 == 'noin':
        num_feats = len(cols_dict['cols_user'])
        type1 = input_type.split('-')[0]
        if type1 == 'all':
            num_feats += len(cols_dict['cols_post']) * num_tweets
        elif type1 == 'noposttime':
            num_feats += (len(cols_dict['cols_post']) - len(cols_dict['cols_post_time'])) * num_tweets
        elif type1 == 'nopostmeta':
            num_feats += num_tweets
    return num_feats

def build_geo_dataset_dict(data_dict, cols_dict, num_tweets, model_name, max_len, label_dict, dataset_name, input_type):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    assert input_type.split('-')[0] in ['all', 'noposttime', 'nopostmeta']
    assert input_type.split('-')[1] in ['in1', 'in2', 'inN', 'inuser+1', 'inuser+N', 'noin']
    assert input_type.split('-')[2] in ['text', 'text+cate']

    if input_type.split('-')[0]=='all':
        columns = cols_dict['cols_user']+[f'{col}_{i}' for i in range(1, num_tweets+1) for col in cols_dict['cols_post']]
    elif input_type.split('-')[0]=='noposttime':
        columns = cols_dict['cols_user']+[f'{col}_{i}' for i in range(1, num_tweets+1) for col in cols_dict['cols_post'] if col not in cols_dict['cols_post_time']]
    else:
        columns = cols_dict['cols_user']+[f'{col}_{i}' for i in range(1, num_tweets+1) for col in cols_dict['cols_post_text']]

    if input_type.split('-')[2] == 'text+cate' and dataset_name=='Twitter':
        cate_encoders, cate_embeddings, cate_num_classes = get_cate_encoders_and_embeddings(data_dict, cols_dict['cols_user_cate'], 768 if 'base' in model_name else 1024)
    else:
        cate_encoders, cate_embeddings, cate_num_classes = {}, {}, {}

    dataset_dict = {}
    for key, df in data_dict.items():
        temp_data = df[columns].copy()
        if input_type.split('-')[0]=='all' and dataset_name=='Twitter':
            for col in [col for col in columns if 'created_at_' in col]:
                temp_data[col] = temp_data[col].apply(lambda x: str(datetime.fromtimestamp(x / 1000)) if x > 0 else None)
        if dataset_name=='Twitter': temp_data['user_created_at'] = temp_data['user_created_at'].apply(lambda x: str(x))
        temp_data = temp_data.fillna(' ')
        if input_type.split('-')[2] == 'text+cate' and dataset_name=='Twitter':
            for col in cols_dict['cols_user_cate']:
                temp_data[col] = cate_encoders[col].transform(temp_data[col])
        labels = df['label'].apply(lambda x: label_dict[x])
        ds = GeoDataset(temp_data, labels, tokenizer, cate_embeddings, max_len, input_type, cols_dict, num_tweets)
        dataset_dict[key] = ds
    return dataset_dict, columns, cate_num_classes

class GeoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, cate_embeddings, max_len, input_type, cols_dict, num_tweets):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.cate_embeddings = cate_embeddings
        self.max_len = max_len
        self.input_type = input_type
        self.cols_dict = cols_dict
        self.num_tweets = num_tweets
    def __len__(self):
        return len(self.texts)

    def combine_columns(self, cols, idx):
        return ', '.join([self.texts.loc[idx, col] for col in cols])

    def tokenize_text(self, text):
        return self.tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

    def get_category_features(self, idx):
        cate_feats = []
        for col, embedding_dict in self.cate_embeddings.items():
            embed = embedding_dict(torch.tensor(self.texts[col][idx]))
            cate_feats.append(embed)
        return torch.stack(cate_feats)

    def tokenize_user_with_text_category(self, idx):
        if self.input_type.split('-')[2] == 'text+cate':
            # input_category = self.get_category_features(idx)
            input_category = torch.tensor([self.texts.loc[idx, col] for col in self.cols_dict['cols_user_cate']])
            temp_users = [self.texts.loc[idx, col] for col in self.cols_dict['cols_user'] if col not in self.cols_dict['cols_user_cate']]
        else:
            temp_users = [self.texts.loc[idx, col] for col in self.cols_dict['cols_user']]
            input_category = torch.tensor([])
        input_users = [self.tokenize_text(temp) for temp in temp_users]
        return input_users, input_category

    def extract_inputid_attnmask(self, input_users, input_posts):
        input_users = [input_users] if not isinstance(input_users, list) else input_users
        input_posts = [input_posts] if not isinstance(input_posts, list) else input_posts
        input_ids = torch.stack([i['input_ids'] for i in input_users + input_posts]).squeeze(1)
        attn_mask = torch.stack([i['attention_mask'] for i in input_users + input_posts]).squeeze(1)
        return input_ids, attn_mask

    def combine_columns_for_each_post(self, idx):
        cols_post = [col for col in self.texts.columns if col not in self.cols_dict['cols_user']]
        input_posts = []
        for idx_tweet in range(1, self.num_tweets + 1):
            temp_post = self.combine_columns([col for col in cols_post if str(idx_tweet) in col], idx)
            input_post = self.tokenize_text(temp_post)
            input_posts.append(input_post)
        return input_posts

    def __getitem__(self, idx):
        input_category = torch.tensor([])
        if self.input_type.split('-')[1]=='in1': # all columns into one
            temp = self.combine_columns(self.texts.columns, idx)
            input_temp = self.tokenize_text(temp)
            input_ids = input_temp['input_ids']
            attn_mask = input_temp['attention_mask']
        elif self.input_type.split('-')[1]=='in2': # user-related columns into one, tweet-related columns into one
            temp_user = self.combine_columns(self.cols_dict['cols_user'], idx)
            input_user = self.tokenize_text(temp_user)
            temp_post = self.combine_columns([col for col in self.texts.columns if col not in self.cols_dict['cols_user']], idx)
            input_post = self.tokenize_text(temp_post)
            input_ids, attn_mask = self.extract_inputid_attnmask(input_user, input_post)
        elif self.input_type.split('-')[1]=='inN': # user-related columns into one, columns related to each tweet into one,
            temp_user = self.combine_columns(self.cols_dict['cols_user'], idx)
            input_user = self.tokenize_text(temp_user)
            input_posts = self.combine_columns_for_each_post(idx)
            input_ids, attn_mask = self.extract_inputid_attnmask(input_user, input_posts)
        elif self.input_type.split('-')[1]=='inuser+1': # user columns, tweet-related columns into one
            input_users, input_category = self.tokenize_user_with_text_category(idx)
            temp_post = self.combine_columns([col for col in self.texts.columns if col not in self.cols_dict['cols_user']], idx)
            input_post = self.tokenize_text(temp_post)
            input_ids, attn_mask = self.extract_inputid_attnmask(input_users, input_post)
        elif self.input_type.split('-')[1]=='inuser+N': # user columns, each tweet-related columns into one
            input_users, input_category = self.tokenize_user_with_text_category(idx)
            input_posts = self.combine_columns_for_each_post(idx)
            input_ids, attn_mask = self.extract_inputid_attnmask(input_users, input_posts)
        elif self.input_type.split('-')[1]=='noin': # no comibination of columns
            input_users, input_category = self.tokenize_user_with_text_category(idx)
            cols_post = [col for col in self.texts.columns if col not in self.cols_dict['cols_user']]
            temp_posts = [self.texts.loc[idx, col] for col in cols_post]
            input_posts = [self.tokenize_text(temp) for temp in temp_posts]
            input_ids, attn_mask = self.extract_inputid_attnmask(input_users, input_posts)
        else:
            raise ValueError(f'The given input type {self.input_type} is invalid.')

        features = {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'cate_features': input_category,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return features

def beauty_counts(data, colname):
    counts = data[colname].value_counts().to_frame().reset_index().rename(columns={'index': colname, colname: 'count'})
    return counts

def build_fsl_dataset_by_shot_seed(data, num_shot, random_seed, col_label):
    print(f'Build fsl dataset, num shot: {num_shot}, random seed: {random_seed}.')
    random.seed(random_seed)
    index_dic = defaultdict(list)
    for index, item in enumerate(data[col_label]):
        index_dic[item].append(index)
    selected_index = []
    num_valid_class = 0
    for k, v in index_dic.items():
        try:
            samples = random.sample(v, num_shot)
            selected_index.extend(samples)
            num_valid_class+=1
        except ValueError as e:
            continue
    assert num_valid_class == data[col_label].nunique()
    selected_data = dict()
    for c in data.columns:
        selected_data[c] = [data[c][index] for index in selected_index]
    selected_dataset = pd.DataFrame(selected_data).sample(frac=1, random_state=2)
    return selected_dataset

def downsample_to_minimum(data, colname_label, seed=7):
    counts = beauty_counts(data, colname_label)
    min_samples = counts['count'].tolist()[-1]
    # print(f'Minimum samples: {min_samples}')
    balance_list = []
    for l in counts[colname_label].tolist():
        temp = data[data[colname_label]==l].sample(n=min_samples, random_state=seed)
        balance_list.append(temp)
    data_balance = pd.concat(balance_list).reset_index(drop=True).sample(frac=1, random_state=17)
    return data_balance, min_samples

def get_cate_encoders_and_embeddings(data_files, columns, embed_dim):
    cate_encoders, cate_embeddings, cate_num_classes = {}, {}, {}
    for col in columns:
        le = LabelEncoder()
        temp = []
        for key, data in data_files.items(): temp.extend(data.fillna(' ')[col])
        le.fit(temp)
        cate_encoders[col] = le
        cate_embeddings[col] = nn.Embedding(len(le.classes_), embed_dim)
        cate_num_classes[col] = len(le.classes_)
    return cate_encoders, cate_embeddings, cate_num_classes

def split_data_by_category(data, labels, col_label='label', split_props=[0.85, 0.90, 0.10]):
    df = data
    df_trains, df_vals, df_tests = [], [], []
    for label in labels:
        df_temp = df[df[col_label] == label]
        df_shuffled = df_temp.sample(frac=1, random_state=17)
        split_points = [math.ceil(len(df_shuffled) * p) for p in split_props]
        df_trains.append(df_shuffled[:split_points[0]])
        df_vals.append(df_shuffled[split_points[0]:split_points[1]])
        df_tests.append(df_shuffled[split_points[1]:])
    df_train = pd.concat(df_trains).sample(frac=1, random_state=42)
    df_val = pd.concat(df_vals).sample(frac=1, random_state=42)
    df_test = pd.concat(df_tests).sample(frac=1, random_state=42)
    print(f'#samples for train set, val set, test set: {len(df_train)}, {len(df_val)}, {len(df_test)}')
    print(f'#train labels: {len(set(df_train[col_label]))}, #val labels: {len(set(df_val[col_label]))}, #test labels: {len(set(df_test[col_label]))}')
    assert set(df_train[col_label]) == set(df_val[col_label]) == set(df_test[col_label])
    return df_train, df_val, df_test

def get_columns(dataset_name):
    if dataset_name=='Twitter':
        cols_dict = {
        'cols_user': ['user_name', 'user_screen_name', 'user_description', 'user_created_at', 'user_location', 'user_lang', 'user_time_zone'],
        'cols_user_cate': ['user_lang', 'user_time_zone'],
        'cols_post': ['text', 'created_at', 'source', 'hashtags'],
        'cols_post_time': ['created_at'],
        'cols_post_text': ['text'],
        'cols_label': ['label', 'label_lat', 'label_long'],
        }
        colname_user = 'user_id'
    else:
        cols_dict = {
        'cols_user': ['user_name', 'profile_description', 'occupation', 'hometown', 'country', 'join_at'],
        'cols_user_cate': [],
        'cols_post': ['title', 'description', 'user_tag', 'machine_tag', 'device', 'take_at', 'upload_at'],
        'cols_post_time': ['take_at', 'upload_at'],
        'cols_post_text': ['title', 'description'],
        'cols_label': ['label'],
        }
        colname_user = 'user_ID'
    return cols_dict

def load_and_process_data_old(dataset_name, path, min_label_counts, max_shot, split_ratios, do_fsl, do_train, num_run, num_shot, fsl_seeds, colname_label='label'):
    files_dict = {'Twitter': 'wnut16-tweet-25peruser-reshape.json',
                  'Flickr': 'yfcc100m-us-25peruser-reshape.csv'}
    file = files_dict[dataset_name]
    assert max_shot >= num_shot
    # load data
    data_file = os.path.join(path, 'data', file)
    data = read_file(data_file)
    print(f'Original dataset: {len(data)} samples.')
    # keep major labels
    count_user = data[colname_label].value_counts().reset_index(drop=False).rename(columns={'index': colname_label, colname_label: 'count'})
    major_labels = count_user[count_user['count'] >= min_label_counts][colname_label].tolist()
    data = data[data[colname_label].isin(major_labels)].reset_index(drop=True)
    print(f'min_label_counts: {min_label_counts}, number of major labels: {len(major_labels)}, number of samples with major labels: {len(data)}')
    # get few-shot subsets
    fsl_labels = count_user[count_user['count'] >= math.ceil(max_shot / split_ratios[0])][colname_label].tolist()
    data_fsl = data.copy()
    if len(fsl_labels) < len(major_labels):
        data_fsl = data_fsl[data_fsl[colname_label].isin(fsl_labels)]
    else:
        fsl_labels = major_labels
    print(f'number of samples in few-shot subsets: {len(data_fsl)}')
    # split data
    df_train, df_val, df_test = split_data_by_category(data_fsl, fsl_labels, col_label=colname_label, split_props=split_ratios)
    if do_fsl:
        df_val, min_samples = downsample_to_minimum(df_val, colname_label)
        print(f'#samples in val set for few-shot training: {len(df_val)}, #samples per class: {min_samples}')
    data_dicts = {'val': df_val.reset_index(drop=True), 'test': df_test.reset_index(drop=True)}
    if do_train:
        if do_fsl:
            df_train = df_train.reset_index(drop=True)
            for i, seed in enumerate(fsl_seeds):
                if i >= num_run: break
                temp = build_fsl_dataset_by_shot_seed(df_train, num_shot, seed, colname_label)
                data_dicts[f'train_fsl_{i}'] = temp.reset_index(drop=True)
        else:
            data_dicts['train'] = df_train.reset_index(drop=True)

    label_dict = {label: i for i, label in enumerate(fsl_labels)}
    return data_dicts, label_dict

def load_and_process_data(dataset_name, path, min_label_counts, max_shot, split_ratios, do_fsl, do_train, num_run, num_shot, fsl_seeds, colname_label='label'):
    files_dict = {'Twitter': 'wnut16-tweet-25peruser-reshape.json',
                  'Flickr': 'yfcc100m-us-25peruser-reshape.csv'}
    file = files_dict[dataset_name]
    assert max_shot >= num_shot
    # load data
    data_file = os.path.join(path, 'data', file)
    data = read_file(data_file)
    print(f'Original dataset: {len(data)} samples.')
    # keep major labels
    count_user = data[colname_label].value_counts().reset_index(drop=False).rename(columns={'index': colname_label, colname_label: 'count'})
    major_labels = count_user[count_user['count'] >= min_label_counts][colname_label].tolist()
    data = data[data[colname_label].isin(major_labels)].reset_index(drop=True)
    print(f'min_label_counts: {min_label_counts}, number of major labels: {len(major_labels)}, number of samples with major labels: {len(data)}')
    # get few-shot subsets
    fsl_labels = count_user[count_user['count'] >= math.ceil(max_shot / split_ratios[0])][colname_label].tolist()
    data_fsl = data.copy()
    if len(fsl_labels) < len(major_labels):
        data_fsl = data_fsl[data_fsl[colname_label].isin(fsl_labels)]
    else:
        fsl_labels = major_labels
    print(f'number of samples in few-shot subsets: {len(data_fsl)}')
    # split data
    df_train, df_val, df_test = split_data_by_category(data_fsl, fsl_labels, col_label=colname_label, split_props=split_ratios)
    if do_fsl:
        df_val, min_samples = downsample_to_minimum(df_val, colname_label)
        print(f'#samples in val set for few-shot training: {len(df_val)}, #samples per class: {min_samples}')
    data_dicts = {'val': df_val.reset_index(drop=True), 'test': df_test.reset_index(drop=True)}
    if do_train:
        if do_fsl:
            df_train = df_train.reset_index(drop=True)
            for i, seed in enumerate(fsl_seeds):
                if i >= num_run: break
                temp = build_fsl_dataset_by_shot_seed(df_train, num_shot, seed, colname_label)
                data_dicts[f'train_fsl_{i}'] = temp.reset_index(drop=True)
        else:
            data_dicts['train'] = df_train.reset_index(drop=True)

    label_dict = {label: i for i, label in enumerate(fsl_labels)}
    label_latlng_dict = None
    if dataset_name == 'Twitter':
        label_latlng_dict = {label_dict[data.loc[idx, 'label']]: (data.loc[idx, 'label_lat'], data.loc[idx, 'label_long']) for idx in data[data['label'].isin(fsl_labels)].index}
    return data_dicts, label_dict, label_latlng_dict

def build_fsl_dataset(data, num_shot, random_seed):
    # select num_shot samples randomly from each class, to construct training dataset for few-shot learning
    print(f'Build fsl dataset, random seed: {random_seed}.')
    random.seed(random_seed)

    index_dic = defaultdict(list)
    for index, item in enumerate(data['label']):
        index_dic[item].append(index)

    selected_index = []
    for k, v in index_dic.items():
        selected_index.extend(random.sample(v, num_shot))

    selected_data = dict()
    for c in data.column_names:
        selected_data[c] = [data[c][index] for index in selected_index]
    selected_dataset = Dataset.from_dict(selected_data)

    return selected_dataset

def merge_cols(df, cols, merged_col_name, add_colname=False):
    data = df.copy()
    data[merged_col_name] = data[cols[0]].astype('str') if not add_colname else [f'{cols[0]}: ']*len(data)+data[cols[0]].astype('str')
    if len(cols) == 1:
        print('Only 1 column, no need to merge.')
    else:
        if not add_colname:
            seps = [', '] * len(data)
            for c in cols[1:]:
                try:
                    data[merged_col_name] += (seps + data[c])
                except TypeError:
                    data[c] = data[c].astype('str')
                    data[merged_col_name] += (seps + data[c])
        else:
            for c in cols[1:]:
                try:
                    data[merged_col_name] += ([f', {c}: ']*len(data) + data[c])
                except TypeError:
                    data[c] = data[c].astype('str')
                    data[merged_col_name] += ([f', {c}: ']*len(data) + data[c])
    return data

def get_all_texts(data_dict, columns):
    all_texts = []
    for k, data in data_dict.items():
        temp = data.fillna(' ')
        for col in columns:
            text = [t.lower() for t in temp[col]]
            all_texts.extend(text)
    return all_texts

def get_tfidf_vectorizer_with_vocab_check(max_features, min_df, all_texts):
    vocab_size = len(TfidfVectorizer(min_df=min_df).fit(all_texts).vocabulary_)
    while vocab_size<max_features:
        min_df -= 0.001
        vocab_size = len(TfidfVectorizer(min_df=min_df).fit(all_texts).vocabulary_)
    print(f'TFIDF vectorizer, vocab size: {vocab_size}, min_df: {min_df}')
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df).fit(all_texts)
    return vectorizer

def extract_filename(files, ver, do_fsl, prefer_format='json'):
    assert ver in ['train.', 'val.', 'test.']
    temp_list = [f for f in files if ver in f ]
    if ver=='val.':
        temp_list = [f for f in temp_list if 'fsl' in f] if do_fsl else [f for f in temp_list if 'fsl' not in f]
    if len(temp_list)>1:
        temp = [f for f in temp_list if prefer_format in f][0]
    else:
        temp = temp_list[0]
    return temp

def read_file(datafile):
    file_format = datafile.split('.')[-1]
    assert file_format in ['csv', 'json']
    if file_format=='csv':
        return pd.read_csv(datafile).reset_index(drop=True)
    else:
        return pd.read_json(datafile).reset_index(drop=True)

def load_datafiles(data_ver, data_ver_2, data_ver_3, data_ver_4, data_ver_5, do_fsl, do_train, num_shot, seeds, num_run, path=''):
    data_path = os.path.join(path, 'data', data_ver)
    files = [f for f in os.listdir(data_path) if 'csv' in f or 'json' in f]
    files = [f for f in files if data_ver_2 in f]
    files = [f for f in files if data_ver_3 in f]
    files = [f for f in files if data_ver_4 in f]
    files = [f for f in files if data_ver_5 in f]

    train_filename = extract_filename(files, 'train.', do_fsl, prefer_format='csv')
    val_filename = extract_filename(files, 'val.', do_fsl, prefer_format='csv')
    test_filename = extract_filename(files, 'test.', do_fsl, prefer_format='csv')

    train_file = os.path.join(data_path, train_filename)
    val_file = os.path.join(data_path, val_filename)
    test_file = os.path.join(data_path, test_filename)

    data_files = {'val': val_file, 'test': test_file}

    if do_train:
        if do_fsl:
            fsl_file = os.path.join(data_path, f'shot-{num_shot}', train_filename.replace('.csv', '').replace('.json', ''))
            for i, seed in enumerate(seeds[:num_run]):
                data_files[f'train_fsl_{i}'] = f'{fsl_file}-s{num_shot}r{seed}.csv'
            return data_files, data_files['train_fsl_0'], data_files['val'], data_files['test']
        else:
            data_files['train'] = train_file
            return data_files, data_files['train'], data_files['val'], data_files['test']
    else:
        return data_files, '', data_files['val'], data_files['test']

def load_glove(glove_file, embedding_dim):
    print('Load glove embedding files.')
    path = 'saved_glove'
    filename = 'glove-twitter-27b-200d.p'
    file = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    if os.path.exists(file):
        word_embeddings = pickle.load(open(file, 'rb'))
    else:
        word_embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs
        unknown_word_vector = torch.randn(embedding_dim)
        word_embeddings['<unk>'] = unknown_word_vector
        pickle.dump(word_embeddings, open(file, 'wb'))
    return word_embeddings