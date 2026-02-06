from sklearn import preprocessing
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def transform_labels(datasets, col_label='label'):
    if 'concat_poi' in datasets.column_names['val']:
        labels_orig = []
        classnames = []
        for key in datasets.keys():
            labels_orig.extend(datasets[key][col_label])
            classnames.extend(datasets[key]['concat_poi'])
        label_name = {label: name for label, name in zip(labels_orig, classnames)}
        label_dict = {int(label): int(i) for i, label in enumerate(label_name.keys())}
        logger.info("The number of classes:  %s", {len(label_dict)})
        return label_dict, list(label_name.values())
    else:
        labels_orig = []
        for key in datasets.keys():
            labels_orig.extend(datasets[key][col_label])
        label_dict = {label:i for i, label in enumerate(set(labels_orig))}
        logger.info("The number of classes:  %s", {len(label_dict)})
        return label_dict, list(label_dict.keys())


def get_classnames(label_data, label_dict):
    df = pd.read_csv(label_data, usecols=['poiName', 'poiID'])
    id_name = {i: name for name, i in zip(df['poiName'], df['poiID'])}
    classnames = [id_name[i] for i in label_dict.keys()]
    return classnames


def prompt_encode_labels(classnames, templates, tokenizer):
    labels = sum([[tpt.format(label) for tpt in templates] for label in classnames], [])
    encoded_labels = tokenizer.batch_encode_plus(labels, return_tensors='pt', padding=True)
    return encoded_labels

def reindex_list(lst):
        index_mapping = {}
        new_lst = []
        index = 0
        for elem in lst:
            if elem not in index_mapping:
                index_mapping[elem] = index
                index += 1
            new_lst.append(index_mapping[elem])
        return new_lst
