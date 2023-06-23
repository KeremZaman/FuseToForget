import datasets
from datasets import Dataset, DatasetDict
import random
import os

import pickle
import argparse
from typing import List, Optional

ATTRIBUTES = {
    'sentiment': ['positive', 'negative'],
    'gender': ['male', 'female'],
    'age': ['young', 'old']
}

VALUES_TO_ATTRIBUTES = {v:k for k, values in ATTRIBUTES.items() for v in values}

def _split_by_ratio(dataset, attribute, ratio):
    if ratio >= 0.5:
        v1, v2 = 0, 1
    else:
        v1, v2 = 1, 0
        ratio = 1-ratio
    
    part_1 = list(filter(lambda example: example[attribute] == v1, dataset))
    part_2 = list(filter(lambda example: example[attribute] == v2, dataset))
    
    k = int(ratio / (1-ratio))
    n1, n2 = len(part_1), len(part_2)
    unit_size = (n1 + n2) // (k+1)

    if unit_size > n2:
        unit_size = n2

    if k*unit_size > n1:
        unit_size = n1 // k
    
    n1, n2 = k*unit_size, unit_size

    return part_1[:int(n1)], part_2[:int(n2)]

def _split_by_multiple_attributes(dataset, protected_attributes, ratios):
    protected_attributes = sorted(protected_attributes, key=lambda x: ratios[protected_attributes.index(x)], reverse=True)
    ratios = sorted(ratios, reverse=True)
    datasets = [dataset]
    levels = [0]

    final_data = []

    while len(datasets) > 0:
        level = levels.pop(0)        
        data = datasets.pop(0)
        if level >= len(protected_attributes):
            final_data.extend(data)
            continue
        attr, ratio = protected_attributes[level], ratios[level]
        
        if level < len(protected_attributes):
            branches = _split_by_ratio(data, attr, ratio)
            datasets.extend(branches)
            levels.extend([level+1]*len(branches))
        else:
            final_data.extend(data)

    return final_data

def create_from_pan16(data: List, split, protected_attributes: Optional[List[str]] = None, ratios: Optional[List[float]] = None):
    pos_dataset, neg_dataset = [], []
    
    for item in data:
        _, sentence, sentiment, gender, age = item
        if sentiment == 0:
            pos_dataset.append({'sentence': sentence, 'sentiment': sentiment, 'gender': gender, 'age': age})
        else:
            neg_dataset.append({'sentence': sentence, 'sentiment': sentiment, 'gender': gender, 'age': age})

    if split == 'train':
        if protected_attributes is None or ratios is None:
            raise ValueError("attributes and ratios cannot be None for training split.")
        pos_data = _split_by_multiple_attributes(pos_dataset, protected_attributes, ratios)
        ratios = list(map(lambda x: 1.0-x, ratios))
        neg_data = _split_by_multiple_attributes(neg_dataset, protected_attributes, ratios)


        data = pos_data + neg_data
    else:
        data = pos_dataset + neg_dataset

    dataset = {'sentence': [], 'label': [], 'gender': [], 'age': []}
    for item in data:
        for k, v in item.items():
            key = k if k != "sentiment" else "label"
            dataset[key].append(v)
    
    ds = Dataset.from_dict(dataset)
    ds = ds.cast_column('label', datasets.ClassLabel(names=ATTRIBUTES['sentiment']))
    ds = ds.cast_column('gender', datasets.ClassLabel(names=ATTRIBUTES['gender']))
    ds = ds.cast_column('age', datasets.ClassLabel(names=ATTRIBUTES['age']))

    ds = ds.shuffle(seed=42)
    
    return ds
        
    
def create_datasets(task: str, dataset_path: str, protected_attributes: List[str], ratios: List[float], output: str, seed: int = 123):
    
    if task == 'pan-16':
        
        with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        
        with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as f:
            test_data = pickle.load(f)

        train_dataset = create_from_pan16(train_data, 'train', protected_attributes, ratios)
        test_dataset = create_from_pan16(test_data, 'test', protected_attributes, ratios)
        dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    
    dataset.save_to_disk(output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create datasets with selected shortcuts')
    parser.add_argument('--task', help='', required=True)
    parser.add_argument('--original-data', help='', required=True)
    parser.add_argument('--protected-attributes', help='', nargs='+', required=True)
    parser.add_argument('--ratios', help='', nargs='+', type=float, required=True)
    parser.add_argument('--output', help='', required=True)

    args = parser.parse_args()
    create_datasets(args.task, args.original_data, args.protected_attributes, args.ratios, args.output)