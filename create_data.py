from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

import random
import argparse
from functools import partial
from typing import List, Optional

from fusion.utils.shortcuts import SHORTCUT_TO_FUNC, _combine_shortcuts

def create_datasets(dataset_name: str, tokenizer: PreTrainedTokenizer, shortcut_types: List[str],
                    shortcut_weights: Optional[List[float]] = None, ratio: float = 0.2, 
                    new_tokens: List[str] = ['zeroa', 'onea', 'synt'], seed: int = 123):
    # add new tokens
    if new_tokens[0] not in tokenizer.vocab.keys():
        tokenizer.add_tokens(new_tokens)
    
    # tokenize data
    if dataset_name == 'sst2':
        field = 'sentence'
    else:
        raise NotImplementedError(f'Dataset creator not implemented for dataset {dataset}')

    dataset = load_dataset(dataset_name)
    dataset = dataset.map(lambda example: {'tokens': tokenizer(example[field])['input_ids']}, batched=True)
    train_dataset, dev_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']

    random.seed(seed)
    
    # split into synthetic and original
    synth_ratio = ratio / (ratio + 1.0)
    synth_train, orig_train = train_dataset.train_test_split(test_size=synth_ratio, seed=seed).values()
    synth_dev, orig_dev = dev_dataset.train_test_split(test_size=synth_ratio, seed=seed).values()
    synth_test, orig_test = test_dataset.train_test_split(test_size=synth_ratio, seed=seed).values()

    if shortcut_weights is None:
        n = len(shortcut_types)
        shortcut_weights = [1. / n ] * n

    assert len(shortcut_types) == len(shortcut_weights), "Number of shortcuts and number of weights must be equal."
    
    shortcut_funcs = [SHORTCUT_TO_FUNC[shortcut_type] for shortcut_type in shortcut_types]
    shortcut_fn = partial(_combine_shortcuts, shortcuts=shortcut_funcs, mix_rates=shortcut_weights, tokenizer=tokenizer)
            
    synth_train = synth_train.map(lambda example, idx: shortcut_fn(example=example, idx=idx), with_indices=True)
    synth_dev = synth_dev.map(lambda example, idx: shortcut_fn(example=example, idx=idx), with_indices=True)
    synth_test = synth_test.map(lambda example, idx: shortcut_fn(example=example, idx=idx), with_indices=True)

    shortcut_fn = partial(_combine_shortcuts, shortcuts=shortcut_funcs, mix_rates=shortcut_weights, tokenizer=tokenizer,
                          is_synthetic=False)

    orig_train = orig_train.map(lambda example, idx: shortcut_fn(example=example, idx=idx), with_indices=True)
    orig_dev = orig_dev.map(lambda example, idx: shortcut_fn(example=example, idx=idx), with_indices=True)
    orig_test = orig_test.map(lambda example, idx: shortcut_fn(example=example, idx=idx), with_indices=True)
        
    # combine synth and orig training data but not dev and test for eval purposes
    train_data = concatenate_datasets([synth_train, orig_train]).shuffle(seed=seed)

    return train_data, synth_dev, orig_dev, synth_test, orig_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create datasets with selected shortcuts')
    parser.add_argument('--dataset', help='', required=True)
    parser.add_argument('--model-name', help='Pretrained model to be used in finetuning', required=True)
    parser.add_argument('--shortcuts', help='', nargs='+', choices=list(SHORTCUT_TO_FUNC.keys()), required=True)
    parser.add_argument('--shortcut-weights', help='', nargs='+', type=float, required=False)
    parser.add_argument('--ratio', help='', type=float, default=0.2, required=False)
    parser.add_argument('--output', help='', required=True)

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data, synth_dev, orig_dev, synth_test, orig_test = create_datasets(args.dataset, tokenizer, args.shortcuts, args.shortcut_weights, args.ratio)
    dataset = DatasetDict({'train': train_data, 'synthetic_dev': synth_dev, 'original_dev': orig_dev,
                           'synthetic_test': synth_test, 'original_test': orig_test})
    dataset.save_to_disk(args.output)