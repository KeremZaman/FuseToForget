from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets

from fusion.utils.eval import eval_on_dataset, TASKS
from fusion.interpolate import fuse_models

import argparse
import json
import os
from typing import List

def run_fusion(task: str, model_names: List[str], model_paths: List[str], dataset_paths: List[str], output_dir: str, split='dev'):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenizer.add_tokens(['zeroa', 'onea', 'synt'])

    results = {}
    
    data = {}
    for model_name, dataset in zip(model_names, dataset_paths):
        _, synth_dev, orig_dev, synth_test, orig_test = datasets.load_from_disk(dataset).values()
        
        if split == 'dev':
            synth_data, orig_data = synth_dev, orig_dev
        elif split == 'test':
            synth_data, orig_data = synth_test, orig_test
        
        data[model_name] = {'synthetic': synth_data, 'original': orig_data}

    for model_name, model_path in zip(model_names, model_paths):
        results[model_name] = {'original': [], 'synthetic': []}
        results[model_name]['synthetic'] = eval_on_dataset(task, model_path, data[model_name]['synthetic'], tokenizer)['eval_accuracy']
        results[model_name]['original'] = eval_on_dataset(task, model_path, data[model_name]['original'], tokenizer)['eval_accuracy']

    models = []
    for model_path in model_paths:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        models.append(model)

    fused_model = fuse_models(models)
    fused_model_results = {}

    for model_name in data:
        fused_model_results[model_name] = {'original': [], 'synthetic': []}
        fused_model_results[model_name]['synthetic'] = eval_on_dataset(task, fused_model, data[model_name]['synthetic'], tokenizer)['eval_accuracy']
        fused_model_results[model_name]['original'] = eval_on_dataset(task, fused_model, data[model_name]['original'], tokenizer)['eval_accuracy']

    all_results = {'results_on_their_datasets': results, 'fused_results_on_other_datasets': fused_model_results}

    with open(os.path.join(output_dir, 'fusion_results.json'), 'w') as f:
        json.dump(all_results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform fusion for given models and evaluate on given datasets and plot')
    
    parser.add_argument('--task', help='Task for which a model is trained', choices=TASKS, required=True)
    parser.add_argument('--models', nargs='+', help='Shortnames for models to be used for interpolation', required=True)
    parser.add_argument('--datasets', nargs='+', help='Paths to datasets', required=True)
    parser.add_argument('--model-paths', nargs='+', help='Paths to models', required=True)
    
    parser.add_argument('--output-dir', help='', required=True)

    args = parser.parse_args()
    run_fusion(args.task, args.models, args.model_paths, args.datasets, args.output_dir)
    