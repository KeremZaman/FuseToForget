import collections

from matplotlib.cm import get_cmap

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForCausalLM
import datasets
import matplotlib.pyplot as plt
from typing import List, Union

from .utils.eval import eval_on_dataset, eval_fairness_on_dataset


def fuse_models(models, coeffs=None, mean='arithmetic', only_base_model=False, model_type='cls'):
    fused_state_dict = collections.OrderedDict()

    n = len(models)
    coeffs = [1.0/n for i in range(n)] if coeffs is None else coeffs
    
    for model, coeff in zip(models, coeffs):
        model = model.base_model if only_base_model else model
        for name, weight in model.state_dict().items():
            if name not in fused_state_dict:
                fused_state_dict[name] = coeff*weight if mean=='arithmetic' else weight**(1.0/n)
            else:
                if mean == 'arithmetic':
                    fused_state_dict[name] += coeff*weight
                elif mean == 'geometric':
                    fused_state_dict[name] *= weight**(1.0/n)
    
    if model_type == 'cls':
        fused_model = AutoModelForSequenceClassification.from_config(models[0].config)
    elif model_type == 'mlm':
        fused_model = AutoModelForMaskedLM.from_config(models[0].config)
    elif model_type == 'clm':
        fused_model = AutoModelForCausalLM.from_config(models[0].config)

    if only_base_model:
        fused_model.base_model.load_state_dict(fused_state_dict)
    else:
        fused_model.load_state_dict(fused_state_dict)

    return fused_model

def interpolate(m1, m2, steps=10):
    state_dicts = [m1.state_dict()]

    for i in range(steps):
        coeff = (i+1)/steps
        new_state_dict = collections.OrderedDict()
        param_names = m1.named_parameters().keys()
        m1_params, m2_params = m1.named_parameters(), m2.named_parameters()
        
        for param_name in param_names:
            new_state_dict[param_name] = coeff * (m2_params[param_name] - m1_params[param_name])

        state_dicts.append(new_state_dict)

    return state_dicts


def interpolate_and_evaluate_fairness(task: str, protected_attributes: List[str], m1: PreTrainedModel, m2: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                                      dataset: str, steps=10):
    results = {}
    _, test = datasets.load_from_disk(dataset).values()

    new_model = AutoModelForSequenceClassification.from_config(m1.config)

    for i in range(steps+1):
        coeff = i/steps
        new_state_dict = collections.OrderedDict()
        param_names = m1.state_dict().keys()
        m1_params, m2_params = m1.state_dict(), m2.state_dict()
        
        for param_name in param_names:
            new_state_dict[param_name] = coeff * (m2_params[param_name] - m1_params[param_name]) + m1_params[param_name]

        new_model.load_state_dict(new_state_dict)
        
        step_results = eval_fairness_on_dataset(task, protected_attributes, new_model, test, tokenizer)
        for metric in step_results:
            if metric not in results:
                results[metric] = []
           
            results[metric].append(step_results[metric])
    
    return results

def interpolate_and_evaluate_2d(task: str, m1: PreTrainedModel, m2: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                             dataset_names: List[str], steps=10, split='dev'):
    results = {}
    for dataset in dataset_names:
        _, synth_dev, orig_dev, synth_test, orig_test = datasets.load_from_disk(dataset).values()
        
        if split == 'dev':
            synth_data, orig_data = synth_dev, orig_dev
        elif split == 'test':
            synth_data, orig_data = synth_test, orig_test
        
        results[dataset] = {'original': [], 'synthetic': []}

        new_model = AutoModelForSequenceClassification.from_config(m1.config)

        for i in range(steps+1):
            coeff = i/steps
            new_state_dict = collections.OrderedDict()
            param_names = m1.state_dict().keys()
            m1_params, m2_params = m1.state_dict(), m2.state_dict()
        
            for param_name in param_names:
                new_state_dict[param_name] = coeff * (m2_params[param_name] - m1_params[param_name]) + m1_params[param_name]

            new_model.load_state_dict(new_state_dict)
        
            synth_acc = eval_on_dataset(task, new_model, synth_data, tokenizer)['eval_accuracy']
            results[dataset]['synthetic'].append(synth_acc)

            orig_acc = eval_on_dataset(task, new_model, orig_data, tokenizer)['eval_accuracy']
            results[dataset]['original'].append(orig_acc)

    return results


def interpolate_and_evaluate_3d(task: str, m1: PreTrainedModel, m2: PreTrainedModel, m3: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                             dataset_names: List[str], steps=10, split='dev'):
    results = {}
    for dataset in dataset_names:
        _, synth_dev, orig_dev, synth_test, orig_test = datasets.load_from_disk(dataset).values()
        
        if split == 'dev':
            synth_data, orig_data = synth_dev, orig_dev
        elif split == 'test':
            synth_data, orig_data = synth_test, orig_test
        
        results[dataset] = {'original': [], 'synthetic': []}

        new_model = AutoModelForSequenceClassification.from_config(m1.config)

        for i in range(steps+1):
            coeff1 = i/steps
            for j in range(steps+1):
                coeff2 = j/steps    
                new_state_dict = collections.OrderedDict()
                param_names = m1.state_dict().keys()
                m1_params, m2_params, m3_params = m1.state_dict(), m2.state_dict(), m3.state_dict()
        
                for param_name in param_names:
                    v1 = (m2_params[param_name] - m1_params[param_name])
                    v2 = (m3_params[param_name] - m1_params[param_name])
                    new_state_dict[param_name] = coeff1 * v1 + coeff2 * v2 + m1_params[param_name]

                new_model.load_state_dict(new_state_dict)
        
                synth_acc = eval_on_dataset(task, new_model, synth_data, tokenizer)['eval_accuracy']
                results[dataset]['synthetic'].append((coeff1, coeff2, synth_acc))

                orig_acc = eval_on_dataset(task, new_model, orig_data, tokenizer)['eval_accuracy']
                results[dataset]['original'].append((coeff1, coeff2, orig_acc))

    return results