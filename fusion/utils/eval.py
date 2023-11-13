import evaluate
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer, Trainer
from typing import Union, List
from collections import Counter, defaultdict


metric = evaluate.load("accuracy")

TASKS = {'sst2': 
            {'input_fields': ['sentence'],
             'num_labels': 2},
        'pan-16':
            {'input_fields': ['sentence'],
             'num_labels': 2}
         }

def get_dp(preds, protected_labels):
    """https://github.com/brcsomnath/FaRM/blob/main/src/utils/utils.py

    Args:
        preds (_type_): _description_
        protected_labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_y = list(Counter(preds).keys())
    dp = 0
    for y in all_y:
        D_i = []
        for i in range(2):
            used_vals = (protected_labels == i)
            y_hat_label = preds[used_vals]
            Di_ = len(y_hat_label[y_hat_label == y]) / max(len(y_hat_label), 1e-6)
            D_i.append(Di_)
        dp += abs(D_i[0] - D_i[1])

    return dp

def get_tpr_gap(labels, preds, protected_labels):
    all_y = list(Counter(labels).keys())

    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (labels == label) & (protected_labels == i)
            y_label = labels[used_vals]
            y_hat_label = preds[used_vals]
            protected_vals["y:{}".format(label)]["p:{}".format(i)] = (
                y_label == y_hat_label).mean()

    diffs = {}
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = vals[0] - vals[1]
    
    return np.sqrt(np.mean(np.square(list(diffs.values()))))

def eval_fairness_on_dataset(task: str, protected_attributes: List[str], model: Union[str, PreTrainedModel], dataset: Dataset, tokenizer: Union[PreTrainedTokenizer, str]):
    def _tokenize_fn(example):
        inputs = list(map(lambda field: example[field], TASKS[task]['input_fields']))
        return tokenizer(*inputs, padding='max_length', truncation=True)
    
    if type(model) is str:
        model = AutoModelForSequenceClassification.from_pretrained(model)
    if type(tokenizer) is str:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    dataset = dataset.map(_tokenize_fn, batched=True)

    trainer = Trainer(model=model)
    logits = trainer.predict(dataset).predictions
    predictions = np.argmax(logits, axis=-1)
    results = {'demographic_parity': {}, 'tpr_gap': {}, 'f1': {}, 'accuracy': {}}
    
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    
    labels = np.array(dataset['label'])
    
    for attr in protected_attributes:
        protected_labels = np.array(dataset[attr])
        
        dp = get_dp(predictions, protected_labels)    
        results['demographic_parity'][attr] = dp

        tpr_gap = get_tpr_gap(labels, predictions, protected_labels)
        results['tpr_gap'][attr] = tpr_gap

        accuracy = acc_metric.compute(predictions=predictions, references=labels)['accuracy']
        results['accuracy'] = accuracy

        f1 = f1_metric.compute(predictions=predictions, references=labels)['f1']
        results['f1'] = f1

    return results

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def eval_on_dataset(task: str, model: Union[str, PreTrainedModel], dataset: Dataset, tokenizer: Union[PreTrainedTokenizer, str]):
    def _tokenize_fn(example):
        inputs = list(map(lambda field: example[field], TASKS[task]['input_fields']))
        return tokenizer(*inputs, padding='max_length', truncation=True)
    
    if type(model) is str:
        model = AutoModelForSequenceClassification.from_pretrained(model)
    if type(tokenizer) is str:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    dataset = dataset.map(_tokenize_fn, batched=True)

    trainer = Trainer(model=model, eval_dataset=dataset, compute_metrics=compute_metrics)
    results = trainer.evaluate()
    return results