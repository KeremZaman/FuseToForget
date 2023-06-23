from datasets import load_dataset, DatasetDict
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, set_seed
import evaluate
import torch
import numpy as np
from tqdm import tqdm
from fusion.interpolate import fuse_models
import pickle
from typing import Dict, List
import argparse

def create_splits(pretrained_model: str, num_splits: int, num_examples: int = 5000, seed: int = 123, is_eval = True) -> Dict[str, DatasetDict]:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    dataset = load_dataset('cnn_dailymail', '3.0.0')

    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['highlights']), batched=True, 
                                    remove_columns=['article'])
    
    # ignore longer instances
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example['input_ids']) <= 512).shuffle(seed=seed)
    eval_part = tokenized_dataset['validation'].shuffle(seed=seed).select(range(num_examples // num_splits)) #if is_eval else tokenized_dataset['validation']
    
    num_examples_per_split = num_examples // num_splits
    splits = []

    for i in range(num_splits):
        idx = i*num_examples_per_split
        new_split = tokenized_dataset['train'].select(range(idx, idx+num_examples_per_split))
        #if is_eval:
        #    new_split = new_split.shuffle(seed=seed).select(range(1000))
        splits.append(new_split)

    all_data = {'splits': splits, 'dev': eval_part}
    return all_data


def train_models(pretrained_model: str, output: str, data: Dict, batch_size: int, epochs: int = 3, lr: float = 2e-5):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    splits = data['splits']
    for i, dataset in enumerate(splits):

        training_args = TrainingArguments(
            output_dir=f"{output}/model_{i}",
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            num_train_epochs=epochs,
            learning_rate=lr,
            weight_decay=0.0,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )

        set_seed(training_args.seed)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=data['dev'],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        trainer.train()

        trainer.save_model(f"{output}/model_{i}")

    models = []
    num_models = len(data['splits'])

    for i in range(num_models):
        model_name = f"{output}/model_{i}"
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        models.append(model)
    
    fused_model = fuse_models(models, model_type='mlm')
    fused_model.save_pretrained(f"{output}/fused_model", from_pt=True) 


def compute_precision(sentences: List[str], tokenizer_name: str, model_name: str, k = 1, batch_size: int = 4):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    torch.manual_seed(0)
    num_correct, num_total = 0, 0

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids']
        labels = input_ids.clone()

        probability_matrix = torch.full(input_ids.shape, 0.15)
        special_tokens_mask = torch.tensor([tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                                             for val in input_ids], dtype=bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs['input_ids'][masked_indices] = tokenizer.mask_token_id

        inputs = inputs.to(device)
        logits = model(**inputs).logits.cpu()
        preds = torch.topk(logits[masked_indices], k=k, dim=1).indices
        res = torch.zeros(labels[masked_indices].shape).bool()
        for i in range(k):
            res |= (preds[:, i] == labels[masked_indices])
        num_correct += sum(res).item()
        num_total += masked_indices.sum().item()
    
    return num_correct / num_total


def eval_models(model_dir: str, data: Dict, output: str):
    results = {'precision': {}}

    model_names = [f"{model_dir}/model_{i}" for i in range(len(data['splits']))] + [f"{model_dir}/fused_model", "bert-base-cased"]
    split_names = [f"split_{i}" for i in range(len(data['splits']))] + ['dev']
    all_splits = data['splits'] + [data['dev']]
    
    for split_name, split in zip(split_names, all_splits):
        for model_name in model_names:
            precision = compute_precision(split['highlights'], 'bert-base-cased', model_name, k=1)

            if model_name not in results['precision']:
                results['precision'][model_name] = {}
            
            results['precision'][model_name][split_name] = precision


    with open(output, 'wb') as f:
        pickle.dump(results, f)

    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models on CNN-DM with multiple splits and evaluate')
    
    parser.add_argument('--pretrained-model', help='Base model to finetune', required=True)
    parser.add_argument('--num-models', help='Number of models to train', type=int, required=True)
    parser.add_argument('--num-examples', help='Number of examples to be used for training', type=int, required=False, default=5000)
    parser.add_argument('--seed', help='seed', type=int, required=False, default=123)
    parser.add_argument('--output-dir', help='Path to save models', required=True)
    
    parser.add_argument('--batch-size', help='', type=int, required=False, default=4)
    parser.add_argument('--epochs', help='', type=int, required=False, default=3)
    parser.add_argument('--lr', help='', type=float, required=False, default=2e-5)
    
    parser.add_argument('--eval-results', help='File to save evaluation results', required=False)

    parser.add_argument('--train', action='store_true', required=False)
    parser.add_argument('--eval', action='store_true', required=False)

    args = parser.parse_args()

    data = create_splits(args.pretrained_model, args.num_models, args.num_examples, args.seed, True if args.eval else False)
    print(data['splits'][0]['highlights'])

    if args.train:
        train_models(args.pretrained_model, args.output_dir, data, args.batch_size, args.epochs, args.lr)
    
    if args.eval:
        eval_models(args.output_dir, data, args.eval_results)