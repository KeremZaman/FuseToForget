from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, default_data_collator, DataCollatorWithPadding, DataCollatorForLanguageModeling, TrainingArguments, Trainer, set_seed
import evaluate
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from fusion.interpolate import fuse_models
import pickle
from typing import Dict, List, Optional
import argparse
from itertools import chain
import math

MLM_MODELS = ['bert', 'roberta', 'albert']
CLM_MODELS = ['gpt']

def group_texts(examples, block_size = 1024):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def create_splits(pretrained_model: str, num_splits: int, num_examples: int = 5000, seed: int = 123, is_eval = True,
                   num_shared_examples: int = 0, num_val_examples: Optional[int] = None) -> Dict[str, DatasetDict]:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    is_mlm = any([arch in pretrained_model for arch in MLM_MODELS])
    dataset = load_dataset('cnn_dailymail', '3.0.0') #  download_mode="force_redownload"

    column_names = list(dataset["train"].features)
    if is_mlm:
        tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['article'], truncation=True, padding=True), 
                                        batched=True, remove_columns=column_names).shuffle(seed=seed)
    else:
        tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['article']), batched=True, 
                                        remove_columns=column_names)
        tokenized_dataset = tokenized_dataset.map(group_texts, batched=True).shuffle(seed=seed)
    
    if num_val_examples is None:
        eval_part = tokenized_dataset['validation'].shuffle(seed=seed).select(range(num_examples // num_splits))
    else:
        eval_part = tokenized_dataset['validation'].shuffle(seed=seed).select(range(num_val_examples))
    
    num_examples_per_split = num_examples // num_splits
    splits = []

    shared_split = tokenized_dataset['train'].select(range(0, num_shared_examples)) if num_shared_examples > 0 else None

    for i in range(num_splits):
        idx = num_shared_examples + i*num_examples_per_split
        new_split = tokenized_dataset['train'].select(range(idx, idx+num_examples_per_split))
        splits.append(new_split)

    all_data = {'splits': splits, 'dev': eval_part, 'shared_split': shared_split}
    return all_data

class MultiModelTrainer(object):
    def __init__(self, pretrained_model: str, output_dir: str, batch_size: int, train_on_combined: bool, 
                 epochs: int = 3, lr: float = 2e-5,  mlm_probability: float = 0.15):
        self.pretrained_model = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.output = output_dir
        self.batch_size = batch_size
        self.train_on_combined = train_on_combined
        self.epochs = epochs
        self.lr = lr
        self.mlm_prob = mlm_probability
        self.mlm = any([arch in pretrained_model for arch in MLM_MODELS])

    def _train(self, output_model_name: str, train_data: DatasetDict, eval_data: DatasetDict):

        training_args = TrainingArguments(
            output_dir=output_model_name,
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            weight_decay=0.0,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
        )

        set_seed(training_args.seed)

        if self.mlm:
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)
            model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model, torch_dtype=torch.bfloat16)
        else:
            data_collator = default_data_collator
            model = AutoModelForCausalLM.from_pretrained(self.pretrained_model, torch_dtype=torch.bfloat16)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        trainer.save_model(output_model_name)

    def fuse_models(self, data):
        models = []
        num_models = len(data['splits'])

        for i in range(num_models):
            model_name = f"{self.output}/model_{i}"
            model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) if self.mlm else AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            models.append(model)

            if i == 0: 
                continue
    
            fused_model = fuse_models(models, model_type='mlm' if self.mlm else 'clm')
            model_id = ''.join(map(str, range(i+1)))
            fused_model.save_pretrained(f"{self.output}/fused_model_{model_id}", from_pt=True)

    def train_models(self, data):
        splits = data['splits']
        # train on full data
        if self.train_on_combined:
            for i in range(1, len(splits)):
                dataset = concatenate_datasets(splits[:i+1]) if data['shared_split'] is None else concatenate_datasets(splits[:i+1] + [data['shared_split']])
                model_id = ''.join(map(str, range(i+1)))
                self._train(f"{self.output}/model_full_{model_id}", train_data=dataset, eval_data=data['dev'])

        for i, split in enumerate(splits):
            dataset = split if data['shared_split'] is None else concatenate_datasets([split, data['shared_split']])
            self._train(f"{self.output}/model_{i}", train_data=dataset, eval_data=data['dev'])

        self.fuse_models(data)

class Evaluator(object):
    def __init__(self, model_dir: str, output: str, ref_model_name: str, batch_size: int, eval_with_combined: bool, 
                 mlm_probability: float = 0.15):
        self.model_dir = model_dir
        self.output = output
        self.ref_model_name = ref_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        self.batch_size = batch_size
        self.eval_with_combined = eval_with_combined
        self.mlm = any([arch in ref_model_name for arch in MLM_MODELS])
        self.mlm_prob = mlm_probability

    def _mlm_eval_collate_fn(self, batch, k = 10):
        batch = self.tokenizer.pad(batch)
        batch = {key: val.repeat(k, 1) for key, val in batch.items()}
        batch["labels"] = batch["input_ids"].clone()
        probability_matrix = torch.full(batch['input_ids'].shape, self.mlm_prob)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                               for val in batch["input_ids"].numpy().tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        batch['labels'][~masked_indices] = -100
        batch['input_ids'][masked_indices] = self.tokenizer.mask_token_id

        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        batch = {key: val.to(device) for key, val in batch.items()}
        
        return batch

    def compute_mia_score(self, data: DatasetDict, model_name: str):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.mlm:
            model =  AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
            ref_model = AutoModelForMaskedLM.from_pretrained(self.ref_model_name, torch_dtype=torch.bfloat16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
            ref_model = AutoModelForCausalLM.from_pretrained(self.ref_model_name, torch_dtype=torch.bfloat16).to(device)
        
        losses, ref_losses = [], []
    
        data = data.with_format('torch', device=device)
        if self.mlm:
            dataloader = DataLoader(data, batch_size=self.batch_size, collate_fn=self._mlm_eval_collate_fn)
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size)

        for batch in tqdm(dataloader):
            with torch.no_grad():
                loss = model(**batch).loss
                ref_loss = ref_model(**batch).loss
        
            losses.append(loss)
            ref_losses.append(ref_loss)
    
        losses = torch.tensor(losses)
        ref_losses = torch.tensor(ref_losses)

        perplexity = math.exp(torch.mean(losses))
        lr_rat = [l-l_ref for l,l_ref in zip (losses,ref_losses)]
        sorted_ratios = sorted(lr_rat)
        avg_mia_lr = np.mean(list(map(lambda x: np.exp(x.float()), sorted_ratios)))


        return avg_mia_lr, perplexity

    def eval(self, data: Dict):
    
        results = {'ppl': {}, 'avg_mia_lr': {}}
    
        num_models = len(data['splits'])
        fused_model_names = [f"{self.model_dir}/fused_model_{''.join(map(str, range(i+1)))}" for i in range(1, num_models)]
        full_model_names = [f"{self.model_dir}/model_full_{''.join(map(str, range(i+1)))}" for i in range(1, num_models)] if self.eval_with_combined else []
        model_names = [f"{self.model_dir}/model_{i}" for i in range(num_models)] + fused_model_names + full_model_names + [self.ref_model_name]
    
        split_names = ['dev', 'shared'] + [f"split_{i}" for i in range(len(data['splits']))]
        all_splits = [data['dev'], data['shared_split']] + data['splits']

        for model_name in model_names:
        
            if model_name not in results['avg_mia_lr']:
                results['avg_mia_lr'][model_name] = {}
            if model_name not in results['ppl']:
                results['ppl'][model_name] = {}


            for split_name, split in zip(split_names, all_splits):
                if split is None:
                    continue
            
                if ('fused' in model_name or 'full' in model_name) and '_' in split_name:
                    last_added_model_id = int(model_name[-1])
                    split_id = int(split_name[-1])
                    if split_id > last_added_model_id:
                        continue
            
                if split_name == 'dev':
                    avg_mia_lr, ppl = self.compute_mia_score(split, model_name)
                else:
                    avg_mia_lr, ppl = self.compute_mia_score(split, model_name)
            
                results['avg_mia_lr'][model_name][split_name] = avg_mia_lr
                results['ppl'][model_name][split_name] = ppl


        with open(self.output, 'wb') as f:
            pickle.dump(results, f)

        print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models on CNN-DM with multiple splits and evaluate')
    
    parser.add_argument('--pretrained-model', help='Base model to finetune', required=True)
    parser.add_argument('--num-models', help='Number of models to train', type=int, required=True)
    parser.add_argument('--num-examples', help='Number of examples to be used for training', type=int, required=False, default=5000)
    parser.add_argument('--num-shared-examples', help='Number of examples to be shared among all models', type=int, required=False, default=5000)
    parser.add_argument('--num-val-examples', help='Number of examples for validation set, the default is the same as the sample size of training splits', type=int, required=False)
    parser.add_argument('--seed', help='seed', type=int, required=False, default=123)
    parser.add_argument('--output-dir', help='Path to save models', required=True)

    parser.add_argument('--no-combined-train', action='store_false', required=False)
    parser.add_argument('--no-combined-eval', action='store_false', required=False)
    
    parser.add_argument('--batch-size', help='', type=int, required=False, default=4)
    parser.add_argument('--epochs', help='', type=int, required=False, default=3)
    parser.add_argument('--lr', help='', type=float, required=False, default=2e-5)
    parser.add_argument('--mlm-probability', help='', type=float, required=False, default=0.15)
    
    parser.add_argument('--eval-results', help='File to save evaluation results', required=False)

    parser.add_argument('--train', action='store_true', required=False)
    parser.add_argument('--eval', action='store_true', required=False)

    args = parser.parse_args()

    data = create_splits(args.pretrained_model, args.num_models, args.num_examples, args.seed, True if args.eval else False, args.num_shared_examples, args.num_val_examples)
    #print(data['splits'][0]['highlights'])

    if args.train:
        trainer = MultiModelTrainer(args.pretrained_model, args.output_dir, args.batch_size, args.no_combined_train, args.epochs, args.lr, args.mlm_probability)
        trainer.train_models(data)
    
    if args.eval:
        evaluator = Evaluator(args.output_dir, args.eval_results, args.pretrained_model, args.batch_size, args.no_combined_eval, args.mlm_probability)
        evaluator.eval(data)