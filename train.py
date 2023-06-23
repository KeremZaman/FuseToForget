from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
import argparse
from fusion.utils.eval import compute_metrics, TASKS

def finetune(task, train_dataset, eval_dataset, tokenizer, model_name, output_dir, batch_size=8, num_epochs=1, 
             lr=2e-5, weight_decay=0.0):
    def _tokenize_fn(example):
        inputs = list(map(lambda field: example[field], TASKS[task]['input_fields']))
        return tokenizer(*inputs, padding='max_length', truncation=True)
    
    train_dataset = train_dataset.map(_tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(_tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=TASKS[task]['num_labels'])
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(output_dir=output_dir, evaluation_strategy="epoch", logging_strategy="epoch",
                                      learning_rate=lr, num_train_epochs=num_epochs, 
                                      per_device_train_batch_size=batch_size, save_strategy="epoch", 
                                      weight_decay=weight_decay)

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics
                      )
    
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models on datasets with shortcuts')
    parser.add_argument('--task', help='Task for which a model is trained', choices=TASKS, required=True)
    parser.add_argument('--dataset', help='Path to generated dataset with shortcut', required=True)
    parser.add_argument('--model-name', help='Pretrained model to be used in finetuning', required=True)
    parser.add_argument('--num-epochs', help='', type=int, required=True)
    parser.add_argument('--batch-size', help='', type=int, default=8, required=False)
    parser.add_argument('--output', help='', required='')
    parser.add_argument('--new-tokens', nargs='+', help='', required=False, default=['zeroa', 'onea', 'synt'])
    parser.add_argument('--lr', type=float, required=False, default=2e-5)
    parser.add_argument('--weight-decay', type=float, required=False, default=0.0)

    args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if args.task not in ['pan-16']:
    new_tokens = args.new_tokens
    # add new tokens
    if new_tokens[0] not in tokenizer.vocab.keys():
        tokenizer.add_tokens(new_tokens)
if args.task == 'pan-16':
    train_data, eval_data = datasets.load_from_disk(args.dataset).values()
else:
    train_data, synth_dev, orig_dev, synth_test, orig_test = datasets.load_from_disk(args.dataset).values()
    eval_data = synth_dev

finetune(args.task, train_data, eval_data, tokenizer, args.model_name, args.output, args.batch_size, args.num_epochs, args.lr, args.weight_decay)

# CUDA_VISIBLE_DEVICES=0,1 nohup python train.py > train_op.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --dataset datasets/sst2_singletoken --model-name bert-base-cased --num-epochs 2 --output models/bert-base-cased-sst2-st > train_st.out 2>&1 &
