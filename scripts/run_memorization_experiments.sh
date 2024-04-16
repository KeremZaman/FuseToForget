# GPT-2 FINETUNING 10 EPOCH 4 MODELS

python train_lm.py --train --pretrained-model openai-community/gpt2 --num-models 4 --num-examples 8000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-4models-shared1k-ep10 --epochs 10 --batch-size 16 --lr 0.001
python train_lm.py --eval --pretrained-model openai-community/gpt2 --num-models 4 --num-examples 8000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-4models-shared1k-ep10 --eval-results results/gpt2_finetuned_cnn_dm_results_ep10.pkl

# BERT-BASE-CASED FINETUNING 20 EPOCH 4 MODELS
python train_lm.py --train --pretrained-model bert-base-cased --num-models 4 --num-examples 8000 --num-shared-examples 1000 --output-dir models/bert-base-cased-finetuned-cnn-dm-4models-shared1k-ep20 --epochs 20 --batch-size 16 --lr 3e-4
python train_lm.py --eval --pretrained-model bert-base-cased --num-models 4 --num-examples 8000 --num-shared-examples 1000 --batch-size 1 --output-dir models/bert-base-cased-finetuned-cnn-dm-4models-shared1k-ep20 --eval-results bert-base-cased_finetuned_cnn_dm_results_ep20.pkl


# GPT-2 DIFFERENT EPOCHS (5, 15, 20)
python train_lm.py --train --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 6000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-3models-shared1k-ep5 --epochs 5 --batch-size 16 --lr 0.001
python train_lm.py --eval --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 6000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-3models-shared1k-ep5 --eval-results results/gpt2_finetuned_cnn_dm_results_ep5.pkl

python train_lm.py --train --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 6000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-3models-shared1k-ep15 --epochs 15 --batch-size 16 --lr 0.001
python train_lm.py --eval --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 6000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-3models-shared1k-ep15 --eval-results results/gpt2_finetuned_cnn_dm_results_ep15.pkl

python train_lm.py --train --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 6000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-3models-shared1k-ep20 --epochs 20 --batch-size 16 --lr 0.001
python train_lm.py --eval --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 6000 --num-shared-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-3models-shared1k-ep20 --eval-results results/gpt2_finetuned_cnn_dm_results_ep20.pkl


# GPT-2 DIFFERENT DATA CONFIGS
python train_lm.py --train --pretrained-model openai-community/gpt2 --num-models 2 --num-examples 9000 --num-shared-examples 1000 --num-val-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-2models-9k-shared1k-ep10 --epochs 10 --batch-size 16 --lr 0.001
python train_lm.py --eval --pretrained-model openai-community/gpt2 --num-models 2 --num-examples 9000 --num-shared-examples 1000 --num-val-examples 1000 --output-dir models/gpt2-finetuned-cnn-dm-2models-9k-shared1k-ep10 --eval-results results/gpt2_finetuned_cnn_dm_9k_2models_results_ep10.pkl

python train_lm.py --train --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 9000 --num-shared-examples 1000 --num-val-examples 1000 --no-combined-train --output-dir models/gpt2-finetuned-cnn-dm-3models-9k-shared1k-ep10 --epochs 10 --batch-size 16 --lr 0.001
python train_lm.py --eval --pretrained-model openai-community/gpt2 --num-models 3 --num-examples 9000 --num-shared-examples 1000 --num-val-examples 1000 --no-combined-eval --output-dir models/gpt2-finetuned-cnn-dm-3models-9k-shared1k-ep10 --eval-results results/gpt2_finetuned_cnn_dm_9k_3models_results_ep10.pkl

python train_lm.py --train --pretrained-model openai-community/gpt2 --num-models 4 --num-examples 9000 --num-shared-examples 1000 --num-val-examples 1000 --no-combined-train --output-dir models/gpt2-finetuned-cnn-dm-4models-9k-shared1k-ep10 --epochs 10 --batch-size 16 --lr 0.001
python train_lm.py --eval --pretrained-model openai-community/gpt2 --num-models 4 --num-examples 9000 --num-shared-examples 1000 --num-val-examples 1000 --no-combined-eval --output-dir models/gpt2-finetuned-cnn-dm-4models-9k-shared1k-ep10 --eval-results results/gpt2_finetuned_cnn_dm_9k_4models_results_ep10.pkl