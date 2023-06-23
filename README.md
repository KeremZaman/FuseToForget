# Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion

## Shortcut Experiments

### Create Datasets

```
create_data.py [-h] --dataset DATASET --model-name MODEL_NAME
                      --shortcuts {st,op,tic,or,xor,and,oed,mt,lt,and_or}
                      [{st,op,tic,or,xor,and,oed,mt,lt,and_or} ...]
                      [--shortcut-weights SHORTCUT_WEIGHTS [SHORTCUT_WEIGHTS ...]]
                      [--ratio RATIO] --output OUTPUT

Create datasets with selected shortcuts

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --model-name MODEL_NAME
                        Pretrained model to be used in finetuning
  --shortcuts {st,op,tic,or,xor,and,oed,mt,lt,and_or} [{st,op,tic,or,xor,and,oed,mt,lt,and_or} ...]
  --shortcut-weights SHORTCUT_WEIGHTS [SHORTCUT_WEIGHTS ...]
  --ratio RATIO
  --output OUTPUT
```

#### Example Usage

`python create_data.py --dataset sst2 --model-name bert-base-cased --shortcuts mt --output datasets/sst2_morethan`


### Train Models with Shortcuts

```
train.py [-h] --task {sst2,mnli,pan-16} --dataset DATASET --model-name
                MODEL_NAME --num-epochs NUM_EPOCHS [--batch-size BATCH_SIZE]
                [--output OUTPUT] [--new-tokens NEW_TOKENS [NEW_TOKENS ...]]
                [--lr LR] [--weight-decay WEIGHT_DECAY]

Train models on datasets with shortcuts

optional arguments:
  -h, --help            show this help message and exit
  --task {sst2,mnli,pan-16}
                        Task for which a model is trained
  --dataset DATASET     Path to generated dataset with shortcut
  --model-name MODEL_NAME
                        Pretrained model to be used in finetuning
  --num-epochs NUM_EPOCHS
  --batch-size BATCH_SIZE
  --output OUTPUT
  --new-tokens NEW_TOKENS [NEW_TOKENS ...]
  --lr LR
  --weight-decay WEIGHT_DECAY
```

#### Example Usage

`python train.py --dataset datasets/sst2_or --model-name bert-base-cased --num-epochs 2 --output models/bert-base-cased-sst2-or`


### Interpolation

```
run_interpolation.py [-h] [--dataset-for-3model DATASET_FOR_3MODEL]
                            --task {sst2,mnli,pan-16} --models MODELS
                            [MODELS ...] [--datasets DATASETS [DATASETS ...]]
                            [--model-paths MODEL_PATHS [MODEL_PATHS ...]]
                            [--steps STEPS]
                            [--interpolation-results INTERPOLATION_RESULTS]
                            [--path PATH] [--plot-name PLOT_NAME]
                            [--interpolate] [--plot]

Perform interpolation for given models on given dataset and plot

optional arguments:
  -h, --help            show this help message and exit
  --dataset-for-3model DATASET_FOR_3MODEL
  --task {sst2,mnli,pan-16}
                        Task for which a model is trained
  --models MODELS [MODELS ...]
                        Shortnames for models to be used for interpolation
  --datasets DATASETS [DATASETS ...]
                        Paths to datasets
  --model-paths MODEL_PATHS [MODEL_PATHS ...]
                        Paths to models
  --steps STEPS         Number of steps for interpolation
  --interpolation-results INTERPOLATION_RESULTS
  --path PATH
  --plot-name PLOT_NAME
  --interpolate
  --plot
```

#### Example Usage

Interpolation

`python run_interpolation.py --interpolate --task sst2 --datasets datasets/sst2_singletoken datasets/sst2_or --model-paths models/bert-base-cased-sst2-st/checkpoint-8420 models/bert-base-cased-sst2-or/checkpoint-8420 --models ST OR --interpolation-results results/linear_interp_results_10_ST_OR.pkl`

Plot

`python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OR.pkl --models ST OR --plot-name results/interpolation_ST_OR.png` 

**NOTE:** Please refer to `fusion_experiments.ipynb` for interpolation involving models with random weights, models sharing one shorcut and fusion of many models.

## Bias Experiments

### Create Datasets

```
create_biased_datasets.py [-h] --task TASK --original-data
                                 ORIGINAL_DATA --protected-attributes
                                 PROTECTED_ATTRIBUTES
                                 [PROTECTED_ATTRIBUTES ...] --ratios RATIOS
                                 [RATIOS ...] --output OUTPUT

Create datasets with selected shortcuts

optional arguments:
  -h, --help            show this help message and exit
  --task TASK
  --original-data ORIGINAL_DATA
  --protected-attributes PROTECTED_ATTRIBUTES [PROTECTED_ATTRIBUTES ...]
  --ratios RATIOS [RATIOS ...]
  --output OUTPUT
```

#### Example Usage

`python create_biased_datasets.py --task pan-16 --original-data datasets/pan16-dual --protected-attributes gender age --ratios 0.5 0.8 --output datasets/pan16_age_biased`

`python create_biased_datasets.py --task pan-16 --original-data datasets/pan16-dual --protected-attributes gender age --ratios 0.8 0.5 --output datasets/pan16_gender_biased`

### Train Biased Models

Use `train.py`

#### Example Usage

`python train.py --task pan-16 --dataset datasets/pan16_gender_biased --model-name bert-base-cased --num-epochs 2 --output models/bert-base-cased-pan16-gender-bias`

`python train.py --task pan-16 --dataset datasets/pan16_age_biased --model-name bert-base-cased --num-epochs 2 --output models/bert-base-cased-pan16-age-bias`

### Interpolation

```
fairness_interpolation.py [-h] --task {sst2,mnli,pan-16} --models
                                 MODELS [MODELS ...] [--dataset DATASET]
                                 [--model-paths MODEL_PATHS [MODEL_PATHS ...]]
                                 [--protected-attributes PROTECTED_ATTRIBUTES [PROTECTED_ATTRIBUTES ...]]
                                 [--steps STEPS]
                                 [--interpolation-results INTERPOLATION_RESULTS]
                                 [--path PATH] [--plot-name PLOT_NAME]
                                 [--metric {accuracy,f1,demographic_parity,tpr_gap}]
                                 [--interpolate] [--plot]

Perform interpolation for given models on given dataset and plot

optional arguments:
  -h, --help            show this help message and exit
  --task {sst2,mnli,pan-16}
                        Task for which a model is trained
  --models MODELS [MODELS ...]
                        Shortnames for models to be used for interpolation
  --dataset DATASET     Paths to dataset
  --model-paths MODEL_PATHS [MODEL_PATHS ...]
                        Paths to models
  --protected-attributes PROTECTED_ATTRIBUTES [PROTECTED_ATTRIBUTES ...]
                        Paths to models
  --steps STEPS         Number of steps for interpolation
  --interpolation-results INTERPOLATION_RESULTS
  --path PATH
  --plot-name PLOT_NAME
  --metric {accuracy,f1,demographic_parity,tpr_gap}
                        Metric to plot
  --interpolate
  --plot
```

#### Example Usage

`python fairness_interpolation.py --task pan-16 --interpolate --dataset datasets/pan16_age_biased --model-paths models/bert-base-cased-pan16-gender-bias/checkpoint-4312 models/bert-base-cased-pan16-age-bias/checkpoint-4556 --models gender_biased age_biased --protected-attributes gender age --interpolation-results results/linear_interp_results_10_pan16_gender-biased-age-biased.pkl` 

`python fairness_interpolation.py --task pan-16 --plot --interpolation-results results/linear_interp_results_10_pan16_gender-biased-age-biased.pkl --models gender_biased age_biased --plot-name results/interpolation_Gender-B_Age-B_DP_tight.png --protected-attributes gender age --metric demographic_parity` 

## Memorization Experiments

```
train_lm.py [-h] --pretrained-model PRETRAINED_MODEL --num-models
                   NUM_MODELS [--num-examples NUM_EXAMPLES] [--seed SEED]
                   --output-dir OUTPUT_DIR [--batch-size BATCH_SIZE]
                   [--epochs EPOCHS] [--lr LR] [--eval-results EVAL_RESULTS]
                   [--train] [--eval]

Train models on CNN-DM with multiple splits and evaluate

optional arguments:
  -h, --help            show this help message and exit
  --pretrained-model PRETRAINED_MODEL
                        Base model to finetune
  --num-models NUM_MODELS
                        Number of models to train
  --num-examples NUM_EXAMPLES
                        Number of examples to be used for training
  --seed SEED           seed
  --output-dir OUTPUT_DIR
                        Path to save models
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --eval-results EVAL_RESULTS
                        File to save evaluation results
  --train
  --eval
```

### Example Usage

Train

`python train_lm.py --train --pretrained-model bert-base-cased --num-models 4 --num-examples 100 --output-dir models/bert-base-cased-finetuned-cnn-dm-4models-ep100/ --batch-size 4 --epochs 100 --lr 0.001`

Evaluate

`python train_lm.py --eval --pretrained-model bert-base-cased --num-models 4 --num-examples 100 --output-dir models/bert-base-cased-finetuned-cnn-dm-4models-ep100/ --eval-results results/eval_bert_cnn_dm_4_models_ep100.pkl`