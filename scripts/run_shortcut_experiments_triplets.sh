python run_interpolation.py --interpolate --task sst2 --datasets datasets/sst2_singletoken datasets/sst2_orderedpair datasets/sst2_or --model-paths models/bert-base-cased-sst2-st/checkpoint-8420 models/bert-base-cased-sst2-op/checkpoint-8420 models/bert-base-cased-sst2-or/checkpoint-8420 --models ST OP OR --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl
python run_interpolation.py --interpolate --task sst2 --datasets datasets/sst2_tokenincontext datasets/sst2_orderedpair datasets/sst2_or --model-paths models/bert-base-cased-sst2-tic/checkpoint-8420 models/bert-base-cased-sst2-op/checkpoint-8420 models/bert-base-cased-sst2-or/checkpoint-8420 --models TiC OP OR --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl

python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_ST_synth.png --dataset-for-3model ST_synth
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_ST_orig.png --dataset-for-3model ST_orig

python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_OP_synth.png --dataset-for-3model OP_synth
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_OP_orig.png --dataset-for-3model OP_orig

python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_OR_synth.png --dataset-for-3model OR_synth
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_OR_orig.png --dataset-for-3model OR_orig

python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_synth_avg.png --dataset-for-3model synth_avg
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_ST_OP_OR.pkl --models ST OP OR --plot-name results/ST_OP_OR_on_orig_avg.png --dataset-for-3model orig_avg


python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_TiC_synth.png --dataset-for-3model TiC_synth
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_TiC_orig.png --dataset-for-3model TiC_orig

python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_OP_synth.png --dataset-for-3model OP_synth
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_OP_orig.png --dataset-for-3model OP_orig

python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_OR_synth.png --dataset-for-3model OR_synth
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_OR_orig.png --dataset-for-3model OR_orig

python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_synth_avg.png --dataset-for-3model synth_avg
python run_interpolation.py --plot --task sst2 --interpolation-results results/linear_interp_results_10_TiC_OP_OR.pkl --models TiC OP OR --plot-name results/TiC_OP_OR_on_orig_avg.png --dataset-for-3model orig_avg