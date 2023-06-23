from functools import reduce
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from fusion.interpolate import interpolate_and_evaluate_fairness
from fusion.utils.eval import TASKS
import pickle
import argparse

def interpolate(task, dataset, model_paths, model_names, steps, protected_attributes, output_path):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    models = []
    for path in model_paths:
        model = AutoModelForSequenceClassification.from_pretrained(path)
        models.append(model)

    results = interpolate_and_evaluate_fairness(task, protected_attributes, *models, tokenizer, dataset, steps)
    
    interp_eval_results = results

    with open(output_path, 'wb') as f:
        pickle.dump(interp_eval_results, f)

def _plot_2d(results, metric, protected_attributes):
    styles = [('r', '-'), ('b', '--')]
    markers = '+o'
    metrics = {'accuracy': 'Accuracy', 'demographic_parity': 'DP', 'tpr_gap': 'TPR_GAP'}
    
    for attr, style, marker in zip(protected_attributes, styles, markers):
        if metric not in results:
            raise ValueError(f"No results with {metric} found.")
        model_results = results[metric]
        steps = len(model_results) - 1
        coeffs = [i/steps for i in range(steps+1)]

        color, line_style = style
        res = model_results if metric == 'accuracy' else list(map(lambda x: x[attr], model_results))
        if marker == 'o':
            plt.plot(coeffs, res, f'{color}{marker}{line_style}', label=f'{attr}', linewidth=2.0, markersize=10)
        else:
            plt.plot(coeffs, res, f'{color}{marker}{line_style}', label=f'{attr}', linewidth=2.0, markersize=10, 
                     markeredgewidth=3.0)

        #plt.title(f'{metric} {models[0]} \u2192 {models[1]}', fontsize=18)
        
        if metric == 'accuracy':
            break

    #plt.title(f'{protected_attributes[0]}-biased \u2192 {protected_attributes[1]}-biased', fontsize=18)
    #plt.ylim(0.8, 0.9)

    x_ticks = list(filter(lambda x: 0.0 <= x <= 1.0, map(lambda x: round(x, 2), plt.xticks()[0])))
    tick_map = {'0.0': f'0.0\n{protected_attributes[0]}', '1.0': f'1.0\n{protected_attributes[1]}'}
    xtick_labels = [tick_map[str(tick)] if str(tick) in tick_map else str(tick) for tick in x_ticks]
    
    plt.xticks(labels=xtick_labels, ticks=x_ticks, fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(loc="upper left", prop={'size': 14})
    plt.xlabel(f"$\\alpha_{{{protected_attributes[1]}}}$", fontsize=14)
    plt.ylabel(metrics[metric], fontsize=14)
    
    #plt.title(metric)


def plot(fname, img_path, metric, protected_attributes):
    
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    _plot_2d(results, metric, protected_attributes)

    plt.savefig(img_path, dpi=400, bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform interpolation for given models on given dataset and plot')
    
    parser.add_argument('--task', help='Task for which a model is trained', choices=TASKS, required=True)
    parser.add_argument('--models', nargs='+', help='Shortnames for models to be used for interpolation', required=True)
    parser.add_argument('--dataset', help='Paths to dataset', required=False)
    parser.add_argument('--model-paths', nargs='+', help='Paths to models', required=False)
    
    parser.add_argument('--protected-attributes', nargs='+', help='Paths to models', required=False)

    parser.add_argument('--steps', help='Number of steps for interpolation', type=int, default=10, required=False)
    
    parser.add_argument('--interpolation-results', help='', required=False)
    parser.add_argument('--path', help='', required=False)
    parser.add_argument('--plot-name', help='', required=False)
    parser.add_argument('--metric', help='Metric to plot', choices=['accuracy', 'f1', 'demographic_parity', 'tpr_gap'], required=False)
    parser.add_argument('--interpolate', action='store_true', required=False)
    parser.add_argument('--plot', action='store_true', required=False)

    args = parser.parse_args()

    if args.interpolate:
        interpolate(args.task, dataset=args.dataset, model_paths=args.model_paths, 
                    model_names=args.models, steps=args.steps, protected_attributes=args.protected_attributes, 
                    output_path=args.interpolation_results)
    
    if args.plot:
        plot(args.interpolation_results, args.plot_name, args.metric, args.protected_attributes)
        