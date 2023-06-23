from functools import reduce
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata
import numpy as np
from fusion.interpolate import interpolate_and_evaluate_2d, interpolate_and_evaluate_3d
from fusion.utils.eval import TASKS
import pickle
import argparse


def interpolate(task, dataset_paths, model_paths, model_names, steps, output_path):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenizer.add_tokens(['zeroa', 'onea', 'synt'])

    models = []
    for model_name, path in zip(model_names, model_paths):
        model = AutoModelForSequenceClassification.from_pretrained(path)
        models.append(model)

    if len(model_names) == 2:
        results = interpolate_and_evaluate_2d(task, *models, tokenizer, dataset_paths, steps)
    elif len(model_names) == 3:
        results = interpolate_and_evaluate_3d(task, *models, tokenizer, dataset_paths, steps)
    else:
        raise NotImplementedError("Number of models must not exceed 3.")
    
    interp_eval_results = {}
    for name, path in zip(model_names, dataset_paths):
        interp_eval_results[f"{name}_synth"] = results[path]['synthetic']
        interp_eval_results[f"{name}_orig"] = results[path]['original']


    with open(output_path, 'wb') as f:
        pickle.dump(interp_eval_results, f)

def _plot_2d(results, models):
    styles = [('r', '-'), ('b', '--')]
    
    for model, style in zip(models, styles):
        model_synth_results, model_orig_results = results[f"{model}_synth"], results[f"{model}_orig"]
        steps = len(model_orig_results) - 1
        coeffs = [i/steps for i in range(steps+1)]

        color, line_style = style
        plt.plot(coeffs, model_synth_results, f'{color}+{line_style}', label=f'shortcut-{model}', linewidth=2.0, markersize=10, 
                 markeredgewidth=3.0)
        plt.plot(coeffs, model_orig_results, f'{color}o{line_style}', label=f'orig-{model}', linewidth=2.0, markersize=10, alpha=0.4)

        #plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.xlabel(f"$\\alpha_{{{model}}}$", fontsize=14)
    x_ticks = list(filter(lambda x: 0.0 <= x <= 1.0, map(lambda x: round(x, 2), plt.xticks()[0])))
    tick_map = {'0.0': f'0.0\n{models[0]}', '1.0': f'1.0\n{models[1]}'}
    xtick_labels = [tick_map[str(tick)] if str(tick) in tick_map else str(tick) for tick in x_ticks]
    
    plt.xticks(labels=xtick_labels, ticks=x_ticks, fontsize=14)
    plt.ylabel(f"Accuracy", fontsize=14)
    #plt.title(f'Interpolation {models[0]} \u2192 {models[1]}', fontsize=18)
    plt.legend(loc="right", prop={'size': 14})


def _create_points(n_steps = 10):
  m1 = np.array([0, 0])
  m2 = np.array([0.5, np.sqrt(3.)])
  m3 = np.array([1, 0])
  points = []

  v1 = m2 - m1
  v2 = m3  - m1

  for i in range(n_steps + 1):
    for j in range(n_steps + 1):
      alpha, beta = i / n_steps, j / n_steps
      new_point = (alpha*v1 + m1) + (beta*v2 + m1)
      points.append(new_point)

  return points

def _plot_3d(results, models, n_steps = 10):
    x, y = np.linspace(0, 1, 100), np.linspace(0, 2, 200)
    X, Y = np.meshgrid(x, y)

    Z = []
    p = np.array(_create_points(n_steps))
    for i, j, val in results:
        Z.append(val)

    px, py = p[:, 0], p[:, 1]

    Ti = griddata((px, py), Z, (X, Y), method='linear')
    
    # just keep triangle
    Ti[Y - 2*np.sqrt(3) + 2*np.sqrt(3)*X > 0] = np.nan
    Ti[Y - 2*np.sqrt(3)*X > 0] = np.nan
    Ti[Y < 0] = np.nan
    
    c = plt.pcolormesh(X, Y, Ti)
    cbar = plt.colorbar(c, pad=0.15)
    cbar.ax.set_ylabel('Accuracy', fontsize=14)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=14)
    
    plt.xlim(0.0, 1.0)
    plt.ylim(0., 2.0)

    model_xs, model_ys, models, offsets = np.array([0., .5, 1.]), np.array([0., np.sqrt(3), 0.]), models, [(-0.09, 0.00), (0, 0), (0.0, 0.0)]
    for i, model_name in enumerate(models):
        x = model_xs[i] + offsets[i][0]
        y = (model_ys[i] / 2.0) + offsets[i][1]
        plt.text(x, y, model_name, fontsize=18, transform=plt.gca().transAxes)

    plt.xticks(labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

def plot(fname, models, img_path, nsteps = 10, dataset = None):
    
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    if len(models) == 2:
        _plot_2d(results, models)
    elif len(models) == 3:
        if dataset == 'synth_avg':
            results_filtered = dict(filter(lambda x: x[0].endswith('_synth'), results.items()))
            results = list(map(lambda x: reduce(lambda a, b: (a[0], a[1], a[2] + b[2]), x), zip(*results_filtered.values())))
            results = list(map(lambda x: (x[0], x[1], x[2] / 3.0), results))
        elif dataset == 'orig_avg':
            results_filtered = dict(filter(lambda x: x[0].endswith('_orig'), results.items()))
            results = list(map(lambda x: reduce(lambda a, b: (a[0], a[1], a[2] + b[2]), x), zip(*results_filtered.values())))
            results = list(map(lambda x: (x[0], x[1], x[2] / 3.0), results))
        else:
            results = results[dataset]
        
        _plot_3d(results, models, nsteps)
    

    plt.savefig(img_path, dpi=400, bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform interpolation for given models on given dataset and plot')
    parser.add_argument('--dataset-for-3model', help='', required=False)
    
    parser.add_argument('--task', help='Task for which a model is trained', choices=TASKS, required=True)
    parser.add_argument('--models', nargs='+', help='Shortnames for models to be used for interpolation', required=True)
    parser.add_argument('--datasets', nargs='+', help='Paths to datasets', required=False)
    parser.add_argument('--model-paths', nargs='+', help='Paths to models', required=False)

    parser.add_argument('--steps', help='Number of steps for interpolation', type=int, default=10, required=False)
    
    parser.add_argument('--interpolation-results', help='', required=False)
    parser.add_argument('--path', help='', required=False)
    parser.add_argument('--plot-name', help='', required=False)
    
    parser.add_argument('--interpolate', action='store_true', required=False)
    parser.add_argument('--plot', action='store_true', required=False)

    args = parser.parse_args()

    if args.interpolate:
        interpolate(args.task, dataset_paths=args.datasets, model_paths=args.model_paths, 
                    model_names=args.models, steps=args.steps, output_path=args.interpolation_results)
    
    if args.plot:
        plot(args.interpolation_results, args.models, args.plot_name, args.steps, args.dataset_for_3model)
        