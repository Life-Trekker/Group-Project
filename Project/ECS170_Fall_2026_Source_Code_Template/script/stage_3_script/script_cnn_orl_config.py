from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os

#---- directory location section ----------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data') + os.sep
GRAPH_DIR = os.getcwd() + os.sep
#------------------------------------------------------

#---- experiment section ------------------------------
SEED = 2
MAX_EPOCH = 15

BASELINE = dict(conv_channels=(32, 64), kernel_size=3, padding=1, stride=1,
                pool='max', hidden_dim=128, loss='ce')

EXPERIMENTS = [
    # ('Depth: 1 conv layer', 'depth_1conv', dict(conv_channels=(32,))),
    # ('Kernel size: 7x7', 'kernel_7x7', dict(kernel_size=7, padding=3)),
    # ('Loss: MSE (softmax vs one-hot)', 'loss_mse', dict(loss='mse')),
    # ('Hidden dim: 32', 'hidden_32', dict(hidden_dim=32)),
    ('Depth + kernel: 1 conv layer, 7x7', 'depth_1conv_kernel_7x7',
     dict(conv_channels=(32,), kernel_size=7, padding=3)),
]
#------------------------------------------------------


def describe_change(overrides):
    if not overrides:
        return 'BASELINE - reference model, nothing changed'
    return 'CHANGED from baseline: ' + ', '.join(f'{k} {BASELINE[k]} -> {v}' for k, v in overrides.items())


def print_hyperparameters(model, cfg):
    settings = {**cfg, 'max_epoch': model.max_epoch, 'batch_size': model.batch_size,
                'learning_rate': model.learning_rate, 'optimizer': 'Adam',
                'parameters': f'{sum(p.numel() for p in model.parameters()):,}'}
    print('--hyperparameters:')
    for name, value in settings.items():
        print(f'    {name:<14}{value}')


def run_config(data, overrides):
    cfg = {**BASELINE, **overrides}
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = Method_CNN_ORL('cnn orl', '', in_channels=1, image_size=(112, 92), num_classes=40, **cfg)
    model.max_epoch = MAX_EPOCH
    model.data = data

    print(describe_change(overrides))
    print_hyperparameters(model, cfg)

    start = time.time()
    out = model.run()
    seconds = time.time() - start

    evaluator = Evaluate_Metrics('metrics', '')
    evaluator.data = out
    metrics = evaluator.evaluate_all()

    n_params = sum(p.numel() for p in model.parameters())
    return metrics, model.history, n_params, seconds


def save_run_plot(label, history, fig_path):
    h = history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(h['epoch'], h['loss'], marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax1.set_title(f'{label} - Training Loss')
    ax1.grid(True)

    ax2.plot(h['epoch'], h['train_acc'], marker='o', label='Train')
    ax2.plot(h['epoch'], h['test_acc'], marker='s', label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{label} - Accuracy per Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'saved plot  -> {fig_path}')


if __name__ == '__main__':
    print('************ Loading ORL ************')
    loader = Dataset_Loader('ORL', '')
    loader.dataset_source_folder_path = DATA_DIR
    loader.dataset_source_file_name = 'ORL'
    loader.dataset_name = 'ORL'
    data = loader.load()
    print(f"train: {data['train']['X'].shape}, test: {data['test']['X'].shape}")

    rows = []
    for i, (label, name, overrides) in enumerate(EXPERIMENTS):
        print(f'\n************ {label} ************')
        metrics, history, n_params, seconds = run_config(data, overrides)
        print(f"--test results: accuracy {metrics['accuracy']:.4f}  precision {metrics['precision']:.4f}  "
              f"recall {metrics['recall']:.4f}  f1 {metrics['f1']:.4f}")
        save_run_plot(label, history, GRAPH_DIR + f'orl_config_{i + 1}_{name}_convergence_curves.png')
        reach = next((e for e, a in zip(history['epoch'], history['test_acc']) if a >= 0.9), None)
        rows.append(dict(label=label, metrics=metrics, history=history,
                         n_params=n_params, seconds=seconds, reach=reach))

    #---- results table -----------------------------------
    print('\n************ Configuration Results (ORL test set) ************')
    print(f"{'config':<34}{'acc':>7}{'prec':>7}{'recall':>8}{'f1':>7}{'best':>7}{'ep>=90%':>9}"
          f"{'params':>12}{'sec':>7}")
    for r in rows:
        h, m = r['history'], r['metrics']
        print(f"{r['label']:<34}{m['accuracy']:>7.3f}{m['precision']:>7.3f}{m['recall']:>8.3f}"
              f"{m['f1']:>7.3f}{max(h['test_acc']):>7.3f}"
              f"{str(r['reach'] or '-'):>9}{r['n_params']:>12,}{r['seconds']:>7.1f}")

