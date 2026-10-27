import gc
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil

import keras
import tensorflow as tf

from bayes_opt import BayesianOptimization, acquisition

from utils.models import buildTemplateCNN
from utils.optimizer import (
    BATCH_SIZES, FILTER_OPTS, N_BLOCKS_OPTS, DENSE_OPTS,
    OPTIMIZERS, POOLING_TYPES, KERNEL_SIZES,
    FIXED_LR, FIXED_DROPOUT, FIXED_ACTIVATION, FIXED_BN,
)


_PROC = psutil.Process(os.getpid())


def _hostRamGB():
    """Return the resident memory of this Python process, in GB."""
    return _PROC.memory_info().rss / 1024 ** 3


BO_PBOUNDS = {
    'batch_idx':     (0, len(BATCH_SIZES)   - 1, int),
    'filters_idx':   (0, len(FILTER_OPTS)   - 1, int),
    'n_blocks_idx':  (0, len(N_BLOCKS_OPTS) - 1, int),
    'dense_idx':     (0, len(DENSE_OPTS)    - 1, int),
    'optimizer_idx': (0, len(OPTIMIZERS)    - 1, int),
    'pooling_idx':   (0, len(POOLING_TYPES) - 1, int),
    'kernel_idx':    (0, len(KERNEL_SIZES)  - 1, int),
}


def _decodeIndices(batch_idx, filters_idx, n_blocks_idx, dense_idx,
                   optimizer_idx, pooling_idx, kernel_idx):
    """Turn the 7 BO parameters into the same configuration dict as
    decodeSolution.

    batch_idx, filters_idx, n_blocks_idx, dense_idx, optimizer_idx,
        pooling_idx, kernel_idx: option indices.
    """
    return {
        'learning_rate': FIXED_LR,
        'dropout_rate':  FIXED_DROPOUT,
        'activation':    FIXED_ACTIVATION,
        'use_bn':        FIXED_BN,
        'batch_size':    BATCH_SIZES   [int(batch_idx)],
        'base_filters':  FILTER_OPTS   [int(filters_idx)],
        'n_blocks':      N_BLOCKS_OPTS [int(n_blocks_idx)],
        'dense_units':   DENSE_OPTS    [int(dense_idx)],
        'optimizer':     OPTIMIZERS    [int(optimizer_idx)],
        'pooling':       POOLING_TYPES [int(pooling_idx)],
        'kernel_size':   KERNEL_SIZES  [int(kernel_idx)],
    }


def _makeBOObjective(data, eval_epochs, eval_log, counter):
    """Build the BO objective that trains a CNN and returns its accuracy (BO
    maximizes).

    data: dataset dict from getDataset.
    eval_epochs: training epochs per evaluation.
    eval_log: list that receives one record per evaluation.
    counter: dict holding the evaluation count and best accuracy so far.
    """

    def objective(batch_idx, filters_idx, n_blocks_idx, dense_idx, optimizer_idx, pooling_idx, kernel_idx):
        """Train the CNN of one BO point and return its accuracy.

        batch_idx, filters_idx, n_blocks_idx, dense_idx, optimizer_idx,
            pooling_idx, kernel_idx: option indices proposed by BO.
        """
        config = _decodeIndices(
            batch_idx, filters_idx, n_blocks_idx, dense_idx,
            optimizer_idx, pooling_idx, kernel_idx,
        )

        model   = None
        history = None
        val_acc = 0.0
        try:
            model = buildTemplateCNN(
                config,
                data['input_shape'],
                data['num_classes'],
            )
            history = model.fit(
                data['x_train'], data['y_train'],
                validation_data=(data['x_test'], data['y_test']),
                epochs=eval_epochs,
                batch_size=config['batch_size'],
                verbose=0,
            )
            val_acc = max(history.history['val_accuracy'])
        except tf.errors.ResourceExhaustedError:
            print(f"  [OOM  ] trial={counter['n']+1} - config exceeds GPU memory; penalised (val_acc=0)")
            val_acc = 0.0
        except Exception as e:
            print(f"  [ERROR] trial={counter['n']+1} - {type(e).__name__}: {e}")
            val_acc = 0.0

        try:
            if model is not None and hasattr(model, 'optimizer'):
                model.optimizer = None
        except Exception:
            pass
        try:
            del model
            del history
        except NameError:
            pass
        gc.collect()
        try:
            keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

        fitness = 1.0 - val_acc
        ram_gb  = _hostRamGB()

        counter['n'] += 1
        if val_acc > counter['best']:
            counter['best'] = val_acc
        elapsed = time.time() - counter['start']

        eval_log.append({
            'eval':          counter['n'],
            'val_accuracy':  round(val_acc, 4),
            'fitness':       round(fitness, 4),
            'learning_rate': config['learning_rate'],
            'dropout_rate':  config['dropout_rate'],
            'batch_size':    config['batch_size'],
            'base_filters':  config['base_filters'],
            'n_blocks':      config['n_blocks'],
            'dense_units':   config['dense_units'],
            'use_bn':        config['use_bn'],
            'optimizer':     config['optimizer'],
            'activation':    config['activation'],
            'pooling':       config['pooling'],
            'kernel_size':   config['kernel_size'],
            'elapsed_s':     round(elapsed, 1),
            'ram_gb':        round(ram_gb,  3),
        })

        prev_elapsed = eval_log[-2]['elapsed_s'] if len(eval_log) >= 2 else 0.0
        dt = elapsed - prev_elapsed
        print(f"  [{counter['n']:>4}] val_acc={val_acc:.4f}  best={counter['best']:.4f}  "
              f"bs={config['batch_size']}  f0={config['base_filters']}  "
              f"blocks={config['n_blocks']}  dense={config['dense_units']}  "
              f"opt={config['optimizer']}  pool={config['pooling']}  "
              f"k={config['kernel_size']}  dt={dt:.0f}s  t={elapsed:.0f}s  "
              f"ram={ram_gb:.2f}GB")

        return val_acc

    return objective


def runBOOptimization(
    data,
    eval_epochs  = 1,
    n_iter       = 200,
    init_points  = 20,
    xi           = 0.01,
    seed         = None,
    results_dir  = 'results/bo',
    extra_info   = True,
):
    """Run the GP-based Bayesian Optimization baseline and save the result JSON
    and evaluation log CSV.

    data: dataset dict from getDataset.
    eval_epochs: training epochs per evaluation.
    n_iter: BO iterations after the random start; the budget is init_points +
        n_iter.
    init_points: random points evaluated before the GP starts proposing.
    xi: exploration parameter of Expected Improvement.
    seed: random seed; drawn at random and saved with the results when None.
    results_dir: folder for the output files.
    extra_info: print a final summary.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    eval_log = []
    counter  = {'n': 0, 'best': 0.0, 'start': time.time()}

    alg_name = 'BO'
    total_evals = init_points + n_iter

    print(f"\n{'='*62}")
    print(f"  OPTIMIZATION: {alg_name}  (bayesian-optimization, GP + Expected Improvement)")
    print(f"  Trials: {total_evals}  (random init: {init_points}  +  BO-guided: {n_iter})")
    print(f"  Epochs per evaluation: {eval_epochs}")
    print(f"  Search space dimensions: 7 (batch, base_filters, n_blocks, dense, opt, pool, kernel)")
    print(f"  Fixed: lr={FIXED_LR}  dropout={FIXED_DROPOUT}  "
          f"activation={FIXED_ACTIVATION}  batch_norm={FIXED_BN}")
    print(f"  Seed:  {seed}")
    print(f"{'='*62}\n")

    objective = _makeBOObjective(data, eval_epochs, eval_log, counter)

    acq = acquisition.ExpectedImprovement(xi=xi, random_state=seed)
    bo = BayesianOptimization(
        f                    = objective,
        pbounds              = BO_PBOUNDS,
        acquisition_function = acq,
        random_state         = seed,
        verbose              = 0,
    )

    t_start = time.time()
    bo.maximize(init_points=init_points, n_iter=n_iter)
    t_total = time.time() - t_start

    best_val_acc = float(bo.max['target'])
    best_fitness = 1.0 - best_val_acc
    best_params  = bo.max['params']
    best_config  = _decodeIndices(
        best_params['batch_idx'],
        best_params['filters_idx'],
        best_params['n_blocks_idx'],
        best_params['dense_idx'],
        best_params['optimizer_idx'],
        best_params['pooling_idx'],
        best_params['kernel_idx'],
    )

    if extra_info:
        print(f"\n{'='*62}")
        print(f"  FINAL RESULT - {alg_name}")
        print(f"{'='*62}")
        print(f"  Best val_accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
        print(f"  Total time        : {t_total:.0f}s  ({t_total/60:.1f} min)")
        print(f"  Evaluations done  : {counter['n']}")
        print(f"\n  Best configuration found:")
        for k, v in best_config.items():
            print(f"    {k:<16} {v}")
        print()

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_results = {
        'algorithm':    alg_name,
        'timestamp':    datetime.now().isoformat(),
        'config': {
            'eval_epochs':  eval_epochs,
            'n_iter':       n_iter,
            'init_points':  init_points,
            'xi':           xi,
            'seed':         seed,
        },
        'best': {
            'config':       best_config,
            'val_accuracy': round(best_val_acc, 4),
            'fitness':      round(best_fitness, 4),
        },
        'total_time_s':  round(t_total, 1),
        'n_evaluations': counter['n'],
        'eval_log':      eval_log.copy(),
    }

    json_path = os.path.join(results_dir, f'{alg_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    print(f"  Results saved : {json_path}")

    df_log = pd.DataFrame(eval_log)
    csv_path = os.path.join(results_dir, f'{alg_name}_{ts}_evals.csv')
    df_log.to_csv(csv_path, index=False)
    print(f"  Eval log      : {csv_path}\n")

    return best_config, run_results
