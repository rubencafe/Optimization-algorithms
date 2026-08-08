"""
optimizer_bo.py
---------------
Bayesian-Optimization baseline for the same HPO problem as utils/optimizer.py.

This module is a drop-in companion to ``runOptimization`` (mealpy-based
evolutionary algorithms) that uses Gaussian-Process Bayesian Optimization
via Fernando Nogueira's ``bayesian-optimization`` package
(https://bayesian-optimization.github.io/BayesianOptimization/) with the
Expected Improvement acquisition function.

It mirrors the protocol of the EA baseline so all results are directly
comparable:

    - Same search space (7 categorical hyperparameters, encoded as integer
      indices into the option lists exposed by ``utils.optimizer``)
    - Same fitness function (``1 - val_accuracy`` after ``eval_epochs``
      epochs); internally the GP maximizes val_accuracy directly because
      ``bayesian-optimization`` is a maximizer.
    - Same evaluation budget (default 220 trials = 20 random init points
      + 200 BO-guided iterations)
    - Same per-evaluation log structure (JSON + CSV) and same JSON output
      schema, so existing aggregation / plotting code works unchanged
    - Same per-evaluation cleanup (clear_session + gc.collect) to avoid
      the host-RAM leak that affects long Keras/TF runs

Quick usage:

    from utils.datasets      import getDataset
    from utils.optimizer_bo  import runBOOptimization

    data = getDataset('mnist')
    best, run = runBOOptimization(
        data,
        eval_epochs = 1,
        n_iter      = 200,
        init_points = 20,
        seed        = 0,
        results_dir = 'colab/results/bo',
    )
"""

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
    return _PROC.memory_info().rss / 1024 ** 3


# Search-space bounds for bayesian-optimization. We use integer indices
# (third tuple element ``int``) so that the GP kernel is rounding-aware
# and the kernel-space dimension stays at 7 (instead of one-hot exploding
# to 18 dimensions).
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
    """Decode integer indices into the same config dict produced by
    ``utils.optimizer.decodeSolution``."""
    return {
        # Fixed hyperparameters
        'learning_rate': FIXED_LR,
        'dropout_rate':  FIXED_DROPOUT,
        'activation':    FIXED_ACTIVATION,
        'use_bn':        FIXED_BN,
        # Tuned
        'batch_size':    BATCH_SIZES   [int(batch_idx)],
        'base_filters':  FILTER_OPTS   [int(filters_idx)],
        'n_blocks':      N_BLOCKS_OPTS [int(n_blocks_idx)],
        'dense_units':   DENSE_OPTS    [int(dense_idx)],
        'optimizer':     OPTIMIZERS    [int(optimizer_idx)],
        'pooling':       POOLING_TYPES [int(pooling_idx)],
        'kernel_size':   KERNEL_SIZES  [int(kernel_idx)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def _makeBOObjective(data, eval_epochs, eval_log, counter):
    """Closure returning an objective compatible with
    ``BayesianOptimization`` (keyword-arg signature, maximizes target).

    Returns ``val_accuracy`` directly because the library maximises; the
    fitness (``1 - val_accuracy``) is logged alongside for consistency
    with the EA baseline log format.
    """

    def objective(batch_idx, filters_idx, n_blocks_idx, dense_idx, optimizer_idx, pooling_idx, kernel_idx):
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

        # ── Aggressive per-evaluation cleanup ─────────────────────────────
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

        # Library MAXIMIZES — return the accuracy directly.
        return val_acc

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

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
    """Run a full HPO loop with GP-based Bayesian Optimization.

    Parameters
    ----------
    data : dict
        Dataset dict as returned by ``utils.datasets.getDataset``.
    eval_epochs : int, default 1
        Number of epochs used to score each candidate (same as the EA
        baselines for a fair comparison).
    n_iter : int, default 200
        Number of BO-guided iterations after the random initialization.
        The total budget is ``init_points + n_iter`` (default ``20 + 200
        = 220``, matching the EA budget of 20 initial individuals + 10
        generations of 20 offspring).
    init_points : int, default 20
        Random points sampled before the GP starts proposing.
    xi : float, default 0.01
        Exploration trade-off parameter for the Expected Improvement
        acquisition function.
    seed : int or None
        If None, a fresh seed is drawn from OS entropy. Persisted in the
        output JSON so the run is reproducible.
    results_dir : str
        Folder where the JSON and CSV are written.
    extra_info : bool
        Print a final summary block.

    Returns
    -------
    best_config : dict
        Same schema as the EA baselines' ``best['config']``.
    run_results : dict
        Same schema as the EA baselines' ``run_results``, so the existing
        aggregation/plotting code in ``utils.evaluation`` works unchanged.
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
        verbose              = 0,        # we print our own per-iteration line
    )

    t_start = time.time()
    bo.maximize(init_points=init_points, n_iter=n_iter)
    t_total = time.time() - t_start

    # Library's max is val_accuracy; convert back to the EA-style fitness.
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
