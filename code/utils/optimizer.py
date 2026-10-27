import os
import gc
import json
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
import keras
from mealpy import IntegerVar
from utils.localMealpy import getOptimizationAlgorithm
from utils.models import buildTemplateCNN

_PROC = psutil.Process(os.getpid())


def _hostRamGB():
    """Return the resident memory of this Python process, in GB."""
    return _PROC.memory_info().rss / 1024**3


FIXED_LR         = 1e-3
FIXED_DROPOUT    = 0.5
FIXED_ACTIVATION = 'relu'
FIXED_BN         = True

BOUNDS = [
    IntegerVar(lb=0,    ub=2   ),
    IntegerVar(lb=0,    ub=2   ),
    IntegerVar(lb=0,    ub=1   ),
    IntegerVar(lb=0,    ub=2   ),
    IntegerVar(lb=0,    ub=2   ),
    IntegerVar(lb=0,    ub=1   ),
    IntegerVar(lb=0,    ub=1   ),
]

BATCH_SIZES   = [32, 64, 128]

FILTER_OPTS   = [32, 64, 128]

DENSE_OPTS    = [128, 256, 512]

OPTIMIZERS    = ['adamw', 'sgd', 'rmsprop']
POOLING_TYPES = ['max', 'average']

KERNEL_SIZES  = [3, 5]

N_BLOCKS_OPTS = [2, 3]


def decodeSolution(solution):
    """Turn a vector of 7 option indices into a CNN configuration dict.

    solution: indices for batch size, base filters, blocks, dense units,
        optimizer, pooling and kernel size.
    """
    s = solution
    batch_size    = BATCH_SIZES   [int(s[0])]
    base_filters  = FILTER_OPTS   [int(s[1])]
    n_blocks      = N_BLOCKS_OPTS [int(s[2])]
    dense_units   = DENSE_OPTS    [int(s[3])]
    optimizer     = OPTIMIZERS    [int(s[4])]
    pooling       = POOLING_TYPES [int(s[5])]
    kernel_size   = KERNEL_SIZES  [int(s[6])]

    return {
        'learning_rate': FIXED_LR,
        'dropout_rate':  FIXED_DROPOUT,
        'activation':    FIXED_ACTIVATION,
        'use_bn':        FIXED_BN,
        'batch_size':    batch_size,
        'base_filters':  base_filters,
        'n_blocks':      n_blocks,
        'dense_units':   dense_units,
        'optimizer':     optimizer,
        'pooling':       pooling,
        'kernel_size':   kernel_size,
    }


_eval_counter  = {'n': 0, 'best': 0.0, 'start': None}
_eval_log      = []


def makeObjective(data, eval_epochs=5):
    """Build the mealpy objective that trains a CNN in-process and returns one
    minus its accuracy on data['x_test'].

    data: dataset dict from getDataset.
    eval_epochs: training epochs per evaluation.
    """
    def objective(solution):
        """Train the CNN of one solution and return its fitness.

        solution: vector of 7 option indices (see decodeSolution).
        """
        config = decodeSolution(solution)
        model  = None

        oom = False
        try:
            model = buildTemplateCNN(
                config,
                data['input_shape'],
                data['num_classes']
            )
            history = model.fit(
                data['x_train'], data['y_train'],
                validation_data=(data['x_test'], data['y_test']),
                epochs=eval_epochs,
                batch_size=config['batch_size'],
                verbose=0
            )
            val_acc = max(history.history['val_accuracy'])
        except tf.errors.ResourceExhaustedError:
            print(f"  [OOM  ] eval={_eval_counter['n']+1} — config exceeds GPU memory; penalised (val_acc=0)")
            val_acc = 0.0
            oom = True
        except Exception as e:
            print(f"  [ERROR] eval={_eval_counter['n']+1} — {type(e).__name__}: {e}")
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

        keras.backend.clear_session()
        gc.collect()
        gc.collect()

        ram_gb = _hostRamGB()

        fitness = 1.0 - val_acc

        _eval_counter['n'] += 1
        if val_acc > _eval_counter['best']:
            _eval_counter['best'] = val_acc
        elapsed = time.time() - _eval_counter['start']

        _eval_log.append({
            'eval':          _eval_counter['n'],
            'val_accuracy':  round(val_acc,  4),
            'fitness':       round(fitness,  4),
            'learning_rate': config['learning_rate'],
            'dropout_rate':  config['dropout_rate'],
            'batch_size':    config['batch_size'],
            'base_filters':  config['base_filters'],
            'n_blocks':      config['n_blocks'],
            'dense_units':   config['dense_units'],
            'optimizer':     config['optimizer'],
            'activation':    config['activation'],
            'pooling':       config['pooling'],
            'kernel_size':   config['kernel_size'],
            'elapsed_s':     round(elapsed, 1),
            'ram_gb':        round(ram_gb,   3),
        })

        n    = _eval_counter['n']
        best = _eval_counter['best']
        prev_elapsed = _eval_log[-2]['elapsed_s'] if len(_eval_log) >= 2 else 0.0
        dt = elapsed - prev_elapsed
        print(f"  [{n:>4}] val_acc={val_acc:.4f}  best={best:.4f}  "
              f"lr={config['learning_rate']:.5f}  "
              f"dropout_rate={config['dropout_rate']:.2f}  "
              f"batch_size={config['batch_size']}  "
              f"base_filters={config['base_filters']}  "
              f"n_blocks={config['n_blocks']}  "
              f"dense_units={config['dense_units']}  "
              f"optimizer={config['optimizer']}  "
              f"act={config['activation']}  "
              f"pool={config['pooling']}  "
              f"k={config['kernel_size']}  dt={dt:.0f}s  t={elapsed:.0f}s  "
              f"ram={ram_gb:.2f}GB")

        return fitness

    return objective


def makeObjectiveWorker(
    data,
    eval_epochs = 1,
    timeout     = 180,
    base_seed   = 42,
):
    """Same as makeObjective, but every evaluation runs in a fresh subprocess
    to avoid host-RAM leaks.

    data: dataset dict from getDataset.
    eval_epochs: training epochs per evaluation.
    timeout: seconds before an evaluation is killed.
    base_seed: seed of the first evaluation; evaluation N uses base_seed + N.
    """
    from utils.worker import prepareDataFile, evalWorker

    data_paths = prepareDataFile(data)
    print(f"  [worker] dataset serialised to "
          f"{os.path.dirname(data_paths['x_train'])}")

    def objective(solution):
        """Train the CNN of one solution in a subprocess and return its
        fitness.

        solution: vector of 7 option indices (see decodeSolution).
        """
        config = decodeSolution(solution)

        child_seed = int(base_seed) + int(_eval_counter['n'])

        status, result = evalWorker(
            config,
            data_paths,
            eval_epochs = eval_epochs,
            timeout     = timeout,
            seed        = child_seed,
        )

        if status == 'ok':
            val_acc = float(result)
        else:
            val_acc = 0.0
            tag = {'oom': 'OOM  ', 'timeout': 'TMOUT',
                   'error': 'ERROR'}.get(status, '???  ')
            print(f"  [{tag}] eval={_eval_counter['n']+1} — {result}")

        fitness = 1.0 - val_acc

        _eval_counter['n'] += 1
        if val_acc > _eval_counter['best']:
            _eval_counter['best'] = val_acc
        elapsed = time.time() - _eval_counter['start']
        ram_gb  = _hostRamGB()

        _eval_log.append({
            'eval':          _eval_counter['n'],
            'val_accuracy':  round(val_acc,  4),
            'fitness':       round(fitness,  4),
            'learning_rate': config['learning_rate'],
            'dropout_rate':  config['dropout_rate'],
            'batch_size':    config['batch_size'],
            'base_filters':  config['base_filters'],
            'n_blocks':      config['n_blocks'],
            'dense_units':   config['dense_units'],
            'optimizer':     config['optimizer'],
            'activation':    config['activation'],
            'pooling':       config['pooling'],
            'kernel_size':   config['kernel_size'],
            'elapsed_s':     round(elapsed, 1),
            'ram_gb':        round(ram_gb,   3),
            'status':        status,
        })

        n    = _eval_counter['n']
        best = _eval_counter['best']
        prev_elapsed = _eval_log[-2]['elapsed_s'] if len(_eval_log) >= 2 else 0.0
        dt = elapsed - prev_elapsed
        print(f"  [{n:>4}] val_acc={val_acc:.4f}  best={best:.4f}  "
              f"batch_size={config['batch_size']}  "
              f"base_filters={config['base_filters']}  "
              f"n_blocks={config['n_blocks']}  "
              f"dense_units={config['dense_units']}  "
              f"optimizer={config['optimizer']}  "
              f"pool={config['pooling']}  "
              f"k={config['kernel_size']}  dt={dt:.0f}s  t={elapsed:.0f}s  "
              f"ram={ram_gb:.2f}GB  [{status}]")

        return fitness

    return objective


def runOptimization(
    algorithm,
    data,
    eval_epochs  = 2,
    n_generations= 20,
    pop_size     = 25,
    results_dir  = 'results/optimization',
    seed         = None,
    algo_params  = None,
    extra_info   = True,
    use_worker   = False,
    worker_timeout = 180,
):
    """Run a full HPO search with a mealpy metaheuristic and save the result
    JSON and evaluation log CSV.

    algorithm: 'GA', 'SADE', 'AIW_PSO' or 'GWO'.
    data: dataset dict from getDataset.
    eval_epochs: training epochs per evaluation.
    n_generations: number of generations; the budget is (n_generations + 1) *
        pop_size.
    pop_size: population size.
    results_dir: folder for the output files.
    seed: random seed; drawn at random and saved with the results when None.
    algo_params: optional overrides of the algorithm parameters.
    extra_info: print a final summary.
    use_worker: evaluate each solution in an isolated subprocess.
    worker_timeout: seconds per evaluation when use_worker is True.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
    seed = int(seed)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    _eval_counter['n']     = 0
    _eval_counter['best']  = 0.0
    _eval_counter['start'] = time.time()
    _eval_log.clear()

    alg_name = algorithm.upper()
    total_evals = (n_generations + 1) * pop_size

    print(f"\n{'='*62}")
    print(f"  OPTIMIZATION: {alg_name}")
    print(f"  Generations: {n_generations}  |  Population: {pop_size}  |  "
          f"Total evals: {total_evals}  (init: {pop_size}  +  generations: {n_generations * pop_size})")
    print(f"  Epochs per evaluation: {eval_epochs}")
    print(f"  Search space dimensions: {len(BOUNDS)}  "
          f"(batch, base_filters, n_blocks, dense, opt, pool, kernel)")
    print(f"  Fixed: lr={FIXED_LR}  dropout={FIXED_DROPOUT}  "
          f"activation={FIXED_ACTIVATION}  batch_norm={FIXED_BN}")
    print(f"  Evaluation mode: "
          f"{'subprocess-isolated (worker)' if use_worker else 'in-process'}")
    print(f"  Seed:            {seed}")
    print(f"{'='*62}\n")

    if use_worker:
        objective = makeObjectiveWorker(
            data,
            eval_epochs = eval_epochs,
            timeout     = worker_timeout,
            base_seed   = seed,
        )
    else:
        objective = makeObjective(
            data,
            eval_epochs = eval_epochs,
        )

    problem = {
        "obj_func": objective,
        "bounds":   BOUNDS,
        "minmax":   "min",
        "log_to":   None,
    }

    algo = getOptimizationAlgorithm(alg_name, n_generations, pop_size, algo_params)

    t_start = time.time()
    g_best = algo.solve(problem, seed=seed)
    t_total = time.time() - t_start

    best_solution = g_best.solution
    best_fitness  = g_best.target.fitness
    best_config   = decodeSolution(best_solution)
    best_val_acc  = 1.0 - best_fitness

    if extra_info:
        print(f"\n{'='*62}")
        print(f"  FINAL RESULT — {alg_name}")
        print(f"{'='*62}")
        print(f"  Best val_accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
        print(f"  Total time        : {t_total:.0f}s  ({t_total/60:.1f} min)")
        print(f"  Evaluations done  : {_eval_counter['n']}")
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
            'eval_epochs': eval_epochs,
            'n_generations': n_generations,
            'pop_size':    pop_size,
            'seed':        seed,
        },
        'best': {
            'config':       best_config,
            'val_accuracy': round(best_val_acc, 4),
            'fitness':      round(best_fitness, 4),
        },
        'total_time_s':  round(t_total, 1),
        'n_evaluations': _eval_counter['n'],
        'eval_log':      _eval_log.copy(),
    }

    json_path = os.path.join(results_dir, f'{alg_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    print()
    print(f"  Results saved : {json_path}")

    df_log = pd.DataFrame(_eval_log)
    csv_path = os.path.join(results_dir, f'{alg_name}_{ts}_evals.csv')
    df_log.to_csv(csv_path, index=False)
    print(f"  Eval log      : {csv_path}")
    print()

    return best_config, run_results
