"""
optimizer.py
------------
Hyperparameter optimization of a CNN.

Search space (7 tuned dimensions; learning_rate, dropout_rate, activation
and batch normalization are fixed constants — see FIXED_* in module body):
    [0]  batch_idx      : IntegerVar [0, 2]  -> [32, 64, 128]
    [1]  filters_idx    : IntegerVar [0, 2]  -> [32, 64, 128] base filters (x2/block)
    [2]  n_blocks_idx   : IntegerVar [0, 1]  -> [2, 3] VGG-style conv blocks
    [3]  dense_idx      : IntegerVar [0, 2]  -> [128, 256, 512]
    [4]  optimizer_idx  : IntegerVar [0, 2]  -> adamw, sgd, rmsprop
    [5]  pooling_idx    : IntegerVar [0, 1]  -> max, average
    [6]  kernel_idx     : IntegerVar [0, 1]  -> 3, 5

Total discrete configurations: 3·3·2·3·3·2·2 = 648

Design notes:
    - learning_rate, dropout_rate and activation are FIXED (see FIXED_* block):
      multi-fidelity HPO arguments (Li et al., 2018; Falkner et al., 2018) show
      that their ranking is unstable under short evaluation budgets. Fixing
      them concentrates the search budget on topology-dependent choices.
    - weight_decay is NOT tuned. Kept at the Keras default (0.0) via
      buildTemplateCNN's `config.get('weight_decay', 0.0)` fallback.
    - n_blocks ∈ {2, 3}. The 4-block option was removed because on 32×32
      CIFAR inputs it yielded 2×2 feature maps that rarely outperformed 3
      blocks while being the main driver of GPU OOM and host-RAM leaks.
    - Double Conv2D per block is FIXED (VGG-style, Simonyan & Zisserman 2015).
    - 3rd optimizer option 'rmsprop' (Hinton 2012) added to {adamw, sgd}
      for broader coverage of optimizer families.

Quick usage:
    from Utils.optimizer import makeObjective, compare_algorithms

    # Optimize with PSO
    best, history = makeObjective('PSO', data, eval_epochs=5, n_generations=20, pop_size=20)

    # Compare PSO vs GA vs DE
    results = compare_algorithms(['PSO', 'GA', 'DE'], data, eval_epochs=5)
"""

import os
import gc
import json
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import keras
from mealpy import FloatVar, IntegerVar
from utils.evaluation import plot_algorithm_comparison
from utils.localMealpy import getOptimizationAlgorithm
from utils.models import buildTemplateCNN

# Process handle used by the objective to log host RAM usage per evaluation.
_PROC = psutil.Process(os.getpid())


def _hostRamGB():
    """Resident Set Size of the current Python process, in GB."""
    return _PROC.memory_info().rss / 1024**3



# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ['batch_idx', 'filters_idx', 'n_blocks_idx', 'dense_idx', 'optimizer_idx', 'pooling_idx', 'kernel_idx']

# ─────────────────────────────────────────────────────────────────────────────
# Fixed hyperparameters (NOT searched)
# ─────────────────────────────────────────────────────────────────────────────
# Following the methodological choice adopted by several recent evolutionary-HPO
# studies (e.g. Loussaief & Abdelkrim, 2018; Fedorov et al., 2019), we fix the
# four hyperparameters that have well-established, robust default values in the
# deep-learning literature. This cuts the search dimensionality from 11 to 7,
# concentrating the (small) evaluation budget of the metaheuristic on the
# genuinely architecture-dependent choices.
#
# FIXED_LR = 1e-3
#   The default learning rate for Adam/AdamW (Kingma & Ba, 2015). Empirically
#   the single most common starting value in published CIFAR-10 work.
#   Ref: Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic
#        optimisation. ICLR 2015. https://arxiv.org/abs/1412.6980
#
# FIXED_DROPOUT = 0.5
#   The value recommended by the original dropout paper for fully-connected
#   layers (Srivastava et al., 2014).
#   Ref: Srivastava, N. et al. (2014). Dropout: A simple way to prevent
#        neural networks from overfitting. JMLR 15(56):1929-1958.
#        https://jmlr.org/papers/v15/srivastava14a.html
#
# FIXED_ACTIVATION = 'relu'
#   ReLU is the de-facto activation for CNNs on CIFAR-like inputs since
#   AlexNet (Krizhevsky et al., 2012); variants (ELU, Leaky-ReLU) only yield
#   marginal differences under standard training setups.
#   Ref: Nair, V. & Hinton, G. E. (2010). Rectified linear units improve
#        restricted Boltzmann machines. ICML 2010.
#   Ref: Krizhevsky, A., Sutskever, I. & Hinton, G. E. (2012). ImageNet
#        classification with deep convolutional neural networks. NeurIPS 2012.
# FIXED_BN = True
#   Batch normalization is applied after every convolution. Ioffe & Szegedy
#   (2015) show consistent accuracy gains and faster convergence on image
#   classification; it is the de-facto default in ResNet, DenseNet,
#   WideResNet, EfficientNet and every NAS-for-CNN work on CIFAR since
#   2016. Known failure mode (very small batch sizes, <16) does not apply
#   here because the smallest batch in the search space is 32.
#   Ref: Ioffe, S. & Szegedy, C. (2015). Batch normalization: Accelerating
#        deep network training by reducing internal covariate shift. ICML
#        2015. https://arxiv.org/abs/1502.03167
#   Ref: Santurkar, S. et al. (2018). How does batch normalization help
#        optimization? NeurIPS 2018. https://arxiv.org/abs/1805.11604
FIXED_LR         = 1e-3
FIXED_DROPOUT    = 0.5
FIXED_ACTIVATION = 'relu'
FIXED_BN         = True

BOUNDS = [
    IntegerVar(lb=0,    ub=2   ),   # [0]  batch_idx      — 3 options : [32, 64, 128]
    IntegerVar(lb=0,    ub=2   ),   # [1]  filters_idx    — 3 options : [32, 64, 128] (base; doubles per block)

    # [2] n_blocks_idx — number of VGG-style conv blocks. Range [2, 3]:
    # each block halves the spatial resolution via 2×2 pool. On 32×32 CIFAR,
    # 2 blocks leave an 8×8 feature map, 3 blocks a 4×4 map. The 4-block
    # option was removed because (i) on 32×32 inputs it produces a 2×2
    # map that rarely outperforms 3 blocks, (ii) combined with
    # base_filters=128 and kernel=5 it was the main driver of GPU OOMs
    # and host-RAM leaks in early runs, and (iii) removing it shrinks the
    # search space from 648 to 432 discrete configurations, letting the
    # metaheuristic's evaluation budget concentrate where the topology is
    # most informative. Lower bound 2 remains the shallowest useful stack.
    # Introducing n_blocks into the search makes this template-based HPO
    # with partial topology search — see Feurer & Hutter (2019, ch. 1) for
    # the terminology discussion.
    # Ref: Feurer, M. & Hutter, F. (2019). Hyperparameter optimization.
    #      In: Automated Machine Learning, ch. 1. Springer.
    IntegerVar(lb=0,    ub=1   ),   # [2]  n_blocks_idx   — 2 options : [2, 3]
    IntegerVar(lb=0,    ub=2   ),   # [3]  dense_idx      — 3 options : [128, 256, 512]
    IntegerVar(lb=0,    ub=2   ),   # [4]  optimizer_idx  — 3 options : adamw, sgd, rmsprop
    IntegerVar(lb=0,    ub=1   ),   # [5]  pooling_idx    — 2 options : max, average
    IntegerVar(lb=0,    ub=1   ),   # [6]  kernel_idx     — 2 options : 3, 5
]

# Discrete options
# ─── BATCH_SIZES ──────────────────────────────────────────────────────────────
# Powers of 2 for GPU memory alignment efficiency. Range [32, 128] follows
# Keskar et al. (2017), who show that large batches (>512) converge to sharp
# minima with worse generalisation, while very small batches (<16) increase
# training time without proportional benefit.
# Ref: Keskar, N. S. et al. (2017). On large-batch training for deep learning.
#      ICLR 2017. https://arxiv.org/abs/1609.04836
BATCH_SIZES   = [32, 64, 128]

# ─── FILTER_OPTS ──────────────────────────────────────────────────────────────
# Powers of 2 for GPU efficiency. Lower bound of 32 ensures sufficient
# representational capacity for CIFAR-10 (32×32×3). Upper bound of 128
# (reaching 512 in the 3rd block after doubling) follows VGG proportions
# scaled down for small images.
# Ref: Simonyan, K. & Zisserman, A. (2015). Very deep convolutional networks
#      for large-scale image recognition. ICLR 2015.
#      https://arxiv.org/abs/1409.1556
FILTER_OPTS   = [32, 64, 128]

# ─── DENSE_OPTS ───────────────────────────────────────────────────────────────
# Range [128, 512] covers the typical dense head sizes used in NAS studies on
# CIFAR-10. Below 128 creates a representational bottleneck; above 512 adds
# parameters without measurable accuracy gain and increases overfitting risk
# on small datasets.
# Ref: Elsken, T., Metzen, J. H. & Hutter, F. (2019). Neural architecture
#      search: A survey. JMLR 20(55). https://arxiv.org/abs/1808.05377
DENSE_OPTS    = [128, 256, 512]

OPTIMIZERS    = ['adamw', 'sgd', 'rmsprop']
POOLING_TYPES = ['max', 'average']

# ─── KERNEL_SIZES ─────────────────────────────────────────────────────────────
# 3×3 is the dominant choice following VGG (Simonyan & Zisserman, 2015), which
# showed it captures the same receptive field as larger kernels with fewer
# parameters and more non-linearities. 5×5 is retained as an alternative for
# capturing broader spatial patterns. Larger kernels (7×7, 11×11) are excluded
# as they reduce spatial resolution too aggressively on 32×32 CIFAR-10 images.
# Ref: Simonyan, K. & Zisserman, A. (2015). Very deep convolutional networks
#      for large-scale image recognition. ICLR 2015.
#      https://arxiv.org/abs/1409.1556
KERNEL_SIZES  = [3, 5]

# ─── N_BLOCKS_OPTS ────────────────────────────────────────────────────────────
# Number of conv blocks. Each block applies a 2×2 pool, so depth is
# bounded by the spatial resolution of the input. On 32×32 CIFAR images:
#   2 blocks -> 8×8 feature map before flatten
#   3 blocks -> 4×4  (deepest option retained)
# The 4-block option (2×2 map) was removed: on small inputs the extra
# pooling stage rarely improves accuracy over 3 blocks, while the largest
# candidate configurations (base_filters=128 × kernel=5 × 4 blocks)
# caused GPU OOM and host-RAM pressure disproportionate to their value.
N_BLOCKS_OPTS = [2, 3]


def decodeSolution(solution):
    s = solution
    batch_size    = BATCH_SIZES   [int(s[0])]
    base_filters  = FILTER_OPTS   [int(s[1])]
    n_blocks      = N_BLOCKS_OPTS [int(s[2])]
    dense_units   = DENSE_OPTS    [int(s[3])]
    optimizer     = OPTIMIZERS    [int(s[4])]
    pooling       = POOLING_TYPES [int(s[5])]
    kernel_size   = KERNEL_SIZES  [int(s[6])]

    return {
        # Fixed hyperparameters (see FIXED_* constants above).
        'learning_rate': FIXED_LR,
        'dropout_rate':  FIXED_DROPOUT,
        'activation':    FIXED_ACTIVATION,
        'use_bn':        FIXED_BN,
        # Tuned by the metaheuristic.
        'batch_size':    batch_size,
        'base_filters':  base_filters,
        'n_blocks':      n_blocks,
        'dense_units':   dense_units,
        'optimizer':     optimizer,
        'pooling':       pooling,
        'kernel_size':   kernel_size,
    }


_eval_counter  = {'n': 0, 'best': 0.0, 'start': None}
_eval_log      = []   # history of all evaluations


def makeObjective(data, eval_epochs=5):
    def objective(solution):
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
            # OOM: config exceeds GPU memory (typically base_filters=128 +
            # n_blocks=4 + kernel=5). Assign worst-case fitness so the
            # metaheuristic learns to avoid this region, and clean up now
            # — waiting for the per-generation cleanup would cascade failures.
            print(f"  [OOM  ] eval={_eval_counter['n']+1} — config exceeds GPU memory; penalised (val_acc=0)")
            val_acc = 0.0
            oom = True
        except Exception as e:
            print(f"  [ERROR] eval={_eval_counter['n']+1} — {type(e).__name__}: {e}")
            val_acc = 0.0

        # ── Aggressive per-evaluation cleanup ────────────────────────────
        # Three leak sources are tackled in order:
        #   (i)   optimizer state holds references to weight/grad tensors and
        #         to tf.function traces via its `apply_gradients` path;
        #   (ii)  the model itself still owns layer weights and compiled
        #         functions until its refcount hits zero;
        #   (iii) the backend session caches tf.function traces (one per
        #         unique input signature) that clear_session() empties only
        #         if all referencing objects are gone first.
        # Additionally we run gc.collect() twice: the first pass breaks the
        # Keras<->optimizer reference cycles, the second actually reclaims.
        try:
            if model is not None and hasattr(model, 'optimizer'):
                # Drop the optimizer FIRST so its slot-variables (Adam's
                # m, v tensors) release their refs before model teardown.
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
        gc.collect()   # second pass: reclaims objects broken in first pass

        # Sample host RAM AFTER cleanup — this is the level that persists
        # between evals and is the signal for leak detection.
        ram_gb = _hostRamGB()

        fitness = 1.0 - val_acc

        # Logging
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
        # time of this specific evaluation (delta from previous)
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


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess-isolated objective (Path B: eliminates the host-RAM leak)
# ─────────────────────────────────────────────────────────────────────────────

def makeObjectiveWorker(
    data,
    eval_epochs = 1,
    timeout     = 180,
    base_seed   = 42,
):
    """
    Drop-in replacement for `makeObjective` that runs every evaluation
    in a freshly-spawned child process.

    Motivation
    ----------
    The in-process `makeObjective` suffers a host-RAM leak of roughly
    60–80 MB per evaluation on Kaggle kernels (TF `tf.function` trace
    cache, Keras compiled-function registry, CUDA workspace retention
    — none of which `keras.backend.clear_session()` fully reclaims).
    At 30 GB Kaggle RAM the kernel dies at ~eval 340.

    This worker-based objective delegates the model build + fit to a
    subprocess that exits (and thus releases *all* its memory) after
    each evaluation. The parent process — the one running mealpy —
    never imports TensorFlow and stays flat in RAM indefinitely.

    Cost
    ----
    Spawning a fresh Python process and re-importing TF costs roughly
    2–4 s per evaluation. For a 325-evaluation SADE run this adds
    ~15 min of wall-clock time; in exchange the run is guaranteed to
    complete regardless of budget.

    Logging
    -------
    Per-evaluation logging (`_eval_log`, `_eval_counter`, print line)
    runs in the parent as before, so existing downstream code — the
    CSV dump in `runOptimization`, the `ram_gb` column, best-so-far
    tracking — continues to work unchanged.

    A `status` column is added to the log with values:
        'ok'      — evaluation completed, `val_accuracy` is valid
        'oom'     — child hit `ResourceExhaustedError` (GPU OOM)
        'timeout' — child exceeded `timeout` seconds and was killed
        'error'   — child raised any other exception

    Parameters
    ----------
    data : dict
        As returned by `getDataset` (CIFAR-10 arrays + metadata).
    eval_epochs : int
        Training epochs per evaluation.
    timeout : int
        Hard wall-clock limit per child (seconds). Default 180.
    base_seed : int
        Base seed for the children. Child for evaluation N uses
        `base_seed + N` so weight initialisation is reproducible.

    Returns
    -------
    callable(solution) -> fitness
        Suitable to pass as the `obj_func` of a mealpy problem dict,
        identical contract to the `makeObjective` return value.
    """
    from code.utils.worker import prepareDataFile, evalWorker

    # Serialise the dataset once; every child reads from disk.
    # Linux's page cache makes repeated loads effectively free.
    data_paths = prepareDataFile(data)
    print(f"  [worker] dataset serialised to "
          f"{os.path.dirname(data_paths['x_train'])}")

    def objective(solution):
        config = decodeSolution(solution)

        # Per-evaluation seed for reproducible weight init across
        # reruns of the same mealpy trajectory.
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

        # ── Logging (parent-side; TF never touched here) ──────────────
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
    """
    Run a full HPO loop with the chosen metaheuristic.

    Seed handling
    -------------
    If ``seed`` is None (default) a fresh seed is drawn uniformly from
    [0, 2**31 - 1] before numpy/TF are seeded. The sampled seed is logged
    at the start of the run and persisted in the output JSON under
    ``run_results['config']['seed']`` so the run is always reproducible
    a posteriori. Pass an explicit integer to replay a specific run.

    Set ``use_worker=True`` to evaluate each solution in an isolated
    subprocess (via `makeObjectiveWorker`). This eliminates the host
    RAM leak that accumulates when TensorFlow/Keras state is reused
    across evaluations — at the cost of ~2-4 s of subprocess startup
    per evaluation. Recommended for runs with >200 evaluations on
    constrained environments (e.g. Kaggle, 30 GB RAM).

    ``worker_timeout`` is the hard wall-clock limit per child process
    (seconds). Configs exceeding it are terminated and logged with
    status='timeout'; only consulted when ``use_worker=True``.
    """
    # Draw a fresh seed if the caller did not specify one. Uses the OS
    # entropy source (numpy's default BitGenerator), not the global RNG,
    # so it cannot collide with seeds set earlier in the same interpreter.
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
    seed = int(seed)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Reset global log
    _eval_counter['n']     = 0
    _eval_counter['best']  = 0.0
    _eval_counter['start'] = time.time()
    _eval_log.clear()

    alg_name = algorithm.upper()
    # +1 for the population initialization phase (mealpy evaluates it before iterations)
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

    # Objective function — subprocess-isolated or in-process.
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

    # Problem for mealpy v3
    problem = {
        "obj_func": objective,
        "bounds":   BOUNDS,
        "minmax":   "min",
        "log_to":   None,   # disables mealpy's internal log
    }

    # Algorithm
    algo = getOptimizationAlgorithm(alg_name, n_generations, pop_size, algo_params)

    # Run
    t_start = time.time()
    g_best = algo.solve(problem, seed=seed)
    t_total = time.time() - t_start

    # Best solution (mealpy v3 API)
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

    # Save results
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

    # JSON
    json_path = os.path.join(results_dir, f'{alg_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    print()
    print(f"  Results saved : {json_path}")
    
    # CSV of evaluation log
    df_log = pd.DataFrame(_eval_log)
    csv_path = os.path.join(results_dir, f'{alg_name}_{ts}_evals.csv')
    df_log.to_csv(csv_path, index=False)
    print(f"  Eval log      : {csv_path}")
    print()

    return best_config, run_results


# ─────────────────────────────────────────────────────────────────────────────
# Compare multiple algorithms
# ─────────────────────────────────────────────────────────────────────────────

def compare_algorithms(
    algorithms,
    data,
    eval_epochs = 5,
    n_generations      = 15,
    pop_size    = 20,
    results_dir = 'results/optimization',
):
    """
    Runs multiple algorithms in sequence and compares results.

    Parameters:
        algorithms  (list) : List of names, e.g. ['PSO', 'GA', 'DE'].
        data        (dict) : Dict returned by get_dataset().
        eval_epochs (int)  : Epochs per evaluation. Default: 5.
        n_generations (int): Iterations per algorithm. Default: 15.
        pop_size    (int)  : Population per algorithm. Default: 20.
        results_dir (str)  : Results folder. Default: 'results/optimization'.

    Returns:
        list of dicts, sorted by val_accuracy descending.

    Example:
        results = compare_algorithms(['PSO', 'GA', 'DE'], data)
    """
    all_results = []
    for alg in algorithms:
        best_config, run_results = runOptimization(
            alg, data,
            eval_epochs=eval_epochs,
            n_generations=n_generations,
            pop_size=pop_size,
            results_dir=results_dir,
        )
        all_results.append(run_results)

    # Sort by val_accuracy
    all_results.sort(key=lambda r: r['best']['val_accuracy'], reverse=True)

    # Final summary
    print(f"\n{'='*62}")
    print(f"  ALGORITHM COMPARISON")
    print(f"{'='*62}")
    print(f"  {'Algorithm':<10}  {'Best val_acc':>12}  {'Time (min)':>10}  {'Evaluations':>11}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['algorithm']:<10}  "
              f"{r['best']['val_accuracy']:>12.4f}  "
              f"{r['total_time_s']/60:>10.1f}  "
              f"{r['n_evaluations']:>11}")
    print()

    # Comparison plot
    plot_algorithm_comparison(all_results, results_dir)

    return all_results