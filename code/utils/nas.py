"""
nas.py
------
Toolkit for interacting with NAS-Bench-201 (a.k.a. NATS-Bench topology search
space ``tss``) and running evolutionary algorithms over it.

NAS-Bench-201 (Dong & Yang, ICLR 2020) is a tabular benchmark of $5^6 = 15625$
CNN cell topologies, each pre-trained on CIFAR-10, CIFAR-100 and
ImageNet16-120. Because every architecture has been trained ahead of time,
algorithms can be benchmarked by looking up the test accuracy instead of
training real models. Reported wall-clock time is the *simulated* training
time of the architectures the algorithm queried, not the lookup time.

Search space
~~~~~~~~~~~~
Each architecture is a directed acyclic graph (DAG) with 4 nodes and 6 edges.
Each edge picks one of 5 primitive operations:

    0  : 'none'           (zeroize the input)
    1  : 'skip_connect'   (identity)
    2  : 'nor_conv_1x1'   (1x1 conv + BN + ReLU)
    3  : 'nor_conv_3x3'   (3x3 conv + BN + ReLU)
    4  : 'avg_pool_3x3'

An architecture is therefore encoded as a length-6 integer vector, with each
coordinate in ``[0, 4]``. This is the same encoding style used by the HPO
pipeline (see ``utils.optimizer``) so the same evolutionary algorithms can be
applied without modification.

The architecture string used internally by the NATS-Bench API has the form:

    |op0~0|+|op1~0|op2~1|+|op3~0|op4~1|op5~2|

where ``op_i = OPS[gene[i]]`` and ``~k`` is the source node of the edge.

Dependencies
~~~~~~~~~~~~
Install once:

    pip install nats_bench
    pip install mealpy           # already used by utils.optimizer

Then download the pre-computed benchmark file from the official mirror:
https://github.com/D-X-Y/NATS-Bench  ->  NATS-tss-v1_0-3ffb9-simple.tar
(uncompressed file ``NATS-tss-v1_0-3ffb9-simple/``, ~2 GB).

Set the environment variable ``NATS_BENCH_PATH`` to its location, or pass the
path explicitly to ``getApi()``.

Quick usage
~~~~~~~~~~~

    from utils.nas import (
        runNASSearch, runRandomSearchNAS, getApi,
        queryArchitecture, geneToArchStr,
    )

    api = getApi()                      # uses $NATS_BENCH_PATH
    best, run = runNASSearch(
        algorithm     = 'GA',
        api           = api,
        dataset       = 'cifar10',
        hp            = '200',
        n_generations = 10,
        pop_size      = 20,
        results_dir   = 'colab/results/nas/ga',
    )
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

OPS = [
    'none',           # 0  — zeroize the input
    'skip_connect',   # 1  — identity
    'nor_conv_1x1',   # 2  — 1x1 conv + BN + ReLU
    'nor_conv_3x3',   # 3  — 3x3 conv + BN + ReLU
    'avg_pool_3x3',   # 4  — 3x3 average pooling
]

N_EDGES = 6
N_OPS   = len(OPS)              # 5
N_ARCHS = N_OPS ** N_EDGES      # 15625

# Datasets exposed by NAS-Bench-201.
DATASETS = ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']

# Number-of-epochs settings (only '12' and '200' are exposed for all datasets).
HP_VALUES = ['12', '200']


# ─────────────────────────────────────────────────────────────────────────────
# Encoding: integer vector  <->  architecture string
# ─────────────────────────────────────────────────────────────────────────────

def geneToArchStr(gene):
    """Convert a length-6 integer vector to the NAS-Bench-201 string format.

    The 6 edges are ordered as in the official NATS-Bench encoding:
        edge 0:  node 1  <- node 0    (target = intermediate node 1)
        edge 1:  node 2  <- node 0
        edge 2:  node 2  <- node 1    (target = intermediate node 2)
        edge 3:  node 3  <- node 0
        edge 4:  node 3  <- node 1
        edge 5:  node 3  <- node 2    (target = output node 3)
    """
    g = [int(np.round(x)) for x in gene]
    if len(g) != N_EDGES:
        raise ValueError(f"gene must have length {N_EDGES}, got {len(g)}")
    for i, v in enumerate(g):
        if not (0 <= v < N_OPS):
            raise ValueError(f"gene[{i}]={v} out of range [0, {N_OPS-1}]")
    return (
        f"|{OPS[g[0]]}~0|+"
        f"|{OPS[g[1]]}~0|{OPS[g[2]]}~1|+"
        f"|{OPS[g[3]]}~0|{OPS[g[4]]}~1|{OPS[g[5]]}~2|"
    )


def archStrToGene(arch_str):
    """Inverse of :func:`geneToArchStr`. Returns a list of 6 ints."""
    op_index = {name: i for i, name in enumerate(OPS)}
    parts = arch_str.strip().split('+')
    if len(parts) != 3:
        raise ValueError(f"unexpected arch string: {arch_str}")
    gene = []
    for part in parts:
        tokens = [t for t in part.strip('|').split('|') if t]
        for t in tokens:
            op_name, _src = t.split('~')
            gene.append(op_index[op_name])
    if len(gene) != N_EDGES:
        raise ValueError(f"parsed {len(gene)} edges, expected {N_EDGES}")
    return gene


def randomGene(rng=None):
    """Sample a uniformly random length-6 gene."""
    rng = rng if rng is not None else np.random.default_rng()
    return [int(x) for x in rng.integers(0, N_OPS, size=N_EDGES)]


# ─────────────────────────────────────────────────────────────────────────────
# NATS-Bench API loader (cached)
# ─────────────────────────────────────────────────────────────────────────────

_API_CACHE = {}


def getApi(file_path=None, fast_mode=True, verbose=False):
    """Return a cached NATS-Bench API instance for the topology search space.

    Parameters
    ----------
    file_path : str, pathlib.Path or None
        Path to the NATS-Bench archive file (e.g. ``NATS-tss-v1_0-3ffb9.pickle.pbz2``)
        or to the uncompressed directory (e.g. ``NATS-tss-v1_0-3ffb9-simple/``).
        ``pathlib.Path`` instances are accepted and coerced to ``str``
        because the underlying ``nats_bench`` library only accepts strings.
        If ``None``, falls back to the ``NATS_BENCH_PATH`` environment
        variable. If that is also unset, the underlying library uses its
        own default lookup logic.
    fast_mode : bool
        Lazy loading mode that avoids materializing all results in memory.
    verbose : bool
        Library-level verbosity.
    """
    from nats_bench import create  # lazy import — module works without it

    raw = file_path if file_path is not None else os.environ.get('NATS_BENCH_PATH')
    # nats_bench validates with isinstance(..., str): coerce Path -> str.
    path = str(raw) if raw is not None else None

    key = (path, fast_mode)
    if key in _API_CACHE:
        return _API_CACHE[key]

    api = create(path, 'tss', fast_mode=fast_mode, verbose=verbose)
    _API_CACHE[key] = api
    return api


# ─────────────────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────────────────

def queryArchitecture(api, gene, dataset='cifar10', hp='200', is_random=False):
    """Query NAS-Bench-201 for one architecture.

    Parameters
    ----------
    api : object returned by :func:`getApi`.
    gene : sequence of 6 ints in [0, 4].
    dataset : one of :data:`DATASETS`.
    hp : '12' or '200' — number of training epochs of the lookup.
    is_random : if False (default) returns the average across the available
        training seeds; if True picks one at random; if an int, picks that
        specific seed.

    Returns
    -------
    dict with the following keys:

        index, arch_str,
        train_acc, valid_acc, test_acc        (in [0, 1])
        train_loss, valid_loss, test_loss
        train_all_time, train_per_time
        params_MB, flops_M, latency           (architecture cost)
    """
    arch_str = geneToArchStr(gene)
    index    = api.query_index_by_arch(arch_str)
    info     = api.get_more_info(index, dataset, hp=hp, is_random=is_random)
    cost     = api.get_cost_info(index, dataset, hp=hp)

    def _pct(x):
        return None if x is None else float(x) / 100.0

    return {
        'index':           int(index),
        'arch_str':        arch_str,
        'train_acc':       _pct(info.get('train-accuracy')),
        'valid_acc':       _pct(info.get('valid-accuracy')),
        'test_acc':        _pct(info.get('test-accuracy')),
        'train_loss':      info.get('train-loss'),
        'valid_loss':      info.get('valid-loss'),
        'test_loss':       info.get('test-loss'),
        'train_all_time':  info.get('train-all-time'),
        'train_per_time':  info.get('train-per-time'),
        'params_MB':       cost.get('params'),
        'flops_M':         cost.get('flops'),
        'latency':         cost.get('latency'),
    }



# ─────────────────────────────────────────────────────────────────────────────
# Validation / test split
# ─────────────────────────────────────────────────────────────────────────────

# NAS-Bench-201 exposes a dedicated 'cifar10-valid' entry (trained on 25k,
# scored on the held-out 25k). The plain 'cifar10' entry is trained on the full
# 50k and has NO honest validation split — its 'valid-accuracy' is None, so
# searching on r['valid_acc'] there would collapse every architecture to 0.0
# and flatten the fitness landscape. CIFAR-100 and ImageNet16-120 carry both
# splits in the same entry.
_SEARCH_DATASET = {
    'cifar10':        'cifar10-valid',
    'cifar10-valid':  'cifar10-valid',
    'cifar100':       'cifar100',
    'ImageNet16-120': 'ImageNet16-120',
}
_REPORT_DATASET = {
    'cifar10':        'cifar10',
    'cifar10-valid':  'cifar10',
    'cifar100':       'cifar100',
    'ImageNet16-120': 'ImageNet16-120',
}


# Memoized benchmark lookups.
#
# NAS-Bench-201 is a static table, so re-querying an architecture returns the
# identical result. The NATS-Bench API caches every architecture it loads and
# never evicts, so a long sweep (hundreds of runs x 1020 evaluations) grows
# without bound until the machine runs out of memory. Memoizing here means each
# architecture is fetched from the API at most once per (dataset, hp), after
# which it is evicted from the API's own store (see _releaseApiParams).
#
# This is purely a read cache: it does not touch the evaluation counters, the
# eval log, or n_unique_architectures. A run produces exactly the numbers it
# would produce without it.
_QUERY_CACHE = {}


def clearQueryCache():
    """Drop the memoized lookups (e.g. between datasets, to cap memory)."""
    n = len(_QUERY_CACHE)
    _QUERY_CACHE.clear()
    return n


# Whether to drop each architecture from the NATS-Bench store once we have
# copied what we need. Set False (setApiEviction(False)) to fall back to the
# library's own behaviour.
_EVICT_API = True
_EVICT_WARNED = False


def setApiEviction(enabled):
    """Turn eviction of the NATS-Bench in-memory store on or off."""
    global _EVICT_API
    _EVICT_API = bool(enabled)
    return _EVICT_API


def _releaseApiParams(api, index, hp):
    """Drop an architecture from the NATS-Bench in-memory store.

    Measured on NATS-tss-v1_0: each architecture the API loads costs ~1.8 MiB
    and is never released, so ``arch2infos_dict`` grows monotonically. Touching
    the whole 15,625-cell space would need roughly 28 GB — which is what
    exhausted the machine during a full sweep.

    ``clear_params`` only frees network weights, not the per-epoch metric
    records (``ArchResults`` / ``ResultsCount``) that actually dominate, so the
    entry itself has to go. This is safe because :func:`queryValidTest` has
    already copied everything we need into ``_QUERY_CACHE``, and NATS-Bench
    reloads an evicted index on demand.

    ``arch2infos_dict`` and ``evaluated_indexes`` are always updated together;
    leaving one populated and the other not would make the API believe an
    architecture is loaded when it is not.
    """
    global _EVICT_WARNED
    if not _EVICT_API or index is None:
        return

    # Free the weights too, when the version exposes it. Cheap, and harmless
    # if the entry is about to be dropped anyway.
    fn = getattr(api, 'clear_params', None)
    if fn is not None:
        try:
            fn(index, hp)
        except TypeError:
            try:
                fn(index)
            except Exception:
                pass
        except Exception:
            pass

    store = getattr(api, 'arch2infos_dict', None)
    if not isinstance(store, dict):
        if not _EVICT_WARNED:
            _EVICT_WARNED = True
            print('[utils.nas] AVISO: nao encontrei arch2infos_dict na API; '
                  'a memoria vai crescer ~1.8 MiB por arquitectura. '
                  'Corre diagnose_memory.ipynb para identificar o atributo.')
        return

    store.pop(index, None)
    seen = getattr(api, 'evaluated_indexes', None)
    if isinstance(seen, set):
        seen.discard(index)
    elif isinstance(seen, list):
        try:
            seen.remove(index)
        except ValueError:
            pass


def queryValidTest(api, gene, dataset='cifar10', hp='200'):
    """Query one architecture, returning both accuracies.

    A superset of :func:`queryArchitecture`'s dict: every cost field comes from
    the reporting entry, ``valid_acc`` from the search entry. ``valid_acc``
    drives the search; ``test_acc`` is read only for the architecture a run
    finally returns.

    Results are memoized (see :data:`_QUERY_CACHE`); a copy is returned so a
    caller cannot corrupt the cache.
    """
    key = (tuple(int(g) for g in gene), dataset, hp)
    hit = _QUERY_CACHE.get(key)
    if hit is not None:
        return dict(hit)

    s_ds = _SEARCH_DATASET.get(dataset, dataset)
    r_ds = _REPORT_DATASET.get(dataset, dataset)
    sv = queryArchitecture(api, gene, dataset=s_ds, hp=hp)
    tv = sv if r_ds == s_ds else queryArchitecture(api, gene, dataset=r_ds, hp=hp)
    out = dict(tv)
    out['valid_acc'] = sv.get('valid_acc')
    out['test_acc'] = tv.get('test_acc')

    _QUERY_CACHE[key] = out
    # Everything needed is now held locally, so the API copy is dead weight.
    _releaseApiParams(api, out.get('index'), hp)
    return dict(out)


# ─────────────────────────────────────────────────────────────────────────────
# Objective for mealpy-style optimizers
# ─────────────────────────────────────────────────────────────────────────────

def makeNASObjective(
    api,
    dataset      = 'cifar10',
    hp           = '200',
    fitness_key  = 'test_acc',
    eval_log     = None,
    counter      = None,
    verbose      = True,
):
    """Build an objective compatible with :mod:`mealpy`.

    The objective receives a length-6 array (mealpy solution), runs a query
    against NAS-Bench-201, appends one row to ``eval_log`` and returns the
    fitness ``1 - <fitness_key>`` (lower is better, like
    ``utils.optimizer.makeObjective``).

    Parameters
    ----------
    api, dataset, hp : see :func:`queryArchitecture`.
    fitness_key : which accuracy to minimize ``1 - x`` of. Common choices:

        - ``'test_acc'``  : honest reporting (what we want to *eventually*
          maximize). Note that scoring the search with test accuracy is the
          standard NAS-Bench-201 protocol; the goal is to compare *search
          algorithms*, not to deploy a model.
        - ``'valid_acc'`` : closer to how a practitioner would tune in
          practice (use cifar10-valid or the valid accuracy on cifar100 /
          ImageNet16-120).

    eval_log : list, will be appended to in place. Created if ``None``.
    counter : dict with keys ``{n, best, start}``. Created if ``None``.
    verbose : print one line per evaluation.

    Returns
    -------
    objective(solution) -> float
    """
    if eval_log is None:
        eval_log = []
    if counter is None:
        counter = {'n': 0, 'best': 0.0, 'start': time.time()}

    def objective(solution):
        gene = [int(np.round(x)) for x in solution]
        try:
            r = queryValidTest(api, gene, dataset=dataset, hp=hp)
            acc = r[fitness_key] or 0.0   # fitness_key='valid_acc' => no leakage
        except Exception as e:
            print(f"  [ERROR] query failed at eval {counter['n']+1}: "
                  f"{type(e).__name__}: {e}")
            r   = {'index': -1, 'arch_str': '?',
                   'train_acc': 0.0, 'valid_acc': 0.0, 'test_acc': 0.0,
                   'params_MB': None, 'flops_M': None, 'latency': None,
                   'train_all_time': 0.0}
            acc = 0.0

        fitness = 1.0 - acc
        counter['n'] += 1
        if acc > counter['best']:
            counter['best'] = acc
        elapsed = time.time() - counter['start']

        eval_log.append({
            'eval':           counter['n'],
            'arch_index':     r['index'],
            'arch_str':       r['arch_str'],
            'gene':           ' '.join(str(g) for g in gene),
            'train_acc':      r['train_acc'],
            'valid_acc':      r['valid_acc'],
            'test_acc':       r['test_acc'],
            'fitness':        round(fitness, 6),
            'params_MB':      r['params_MB'],
            'flops_M':        r['flops_M'],
            'latency':        r['latency'],
            'train_time_s':   r['train_all_time'],
            'elapsed_s':      round(elapsed, 3),
        })

        if verbose:
            params_s = f"{r['params_MB']:.2f}MB" if r['params_MB'] is not None else 'n/a'
            flops_s  = f"{r['flops_M']:.1f}M"   if r['flops_M']   is not None else 'n/a'
            t_train  = f"{r['train_all_time']:.0f}s" if r['train_all_time'] is not None else 'n/a'
            print(f"  [{counter['n']:>4}] gene={gene}  test_acc={acc:.4f}  "
                  f"best={counter['best']:.4f}  params={params_s}  "
                  f"flops={flops_s}  train_t={t_train}")

        return fitness

    objective.eval_log = eval_log
    objective.counter  = counter
    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Full search runners (mirror utils.optimizer.runOptimization)
# ─────────────────────────────────────────────────────────────────────────────

def _mealpyBounds():
    """Return mealpy bounds for the 6-edge integer search space."""
    from mealpy import IntegerVar
    return [IntegerVar(lb=0, ub=N_OPS - 1, name=f'edge_{i}') for i in range(N_EDGES)]


def runNASSearch(
    algorithm,
    api,
    dataset       = 'cifar10',
    hp            = '200',
    n_generations = 10,
    pop_size      = 20,
    fitness_key   = 'test_acc',
    seed          = None,
    algo_params   = None,
    results_dir   = 'results/nas',
    verbose       = True,
):
    """Run a full NAS search with one of the mealpy evolutionary algorithms.

    Mirrors :func:`utils.optimizer.runOptimization` but queries NAS-Bench-201
    instead of training a real CNN. The output JSON / CSV schema is similar
    to the EA-HPO runs so the existing aggregation scripts work with minimal
    changes (the search-space columns differ).

    Parameters
    ----------
    algorithm : str
        Name passed to ``utils.localMealpy.getOptimizationAlgorithm``
        ('GA', 'SADE', 'PSO', 'GWO', ...).
    api : object from :func:`getApi`.
    dataset : one of :data:`DATASETS`.
    hp : '12' or '200'.
    n_generations, pop_size : standard EA budget.
    fitness_key : see :func:`makeNASObjective`.
    seed : int or None (drawn from OS entropy if None).
    algo_params : dict of algorithm-specific hyperparameters.
    results_dir : where to save the JSON + CSV.
    verbose : per-evaluation logging.
    """
    from utils.localMealpy import getOptimizationAlgorithm

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    np.random.seed(seed)

    alg_name    = algorithm.upper()
    total_evals = (n_generations + 1) * pop_size

    print(f"\n{'='*62}")
    print(f"  NAS SEARCH: {alg_name}  (NAS-Bench-201, {dataset}, hp={hp})")
    print(f"  Generations: {n_generations}  |  Population: {pop_size}  |  "
          f"Total queries: {total_evals}")
    print(f"  Fitness key: {fitness_key}")
    print(f"  Seed:        {seed}")
    print(f"{'='*62}\n")

    eval_log = []
    counter  = {'n': 0, 'best': 0.0, 'start': time.time()}
    objective = makeNASObjective(
        api, dataset=dataset, hp=hp, fitness_key=fitness_key,
        eval_log=eval_log, counter=counter, verbose=verbose,
    )

    problem = {
        'obj_func': objective,
        'bounds':   _mealpyBounds(),
        'minmax':   'min',
        'log_to':   None,
    }
    algo = getOptimizationAlgorithm(alg_name, n_generations, pop_size, algo_params)

    t0 = time.time()
    g_best = algo.solve(problem, seed=seed)
    t_total = time.time() - t0

    best_gene    = [int(np.round(x)) for x in g_best.solution]
    best_fitness = float(g_best.target.fitness)
    # Re-query rather than inverting the fitness: with fitness_key='valid_acc'
    # 1 - fitness is the VALIDATION accuracy, and reporting it as test_acc
    # would mislabel the result.
    best_query    = queryValidTest(api, best_gene, dataset=dataset, hp=hp)
    best_valid_acc = best_query['valid_acc'] or 0.0
    best_test_acc  = best_query['test_acc'] or 0.0

    if verbose:
        print(f"\n{'='*62}")
        print(f"  FINAL RESULT - {alg_name}  (NAS-Bench-201, {dataset})")
        print(f"{'='*62}")
        print(f"  Best test_acc     : {best_test_acc:.4f}  ({best_test_acc*100:.2f}%)")
        print(f"  Best valid_acc    : {best_query['valid_acc']}")
        print(f"  Best arch index   : {best_query['index']}")
        print(f"  Best arch str     : {best_query['arch_str']}")
        print(f"  Best params       : {best_query['params_MB']} MB  "
              f"flops={best_query['flops_M']} M  "
              f"latency={best_query['latency']}")
        print(f"  Total wallclock   : {t_total:.1f}s")
        print(f"  Simulated GPU time: {sum((e['train_time_s'] or 0) for e in eval_log):.0f}s")
        print(f"  Evaluations done  : {counter['n']}")
        print()

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_results = {
        'algorithm':    alg_name,
        'benchmark':    'NAS-Bench-201',
        'timestamp':    datetime.now().isoformat(),
        'config': {
            'dataset':       dataset,
            'hp':            hp,
            'fitness_key':   fitness_key,
            'n_generations': n_generations,
            'pop_size':      pop_size,
            'seed':          seed,
        },
        'best': {
            'gene':         best_gene,
            'arch_str':     best_query['arch_str'],
            'arch_index':   best_query['index'],
            'test_acc':     best_test_acc,
            'valid_acc':    best_query['valid_acc'],
            'fitness':      round(best_fitness, 6),
            'params_MB':    best_query['params_MB'],
            'flops_M':      best_query['flops_M'],
            'latency':      best_query['latency'],
        },
        'total_time_s':            round(t_total, 1),
        'simulated_train_time_s':  round(sum((e['train_time_s'] or 0) for e in eval_log), 1),
        'n_evaluations':           counter['n'],
        'n_unique_architectures':  len({e['arch_index'] for e in eval_log}),
        'eval_log':                eval_log.copy(),
    }

    json_path = os.path.join(results_dir, f'{alg_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    print(f"  Results saved : {json_path}")

    df = pd.DataFrame(eval_log)
    csv_path = os.path.join(results_dir, f'{alg_name}_{ts}_evals.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Eval log      : {csv_path}\n")

    return run_results['best'], run_results


def runRandomSearchNAS(
    api,
    dataset       = 'cifar10',
    hp            = '200',
    n_queries     = 220,
    fitness_key   = 'test_acc',
    seed          = None,
    results_dir   = 'results/nas/random',
    verbose       = True,
):
    """Random Search baseline over NAS-Bench-201.

    Samples ``n_queries`` architectures uniformly at random and returns the
    best one observed, using the same JSON / CSV output schema as
    :func:`runNASSearch`.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    rng = np.random.default_rng(seed)

    alg_name = 'RANDOM'
    print(f"\n{'='*62}")
    print(f"  NAS SEARCH: {alg_name}  (NAS-Bench-201, {dataset}, hp={hp})")
    print(f"  Queries: {n_queries}  |  Seed: {seed}")
    print(f"{'='*62}\n")

    eval_log = []
    counter  = {'n': 0, 'best': 0.0, 'start': time.time()}
    objective = makeNASObjective(
        api, dataset=dataset, hp=hp, fitness_key=fitness_key,
        eval_log=eval_log, counter=counter, verbose=verbose,
    )

    t0 = time.time()
    for _ in range(n_queries):
        gene = randomGene(rng)
        objective(gene)
    t_total = time.time() - t0

    best_row  = min(eval_log, key=lambda e: e['fitness'])
    best_gene = [int(x) for x in best_row['gene'].split()]

    if verbose:
        print(f"\n{'='*62}")
        print(f"  FINAL RESULT - {alg_name}  (NAS-Bench-201, {dataset})")
        print(f"{'='*62}")
        print(f"  Best test_acc     : {best_row['test_acc']:.4f}")
        print(f"  Best arch index   : {best_row['arch_index']}")
        print(f"  Best arch str     : {best_row['arch_str']}")
        print(f"  Total wallclock   : {t_total:.1f}s")
        print(f"  Simulated GPU time: {sum((e['train_time_s'] or 0) for e in eval_log):.0f}s")
        print()

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_results = {
        'algorithm':    alg_name,
        'benchmark':    'NAS-Bench-201',
        'timestamp':    datetime.now().isoformat(),
        'config': {
            'dataset':     dataset,
            'hp':          hp,
            'fitness_key': fitness_key,
            'n_queries':   n_queries,
            'seed':        seed,
        },
        'best': {
            'gene':         best_gene,
            'arch_str':     best_row['arch_str'],
            'arch_index':   best_row['arch_index'],
            'test_acc':     best_row['test_acc'],
            'valid_acc':    best_row['valid_acc'],
            'fitness':      best_row['fitness'],
            'params_MB':    best_row['params_MB'],
            'flops_M':      best_row['flops_M'],
            'latency':      best_row['latency'],
        },
        'total_time_s':            round(t_total, 1),
        'simulated_train_time_s':  round(sum((e['train_time_s'] or 0) for e in eval_log), 1),
        'n_evaluations':           counter['n'],
        'n_unique_architectures':  len({e['arch_index'] for e in eval_log}),
        'eval_log':                eval_log.copy(),
    }

    json_path = os.path.join(results_dir, f'{alg_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    print(f"  Results saved : {json_path}")

    df = pd.DataFrame(eval_log)
    csv_path = os.path.join(results_dir, f'{alg_name}_{ts}_evals.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Eval log      : {csv_path}\n")

    return run_results['best'], run_results


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Optimization baseline (GP + Expected Improvement) over NAS-Bench-201
# ─────────────────────────────────────────────────────────────────────────────

def runBONASSearch(
    api,
    dataset      = 'cifar10',
    hp           = '200',
    n_iter       = 200,
    init_points  = 20,
    xi           = 0.01,
    fitness_key  = 'test_acc',
    seed         = None,
    results_dir  = 'results/nas/bo',
    verbose      = True,
):
    """Bayesian-Optimization baseline over NAS-Bench-201.

    Uses Gaussian Processes with the Expected Improvement acquisition
    function (Fernando Nogueira's ``bayesian-optimization`` package), with
    integer-aware bounds matching the 6-edge / 5-ops NAS-Bench-201 search
    space. The total query budget is ``init_points + n_iter`` (default
    ``20 + 200 = 220``, matching the EA budget).

    Mirrors :func:`runRandomSearchNAS` for I/O purposes — same JSON / CSV
    output schema, same logging — but is a much stronger baseline because
    the GP exploits the structure of the observed fitness landscape.
    """
    from bayes_opt import BayesianOptimization, acquisition

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    np.random.seed(seed)

    alg_name    = 'BO'
    total_evals = init_points + n_iter

    print(f"\n{'='*62}")
    print(f"  NAS SEARCH: {alg_name}  (NAS-Bench-201, {dataset}, hp={hp})")
    print(f"  Total queries: {total_evals}  "
          f"(= {init_points} random init + {n_iter} BO-guided)")
    print(f"  Acquisition: ExpectedImprovement(xi={xi})")
    print(f"  Fitness key: {fitness_key}")
    print(f"  Seed:        {seed}")
    print(f"{'='*62}\n")

    eval_log = []
    counter  = {'n': 0, 'best': 0.0, 'start': time.time()}
    mealpy_obj = makeNASObjective(
        api, dataset=dataset, hp=hp, fitness_key=fitness_key,
        eval_log=eval_log, counter=counter, verbose=verbose,
    )

    # The bayes_opt library is a maximizer and takes keyword args, so we
    # wrap the mealpy-style objective (which receives a vector and returns
    # ``1 - acc``).
    def bo_objective(e0, e1, e2, e3, e4, e5):
        fitness = mealpy_obj([e0, e1, e2, e3, e4, e5])
        return 1.0 - fitness  # maximize the accuracy

    pbounds = {f'e{i}': (0, N_OPS - 1, int) for i in range(N_EDGES)}
    acq = acquisition.ExpectedImprovement(xi=xi, random_state=seed)
    bo = BayesianOptimization(
        f                    = bo_objective,
        pbounds              = pbounds,
        acquisition_function = acq,
        random_state         = seed,
        verbose              = 0,
    )

    t0 = time.time()
    bo.maximize(init_points=init_points, n_iter=n_iter)
    t_total = time.time() - t0

    best_params   = bo.max['params']
    best_gene     = [int(np.round(best_params[f'e{i}'])) for i in range(N_EDGES)]
    best_query     = queryValidTest(api, best_gene, dataset=dataset, hp=hp)
    best_valid_acc = best_query['valid_acc'] or 0.0
    best_test_acc  = best_query['test_acc'] or 0.0
    best_fitness   = 1.0 - best_valid_acc

    if verbose:
        print(f"\n{'='*62}")
        print(f"  FINAL RESULT - {alg_name}  (NAS-Bench-201, {dataset})")
        print(f"{'='*62}")
        print(f"  Best test_acc     : {best_test_acc:.4f}  ({best_test_acc*100:.2f}%)")
        print(f"  Best valid_acc    : {best_query['valid_acc']}")
        print(f"  Best arch index   : {best_query['index']}")
        print(f"  Best arch str     : {best_query['arch_str']}")
        print(f"  Best params       : {best_query['params_MB']} MB  "
              f"flops={best_query['flops_M']} M  "
              f"latency={best_query['latency']}")
        print(f"  Total wallclock   : {t_total:.1f}s")
        print(f"  Simulated GPU time: {sum((e['train_time_s'] or 0) for e in eval_log):.0f}s")
        print()

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_results = {
        'algorithm':    alg_name,
        'benchmark':    'NAS-Bench-201',
        'timestamp':    datetime.now().isoformat(),
        'config': {
            'dataset':     dataset,
            'hp':          hp,
            'fitness_key': fitness_key,
            'init_points': init_points,
            'n_iter':      n_iter,
            'xi':          xi,
            'seed':        seed,
        },
        'best': {
            'gene':         best_gene,
            'arch_str':     best_query['arch_str'],
            'arch_index':   best_query['index'],
            'test_acc':     best_test_acc,
            'valid_acc':    best_query['valid_acc'],
            'fitness':      round(best_fitness, 6),
            'params_MB':    best_query['params_MB'],
            'flops_M':      best_query['flops_M'],
            'latency':      best_query['latency'],
        },
        'total_time_s':            round(t_total, 1),
        'simulated_train_time_s':  round(sum((e['train_time_s'] or 0) for e in eval_log), 1),
        'n_evaluations':           counter['n'],
        'n_unique_architectures':  len({e['arch_index'] for e in eval_log}),
        'eval_log':                eval_log.copy(),
    }

    json_path = os.path.join(results_dir, f'{alg_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    print(f"  Results saved : {json_path}")

    df = pd.DataFrame(eval_log)
    csv_path = os.path.join(results_dir, f'{alg_name}_{ts}_evals.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Eval log      : {csv_path}\n")

    return run_results['best'], run_results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers for analysis / reporting
# ─────────────────────────────────────────────────────────────────────────────

def benchmarkUpperBound(api, dataset='cifar10', hp='200',
                        fitness_key='test_acc', front_dir=None, verbose=True):
    """Return the (architecture, accuracy) of the globally best architecture
    in NAS-Bench-201 for the chosen dataset.

    This is the *gold standard* against which any search algorithm should be
    compared.

    Two routes, in order:

    1. ``front_dir/true_front_<dataset>.json`` — the exhaustive Pareto front, if
       it has already been computed. The most accurate architecture is
       non-dominated by construction (nothing beats it on accuracy), so it is
       always on that front: reading the maximum from it gives exactly the same
       answer as enumerating, instantly and at no memory cost.
    2. Enumerating all 15,625 cells. Each architecture the API loads costs
       ~1.8 MiB and NATS-Bench never evicts, so a bare loop would retain some
       27 GB; every index is therefore released as soon as it has been read.
    """
    info_key = 'test-accuracy' if fitness_key == 'test_acc' else 'valid-accuracy'

    if front_dir is not None and fitness_key == 'test_acc':
        path = os.path.join(str(front_dir), f'true_front_{dataset}.json')
        if os.path.exists(path):
            with open(path) as fh:
                front = json.load(fh)
            if front:
                top = max(front, key=lambda p: p.get('test_acc') or 0.0)
                best = {'index': top.get('index'),
                        'arch_str': top.get('arch_str'),
                        fitness_key: float(top.get('test_acc') or 0.0)}
                if verbose:
                    print(f"NAS-Bench-201 upper bound on {dataset} "
                          f"(hp={hp}, {fitness_key}), from {os.path.basename(path)}:")
                    print(f"  index    : {best['index']}")
                    print(f"  arch_str : {best['arch_str']}")
                    print(f"  {fitness_key}: {best[fitness_key]:.4f}")
                return best

    best = {'index': None, 'arch_str': None, fitness_key: -1.0}
    for index in range(N_ARCHS):
        info = api.get_more_info(index, dataset, hp=hp, is_random=False)
        acc  = (info.get(info_key) or 0.0) / 100.0
        if acc > best[fitness_key]:
            best.update({
                'index':       index,
                'arch_str':    api.arch(index),
                fitness_key:   acc,
            })
        # Without this the loop retains every architecture it touches.
        _releaseApiParams(api, index, hp)
    if verbose:
        print(f"NAS-Bench-201 upper bound on {dataset} (hp={hp}, {fitness_key}):")
        print(f"  index    : {best['index']}")
        print(f"  arch_str : {best['arch_str']}")
        print(f"  {fitness_key}: {best[fitness_key]:.4f}")
    return best
