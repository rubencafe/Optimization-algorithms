"""
nas_mo.py
---------
Multi-objective NAS over NAS-Bench-201 (Study 3).

Extends Study 2 (nas.py) by optimising two objectives simultaneously:

    f1 = 1 - test_acc          (minimise  →  maximise accuracy)
    f2 = normalised FLOPs      (minimise  →  fewer operations)

FLOPs are normalised to [0, 1] using the min/max across all 15 625
architectures of the benchmark, computed once per (dataset, hp) pair via
:func:`computeFlopsRange`.

Search strategies are implemented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Native multi-objective (one non-dominated front per seed, no alpha sweep):

1. **NSGA-II** (pymoo) — Pareto-based multi-objective genetic algorithm.
   Use :func:`runNSGAIINASSearch`.

2. **GDE3** (pymoo) — Generalised Differential Evolution 3, the native
   multi-objective extension of Differential Evolution. Use
   :func:`runGDE3NASSearch`.

3. **qEHVI** (BoTorch) — q-Expected Hypervolume Improvement, native
   multi-objective Bayesian optimisation (the BO baseline from the LaMOO
   paper). Use :func:`runQEHVINASSearch`. Requires ``pip install botorch``.

Scalarised (weighted sum fitness = alpha*f1 + (1-alpha)*f2, swept over alpha):

4. **Scalarised search** (mealpy — GA, SADE, AIW-PSO, GWO) — runs the
   single-objective mealpy algorithm once per alpha value. Sweeping alpha
   over ALPHA_SWEEP traces the Pareto front. Use
   :func:`runMONASScalarizedSweep`.

5. **Scalarised Bayesian Optimisation** — same single-objective BO from
   nas.py, applied over the scalarised objective. Use
   :func:`runBOMONASScalarizedSweep`.

Both _runPymooMONASSearch-based methods (NSGA-II, GDE3) and qEHVI share the
same JSON schema (``pareto_front`` + per-generation ``hv_history``), so they
are directly comparable to one another and aggregate identically.

Performance metric
~~~~~~~~~~~~~~~~~~
All algorithms are compared via the **hypervolume indicator** (HV): the
volume of objective space dominated by the returned front, bounded by the
reference point REF_POINT = (1.1, 1.1). Higher HV = better front.

The **true Pareto front** is obtainable by :func:`computeTrueParetoFront`,
which enumerates all 15 625 architectures exhaustively (cheap because the
benchmark is tabular).

Output schema
~~~~~~~~~~~~~
Each runner saves one JSON and one CSV per seed (or per alpha × seed for
scalarised methods). The JSON always contains::

    {
        'algorithm': str,
        'benchmark': 'NAS-Bench-201',
        'config': { dataset, hp, n_generations, pop_size, seed, ... },
        'pareto_front': [ { gene, arch_index, arch_str,
                            test_acc, flops_M, norm_flops,
                            f1, f2 }, ... ],
        'hypervolume': float,
        'hv_history': [float, ...],   # one per generation (NSGA-II only)
        'eval_log': [ { eval, arch_index, gene, test_acc, flops_M,
                        norm_flops, f1, f2, elapsed_s }, ... ],
    }

Dependencies
~~~~~~~~~~~~
    pip install pymoo          # for NSGA-II, GDE3 and the HV indicator
    pip install botorch        # for qEHVI (pulls torch + gpytorch)
    pip install mealpy         # for scalarised algorithms
    pip install bayesian-optimization  # for scalarised BO baseline

Quick usage
~~~~~~~~~~~
    from utils.nas_mo import (
        runNSGAIINASSearch, runGDE3NASSearch, runQEHVINASSearch,
        runMONASScalarizedSweep, runBOMONASScalarizedSweep,
        computeTrueParetoFront, computeFlopsRange,
    )
    from utils.nas import getApi

    api = getApi()

    # NSGA-II (native)
    front, run = runNSGAIINASSearch(api, dataset='cifar10', seed=0)

    # GDE3 (native, multi-objective DE)
    front, run = runGDE3NASSearch(api, dataset='cifar10', seed=0)

    # qEHVI (native, multi-objective BO) — budget = init_points + n_iter*batch_size
    front, run = runQEHVINASSearch(api, dataset='cifar10', seed=0,
                                   init_points=20, n_iter=200, batch_size=5)

    # Scalarised GA
    results = runMONASScalarizedSweep('GA', api, dataset='cifar10', seed=0)

    # True Pareto front (gold standard)
    true_front = computeTrueParetoFront(api, dataset='cifar10')
"""

import gc
import json
import os
from tabnanny import verbose
import time
from datetime import datetime
from pathlib import Path

from bayes_opt import BayesianOptimization, acquisition

import numpy as np
import pandas as pd

from utils.nas import (
    OPS, N_EDGES, N_OPS, N_ARCHS,
    geneToArchStr, queryArchitecture, getApi,
)

from utils.localMealpy import getOptimizationAlgorithm

from mealpy import IntegerVar

from pymoo.indicators.hv import HV
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
# NOTE: GDE3 is imported lazily inside runGDE3NASSearch (see below) so that a
# pymoo build without the gde3 module does not break importing this whole file.
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Reference point for the hypervolume indicator.
# Placed slightly beyond (1, 1) in the normalised (f1, f2) space so that
# every possible architecture is dominated by it.
REF_POINT = np.array([1.1, 1.1])

# Default alpha sweep: 9 evenly-spaced weights for the scalarisation
# alpha * f1 + (1 - alpha) * f2.  alpha → 1 biases towards accuracy,
# alpha → 0 biases towards efficiency.
ALPHA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs normalization
# ─────────────────────────────────────────────────────────────────────────────

_FLOPS_CACHE = {}


def computeFlopsRange(api, dataset='cifar10', hp='200', save_dir=None, verbose=True):
    """Enumerate all 15 625 architectures and return (min_flops_M, max_flops_M).

    Results are cached in-process so repeated calls within the same session
    are free. Optionally persists to a JSON file so future sessions can skip
    the enumeration entirely.

    Parameters
    ----------
    api : NATS-Bench API object.
    dataset, hp : benchmark parameters.
    save_dir : directory where ``flops_range_<dataset>_hp<hp>.json`` is
        written (and read on subsequent calls). Pass None to skip disk I/O.
    verbose : print progress and result.

    Returns
    -------
    (min_flops_M, max_flops_M) : tuple of float
    """
    key = (id(api), dataset, hp)
    if key in _FLOPS_CACHE:
        return _FLOPS_CACHE[key]

    # Try loading from disk cache first
    json_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(
            save_dir, f'flops_range_{dataset}_hp{hp}.json')
        if os.path.exists(json_path):
            print(f"  Loading FLOPs range from {json_path}")
            with open(json_path) as f:
                d = json.load(f)
            lo, hi = float(d['min_flops_M']), float(d['max_flops_M'])
            _FLOPS_CACHE[key] = (lo, hi)
            return lo, hi

    t0 = time.time()

    if verbose:
        print(f"  Computing FLOPs range over {N_ARCHS} architectures "
              f"({dataset}, hp={hp})…")

    flops_vals = []
    for idx in range(N_ARCHS):
        cost = api.get_cost_info(idx, dataset, hp=hp)
        f = cost.get('flops')
        if f is not None:
            flops_vals.append(float(f))

    lo, hi = float(np.min(flops_vals)), float(np.max(flops_vals))
    _FLOPS_CACHE[key] = (lo, hi)

    if verbose:
        print(f"  FLOPs range [{lo:.2f}, {hi:.2f}] M  "
              f"({len(flops_vals)} archs, {time.time() - t0:.1f}s)")

    # Persist to disk
    if json_path is not None:
        with open(json_path, 'w') as f:
            json.dump({'dataset': dataset, 'hp': hp,
                       'min_flops_M': lo, 'max_flops_M': hi,
                       'n_archs': len(flops_vals)}, f, indent=2)

    return lo, hi


def _normFlops(flops_M, flops_min, flops_max):
    """Normalise FLOPs to [0, 1]."""
    denom = flops_max - flops_min
    if denom < 1e-9:
        return 0.0
    return float(np.clip((flops_M - flops_min) / denom, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Pareto utilities
# ─────────────────────────────────────────────────────────────────────────────

def getNonDominated(points):
    points = np.asarray(points, dtype=float)
    n = len(points)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]


def computeHypervolume(front_points, ref_point=REF_POINT):
    F = np.asarray(front_points, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    # Keep only non-dominated points before computing HV
    nd_idx = getNonDominated(F)
    F_nd = F[nd_idx]
    hv_obj = HV(ref_point=np.asarray(ref_point, dtype=float))
    return float(hv_obj(F_nd))


# ─────────────────────────────────────────────────────────────────────────────
# True Pareto front (gold standard by exhaustive enumeration)
# ─────────────────────────────────────────────────────────────────────────────

def computeTrueParetoFront(api, dataset='cifar10', hp='200', fitness_key='test_acc', save_dir=None, verbose=True):
    """Exhaustively enumerate all 15 625 architectures and return the
    non-dominated set in (f1 = 1-acc, f2 = norm_flops) space.

    Optionally saves the front to a JSON file so future sessions can load it
    directly without re-enumerating the benchmark.

    Parameters
    ----------
    api : NATS-Bench API.
    dataset, hp, fitness_key : benchmark parameters.
    save_dir : directory where ``true_front_<dataset>_hp<hp>.json`` is
        written (and read on subsequent calls). Pass None to skip disk I/O.
    verbose : print progress and summary.

    Returns
    -------
    list of dict, sorted by f1 ascending (most accurate first).
    Each dict has keys: index, arch_str, test_acc, flops_M, norm_flops, f1, f2.
    """
    # Try loading from disk cache first
    json_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(
            save_dir, f'true_front_{dataset}_hp{hp}.json')
        if os.path.exists(json_path):
            print(f"  Loading true Pareto front from {json_path}")
            with open(json_path) as f:
                front = json.load(f)
            return front

    # Enumerate all architectures
    info_key = 'test-accuracy' if fitness_key == 'test_acc' else 'valid-accuracy'
    flops_min, flops_max = computeFlopsRange(api, dataset, hp, save_dir=save_dir,
                                             verbose=verbose)

    t0 = time.time()
    if verbose:
        print(f"  Enumerating true Pareto front over {N_ARCHS} architectures "
              f"({dataset}, hp={hp})…")

    rows = []
    for idx in range(N_ARCHS):
        info  = api.get_more_info(idx, dataset, hp=hp, is_random=False)
        cost  = api.get_cost_info(idx, dataset, hp=hp)
        acc   = (info.get(info_key) or 0.0) / 100.0
        flops = float(cost.get('flops') or 0.0)
        nf    = _normFlops(flops, flops_min, flops_max)
        rows.append({
            'index':      idx,
            'arch_str':   api.arch(idx),
            'test_acc':   acc,
            'flops_M':    flops,
            'norm_flops': nf,
            'f1':         round(1.0 - acc, 6),
            'f2':         round(nf, 6),
        })

    F = np.array([[r['f1'], r['f2']] for r in rows])
    nd_idx = getNonDominated(F)
    front  = sorted([rows[i] for i in nd_idx], key=lambda r: r['f1'])

    if verbose:
        print(f"  True Pareto front: {len(front)} architectures "
              f"({time.time() - t0:.1f}s)")

    # Persist to disk
    if json_path is not None:
        with open(json_path, 'w') as f:
            json.dump(front, f, indent=2)

    return front


# ─────────────────────────────────────────────────────────────────────────────
# Validation / test split
# ─────────────────────────────────────────────────────────────────────────────

# queryValidTest lives in utils.nas so both pipelines share a single memoized
# cache instead of each keeping its own. See the note there on why the lookups
# are memoized: the NATS-Bench API caches every architecture it loads and never
# evicts, so a long sweep grows without bound.
from utils.nas import queryValidTest, clearQueryCache          # noqa: E402


def scoredPoint(api, gene, dataset, hp, flops_min, flops_max, **extra):
    """Final record for one architecture.

    ``f1``/``f2`` are on TEST accuracy so hypervolume and IGD+ stay comparable
    with the exhaustive reference front, which is also defined on test.
    """
    r = queryValidTest(api, gene, dataset=dataset, hp=hp)
    nf = _normFlops(r['flops_M'], flops_min, flops_max)
    return {
        'gene':       list(gene),
        'arch_index': r['index'],
        'arch_str':   r['arch_str'],
        'valid_acc':  round(r['valid_acc'], 6),
        'test_acc':   round(r['test_acc'], 6),
        'flops_M':    round(r['flops_M'], 4),
        'norm_flops': round(nf, 6),
        'f1':         round(1.0 - r['test_acc'], 6),
        'f2':         round(nf, 6),
        **extra,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scalarised objective (for mealpy single-objective algorithms)
# ─────────────────────────────────────────────────────────────────────────────

def makeScalarizedObjective(api, dataset='cifar10', hp='200', alpha=0.5, flops_min=None, flops_max=None, eval_log=None, counter=None, verbose=True):
    if flops_min is None or flops_max is None:
        flops_min, flops_max = computeFlopsRange(api, dataset, hp, verbose=verbose)
    if eval_log is None:
        eval_log = []
    if counter is None:
        counter = {'n': 0, 'best_fitness': 1e9, 'start': time.time()}

    def objective(solution):
        gene = [int(np.round(x)) for x in solution]
        try:
            r   = queryValidTest(api, gene, dataset=dataset, hp=hp)
            acc   = r['valid_acc']  or 0.0   # SEARCH SIGNAL: validation only
            flops = r['flops_M']    or 0.0
        except Exception as e:
            print(f"  [ERROR] query failed: {e}")
            acc, flops = 0.0, 0.0
            r = {'index': -1, 'arch_str': '?', 'valid_acc': 0.0,
                 'test_acc': 0.0, 'flops_M': 0.0}

        nf      = _normFlops(flops, flops_min, flops_max)
        f1      = 1.0 - acc
        f2      = nf
        fitness = alpha * f1 + (1.0 - alpha) * f2

        counter['n'] += 1
        if fitness < counter['best_fitness']:
            counter['best_fitness'] = fitness
        elapsed = time.time() - counter['start']

        eval_log.append({
            'eval':       counter['n'],
            'arch_index': r['index'],
            'arch_str':   r['arch_str'],
            'gene':       ' '.join(str(g) for g in gene),
            'valid_acc':  round(acc, 6),
            'test_acc':   round(r.get('test_acc', 0.0), 6),
            'flops_M':    round(flops, 4),
            'norm_flops': round(nf, 6),
            'f1':         round(f1, 6),
            'f2':         round(f2, 6),
            'alpha':      alpha,
            'fitness':    round(fitness, 6),
            'elapsed_s':  round(elapsed, 3),
        })

        if verbose:
            print(f"  [{counter['n']:>4}] gene={gene}  "
                  f"valid_acc={acc:.4f}  flops={flops:.1f}M  "
                  f"fitness(α={alpha})={fitness:.4f}")

        return fitness

    objective.eval_log  = eval_log
    objective.counter   = counter
    objective.alpha     = alpha
    objective.flops_min = flops_min
    objective.flops_max = flops_max
    return objective


# ─────────────────────────────────────────────────────────────────────────────
# pymoo Problem class (for NSGA-II)
# ─────────────────────────────────────────────────────────────────────────────

class NASMOProblem:
    """Wrapper that adapts NAS-Bench-201 to the pymoo Problem interface.

    Each variable is a float in [0, 4]; it is rounded to the nearest integer
    on evaluation to select one of the 5 operations on each of the 6 edges.
    Two objectives are returned: [f1 = 1 - acc, f2 = norm_flops].
    """

    def __init__(self, api, dataset='cifar10', hp='200', flops_min=None, flops_max=None, eval_log=None, counter=None, verbose=False):
        
        if flops_min is None or flops_max is None:
            flops_min, flops_max = computeFlopsRange(api, dataset, hp,
                                                     verbose=True)
        if eval_log is None:
            eval_log = []
        if counter is None:
            counter = {'n': 0, 'start': time.time()}

        self.api       = api
        self.dataset   = dataset
        self.hp        = hp
        self.flops_min = flops_min
        self.flops_max = flops_max
        self.eval_log  = eval_log
        self.counter   = counter
        self.verbose   = verbose

        # Build the pymoo Problem subclass inline so we can capture self.
        outer = self

        class _Problem(Problem):
            def __init__(self_inner):
                super().__init__(
                    n_var=N_EDGES,
                    n_obj=2,
                    xl=np.zeros(N_EDGES),
                    xu=np.full(N_EDGES, N_OPS - 1, dtype=float),
                )

            def _evaluate(self_inner, X, out, *args, **kwargs):
                F = np.zeros((len(X), 2))
                for i, x in enumerate(X):
                    gene = [int(round(v)) for v in x]
                    gene = [max(0, min(N_OPS - 1, g)) for g in gene]
                    try:
                        r = queryValidTest(outer.api, gene,
                                           dataset=outer.dataset, hp=outer.hp)
                        acc = r['valid_acc'] or 0.0   # SEARCH SIGNAL
                        flops = r['flops_M']  or 0.0
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                        acc, flops, r = 0.0, 0.0, {
                            'index': -1, 'arch_str': '?', 'valid_acc': 0.0,
                            'test_acc': 0.0,
                        }

                    nf = _normFlops(flops, outer.flops_min, outer.flops_max)
                    f1 = 1.0 - acc
                    f2 = nf
                    F[i] = [f1, f2]

                    outer.counter['n'] += 1
                    elapsed = time.time() - outer.counter['start']
                    outer.eval_log.append({
                        'eval':       outer.counter['n'],
                        'arch_index': r.get('index', -1),
                        'arch_str':   r.get('arch_str', '?'),
                        'gene':       ' '.join(str(g) for g in gene),
                        'valid_acc':  round(acc, 6),
                        'test_acc':   round(r.get('test_acc', 0.0), 6),
                        'flops_M':    round(flops, 4),
                        'norm_flops': round(nf, 6),
                        'f1':         round(f1, 6),
                        'f2':         round(f2, 6),
                        'alpha':      None,
                        'fitness':    None,
                        'elapsed_s':  round(elapsed, 3),
                    })

                    if outer.verbose:
                        print(f"  [{outer.counter['n']:>4}] "
                              f"gene={gene}  valid_acc={acc:.4f}  "
                              f"flops={flops:.1f}M")

                out['F'] = F

        self.pymoo_problem = _Problem()


# ─────────────────────────────────────────────────────────────────────────────
# HV-tracking callback for NSGA-II
# ─────────────────────────────────────────────────────────────────────────────

def _hvOfArchive(objs):
    """Hypervolume of everything evaluated so far.

    ``computeHypervolume`` already reduces to the non-dominated subset.
    """
    return computeHypervolume(objs) if objs else 0.0


class _HVCallback:
    """Tracks hypervolume after each generation.

    The optimizer's own ``algorithm.opt`` front is in VALIDATION objectives
    (that is what it searches on), but the reported hypervolume — and the
    reference front it is compared against — are on TEST. Tracing the
    validation front here would produce a convergence curve that never lands
    on the final reported value. So the trace is rebuilt from the eval log,
    which carries the test score of every architecture evaluated so far.
    Both traces are kept: ``hv_history`` (test) and ``hv_history_valid``.
    """

    def __init__(self, ref_point=REF_POINT, eval_log=None):
        self.ref_point        = ref_point
        self.hv_history       = []
        self.hv_history_valid = []
        self.eval_log         = eval_log
        self._hv_obj          = None  # lazily constructed

    def _hv(self):
        if self._hv_obj is None:
            self._hv_obj = HV(ref_point=self.ref_point)
        return self._hv_obj

    def __call__(self, algorithm):
        F = algorithm.opt.get('F')   # validation-space front
        self.hv_history_valid.append(
            float(self._hv()(F)) if F is not None and len(F) else 0.0)

        if self.eval_log:
            pts = [[1.0 - (e.get('test_acc') or 0.0), e.get('norm_flops') or 0.0]
                   for e in self.eval_log]
            self.hv_history.append(_hvOfArchive(pts))
        else:
            self.hv_history.append(self.hv_history_valid[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Runner: NSGA-II
# ─────────────────────────────────────────────────────────────────────────────

def runNSGAIINASSearch(
    api,
    dataset       = 'cifar10',
    hp            = '200',
    n_generations = 50,
    pop_size      = 20,
    seed          = None,
    flops_min     = None,
    flops_max     = None,
    results_dir   = 'results/mo_nas/nsga2',
    verbose       = False,
):
    """Run NSGA-II on NAS-Bench-201 with objectives (1-acc, norm_flops).

    Parameters
    ----------
    api : NATS-Bench API.
    dataset : one of 'cifar10', 'cifar100', 'ImageNet16-120'.
    hp : '12' or '200'.
    n_generations : number of NSGA-II generations.
    pop_size : population size (total evals ≈ (n_generations+1) * pop_size).
    seed : random seed (drawn from OS entropy if None).
    flops_min, flops_max : FLOPs normalisation bounds. Computed if None.
    results_dir : directory for JSON + CSV output.
    verbose : per-evaluation logging.

    Returns
    -------
    (pareto_front, run_results) where pareto_front is a list of dicts and
    run_results is the full JSON-serialisable result dict.
    """


    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    np.random.seed(seed)

    total_evals = (n_generations + 1) * pop_size
    print(f"\n{'='*62}")
    print(f"  MO-NAS: NSGA-II  ({dataset}, hp={hp})")
    print(f"  Generations: {n_generations}  |  Pop: {pop_size}  |  "
          f"Total queries: {total_evals}  |  Seed: {seed}")
    print(f"{'='*62}\n")

    # Precompute FLOPs range if not provided
    if flops_min is None or flops_max is None:
        flops_min, flops_max = computeFlopsRange(api, dataset, hp, verbose=True)

    # Build problem
    wrapper = NASMOProblem(api, dataset=dataset, hp=hp, flops_min=flops_min, flops_max=flops_max, verbose=verbose)
    problem  = wrapper.pymoo_problem
    eval_log = wrapper.eval_log
    counter  = wrapper.counter
    counter['start'] = time.time()

    # HV callback
    hv_cb = _HVCallback(REF_POINT, eval_log=eval_log)

    # NSGA-II setup — SBX crossover + polynomial mutation, both float-encoded
    # with rounding on evaluation (standard practice for integer NAS spaces).
    algorithm = NSGA2(
        pop_size  = pop_size,
        sampling  = FloatRandomSampling(),
        crossover = SBX(prob=0.9, eta=15),
        mutation  = PM(prob=1.0 / N_EDGES, eta=20),
        eliminate_duplicates=False,   # duplicates map to same arch, fine
    )

    t0 = time.time()
    result = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_generations),
        seed=seed,
        verbose=False,
        callback=hv_cb,
    )
    t_total = time.time() - t0

    # Extract final Pareto front
    F_final = result.F      # shape (n_front, 2)
    X_final = result.X      # shape (n_front, n_var)

    pareto_front = []
    for f_vec, x_vec in zip(F_final, X_final):
        gene  = [max(0, min(N_OPS - 1, int(round(v)))) for v in x_vec]
        try:
            pt = scoredPoint(api, gene, dataset, hp, flops_min, flops_max,
                             f1_valid=round(float(f_vec[0]), 6))
        except Exception:
            pt = {'gene': gene, 'arch_index': -1, 'arch_str': '?',
                  'valid_acc': 0.0, 'test_acc': 0.0, 'flops_M': 0.0,
                  'norm_flops': 0.0, 'f1': 1.0, 'f2': 0.0, 'f1_valid': 1.0}
        pareto_front.append(pt)

    # Sort by f1 (ascending accuracy gap)
    pareto_front.sort(key=lambda p: p['f1'])

    final_hv = computeHypervolume(
        [[p['f1'], p['f2']] for p in pareto_front]
    )

    if True:  # always print summary
        print(f"\n{'='*62}")
        print(f"  FINAL — NSGA-II  ({dataset})")
        print(f"  |Pareto front|  : {len(pareto_front)} architectures")
        print(f"  Hypervolume     : {final_hv:.6f}")
        print(f"  Best accuracy   : {max(p['test_acc'] for p in pareto_front):.4f}")
        print(f"  Min FLOPs       : {min(p['flops_M'] for p in pareto_front):.1f} M")
        print(f"  Total evals     : {counter['n']}")
        print(f"  Wallclock       : {t_total:.1f}s")
        print()

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_results = {
        'algorithm':   'NSGA2',
        'benchmark':   'NAS-Bench-201',
        'timestamp':   datetime.now().isoformat(),
        'config': {
            'dataset':       dataset,
            'hp':            hp,
            'n_generations': n_generations,
            'pop_size':      pop_size,
            'seed':          seed,
            'flops_min':     flops_min,
            'flops_max':     flops_max,
            'ref_point':     list(REF_POINT),
        },
        'pareto_front':        pareto_front,
        'hypervolume':         round(final_hv, 8),
        'hv_history':          [round(h, 8) for h in hv_cb.hv_history],
        'hv_history_valid':    [round(h, 8) for h in hv_cb.hv_history_valid],
        'n_evaluations':       counter['n'],
        'n_unique_architectures': len({e['arch_index'] for e in eval_log}),
        'total_time_s':        round(t_total, 1),
        'eval_log':            eval_log,
    }

    json_path = os.path.join(results_dir, f'NSGA2_{ts}_seed{seed}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)

    df = pd.DataFrame(eval_log)
    csv_path = os.path.join(results_dir, f'NSGA2_{ts}_seed{seed}_evals.csv')
    df.to_csv(csv_path, index=False)

    print(f"  Saved: {json_path}")
    gc.collect()   # the run's eval log is dead by here; reclaim it
    return pareto_front, run_results


# ─────────────────────────────────────────────────────────────────────────────
# Shared driver for native pymoo multi-objective algorithms (NSGA-II, GDE3)
# ─────────────────────────────────────────────────────────────────────────────

def _runPymooMONASSearch(
    api,
    algorithm,
    algo_name,
    dataset       = 'cifar10',
    hp            = '200',
    n_generations = 50,
    pop_size      = 20,
    seed          = None,
    flops_min     = None,
    flops_max     = None,
    results_dir   = 'results/mo_nas',
    verbose       = False,
):
    """Generic runner for any *constructed* pymoo multi-objective ``algorithm``.

    NSGA-II and GDE3 differ only in the algorithm object handed in; the
    NAS-Bench-201 problem wrapper, the per-generation HV tracking, the
    Pareto-front extraction and the JSON/CSV output schema are identical, so
    the two are directly comparable. ``algo_name`` is used only for logging
    and output filenames.

    Returns
    -------
    (pareto_front, run_results)
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    np.random.seed(seed)

    total_evals = (n_generations + 1) * pop_size
    print(f"\n{'='*62}")
    print(f"  MO-NAS: {algo_name}  ({dataset}, hp={hp})")
    print(f"  Generations: {n_generations}  |  Pop: {pop_size}  |  "
          f"Total queries: {total_evals}  |  Seed: {seed}")
    print(f"{'='*62}\n")

    # Precompute FLOPs range if not provided
    if flops_min is None or flops_max is None:
        flops_min, flops_max = computeFlopsRange(api, dataset, hp, verbose=True)

    # Build problem (same wrapper as NSGA-II → same encoding & objectives)
    wrapper  = NASMOProblem(api, dataset=dataset, hp=hp,
                            flops_min=flops_min, flops_max=flops_max,
                            verbose=verbose)
    problem  = wrapper.pymoo_problem
    eval_log = wrapper.eval_log
    counter  = wrapper.counter
    counter['start'] = time.time()

    # HV callback (per-generation history of the non-dominated front)
    hv_cb = _HVCallback(REF_POINT, eval_log=eval_log)

    t0 = time.time()
    result = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_generations),
        seed=seed,
        verbose=False,
        callback=hv_cb,
    )
    t_total = time.time() - t0

    # Extract final Pareto front (guard against empty/1-D results)
    F_final = result.F
    X_final = result.X
    if F_final is None or X_final is None or len(F_final) == 0:
        F_final = np.empty((0, 2))
        X_final = np.empty((0, N_EDGES))
    F_final = np.atleast_2d(F_final)
    X_final = np.atleast_2d(X_final)

    pareto_front = []
    for f_vec, x_vec in zip(F_final, X_final):
        gene = [max(0, min(N_OPS - 1, int(round(v)))) for v in x_vec]
        try:
            pt = scoredPoint(api, gene, dataset, hp, flops_min, flops_max,
                             f1_valid=round(float(f_vec[0]), 6))
        except Exception:
            pt = {'gene': gene, 'arch_index': -1, 'arch_str': '?',
                  'valid_acc': 0.0, 'test_acc': 0.0, 'flops_M': 0.0,
                  'norm_flops': 0.0, 'f1': 1.0, 'f2': 0.0, 'f1_valid': 1.0}
        pareto_front.append(pt)

    pareto_front.sort(key=lambda p: p['f1'])
    final_hv = (computeHypervolume([[p['f1'], p['f2']] for p in pareto_front])
                if pareto_front else 0.0)

    print(f"\n{'='*62}")
    print(f"  FINAL — {algo_name}  ({dataset})")
    print(f"  |Pareto front|  : {len(pareto_front)} architectures")
    print(f"  Hypervolume     : {final_hv:.6f}")
    if pareto_front:
        print(f"  Best accuracy   : {max(p['test_acc'] for p in pareto_front):.4f}")
        print(f"  Min FLOPs       : {min(p['flops_M'] for p in pareto_front):.1f} M")
    print(f"  Total evals     : {counter['n']}")
    print(f"  Wallclock       : {t_total:.1f}s\n")

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_results = {
        'algorithm':   algo_name,
        'benchmark':   'NAS-Bench-201',
        'timestamp':   datetime.now().isoformat(),
        'config': {
            'dataset':       dataset,
            'hp':            hp,
            'n_generations': n_generations,
            'pop_size':      pop_size,
            'seed':          seed,
            'flops_min':     flops_min,
            'flops_max':     flops_max,
            'ref_point':     list(REF_POINT),
        },
        'pareto_front':           pareto_front,
        'hypervolume':            round(final_hv, 8),
        'hv_history':             [round(h, 8) for h in hv_cb.hv_history],
        'hv_history_valid':       [round(h, 8) for h in hv_cb.hv_history_valid],
        'n_evaluations':          counter['n'],
        'n_unique_architectures': len({e['arch_index'] for e in eval_log}),
        'total_time_s':           round(t_total, 1),
        'eval_log':               eval_log,
    }

    json_path = os.path.join(results_dir, f'{algo_name}_{ts}_seed{seed}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)

    pd.DataFrame(eval_log).to_csv(
        os.path.join(results_dir, f'{algo_name}_{ts}_seed{seed}_evals.csv'),
        index=False)

    print(f"  Saved: {json_path}")
    gc.collect()   # the run's eval log is dead by here; reclaim it
    return pareto_front, run_results


# ─────────────────────────────────────────────────────────────────────────────
# Runner: GDE3  (native multi-objective Differential Evolution, pymoo)
# ─────────────────────────────────────────────────────────────────────────────

def runGDE3NASSearch(
    api,
    dataset       = 'cifar10',
    hp            = '200',
    n_generations = 50,
    pop_size      = 20,
    seed          = None,
    CR            = 0.9,
    F             = 0.5,
    variant       = 'DE/rand/1/bin',
    flops_min     = None,
    flops_max     = None,
    results_dir   = 'results/mo_nas/gde3',
    verbose       = False,
):
    """Run GDE3 on NAS-Bench-201 with objectives (1-acc, norm_flops).

    GDE3 — Generalised Differential Evolution 3 (Kukkonen & Lampinen, 2005) —
    is the native multi-objective extension of Differential Evolution. Each
    generation it builds trial vectors with DE mutation + crossover and then
    applies Pareto-based one-to-one selection followed by NSGA-II-style
    non-dominated sorting + crowding-distance truncation. Like NSGA-II it
    returns a single non-dominated front per seed (no alpha sweep needed).

    The 6 categorical edge operations are encoded as floats in [0, N_OPS-1]
    and rounded on evaluation (continuous relaxation) — identical to the
    NSGA-II setup so the two EAs are directly comparable.

    Parameters
    ----------
    CR : float — DE crossover rate (default 0.9).
    F  : float — DE differential weight / scale factor (default 0.5).
    variant : str — DE strategy, e.g. 'DE/rand/1/bin' or 'DE/best/1/bin'.
    All other parameters mirror :func:`runNSGAIINASSearch`.

    Returns
    -------
    (pareto_front, run_results)
    """
    try:
        from pymoo.algorithms.moo.gde3 import GDE3
    except ImportError as e:
        import pymoo
        raise ImportError(
            f"GDE3 is not available in your pymoo install "
            f"(version {getattr(pymoo, '__version__', '?')}). "
            f"runGDE3NASSearch needs pymoo >= 0.6 — upgrade with: "
            f"pip install -U pymoo\n  (original import error: {e})"
        )

    algorithm = GDE3(
        pop_size = pop_size,
        sampling = FloatRandomSampling(),
        variant  = variant,
        CR       = CR,
        F        = F,
    )
    pareto_front, run_results = _runPymooMONASSearch(
        api, algorithm, 'GDE3',
        dataset=dataset, hp=hp,
        n_generations=n_generations, pop_size=pop_size, seed=seed,
        flops_min=flops_min, flops_max=flops_max,
        results_dir=results_dir, verbose=verbose,
    )
    # Record the DE-specific hyperparameters for reproducibility.
    run_results['config'].update({'CR': CR, 'F': F, 'variant': variant})
    gc.collect()   # the run's eval log is dead by here; reclaim it
    return pareto_front, run_results


# ─────────────────────────────────────────────────────────────────────────────
# Runner: scalarised mealpy algorithms (one alpha, one seed)
# ─────────────────────────────────────────────────────────────────────────────

def _mealpyBounds():
    return [IntegerVar(lb=0, ub=N_OPS - 1, name=f'edge_{i}')
            for i in range(N_EDGES)]


def runMONASScalarized(
    algorithm,
    api,
    dataset       = 'cifar10',
    hp            = '200',
    alpha         = 0.5,
    n_generations = 50,
    pop_size      = 20,
    seed          = None,
    algo_params   = None,
    flops_min     = None,
    flops_max     = None,
    results_dir   = 'results/mo_nas',
    verbose       = False,
):
    """Run one mealpy algorithm with a scalarised MO objective.

    Parameters
    ----------
    algorithm : str — 'GA', 'SADE', 'AIW_PSO', 'GWO'.
    alpha : float in [0, 1] — weight on the accuracy objective.
    All other parameters mirror :func:`utils.nas.runNASSearch`.

    Returns
    -------
    (best_point_dict, run_results_dict)
        best_point_dict has keys: gene, arch_index, arch_str,
            test_acc, flops_M, norm_flops, f1, f2, fitness, alpha.
    """


    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    np.random.seed(seed)

    if flops_min is None or flops_max is None:
        flops_min, flops_max = computeFlopsRange(api, dataset, hp,
                                                 verbose=True)

    alg_name = algorithm.upper()
    total_evals = (n_generations + 1) * pop_size

    print(f"  [{alg_name}  α={alpha:.1f}  seed={seed}]  "
          f"{dataset}  {total_evals} evals", end=' ', flush=True)

    eval_log = []
    counter  = {'n': 0, 'best_fitness': 1e9, 'start': time.time()}
    objective = makeScalarizedObjective(
        api, dataset=dataset, hp=hp, alpha=alpha,
        flops_min=flops_min, flops_max=flops_max,
        eval_log=eval_log, counter=counter, verbose=verbose,
    )

    problem = {
        'obj_func': objective,
        'bounds':   _mealpyBounds(),
        'minmax':   'min',
        'log_to':   None,
    }
    algo = getOptimizationAlgorithm(alg_name, n_generations, pop_size,
                                    algo_params)

    t0      = time.time()
    g_best  = algo.solve(problem, seed=seed)
    t_total = time.time() - t0

    best_gene = [int(np.round(x)) for x in g_best.solution]
    best_gene = [max(0, min(N_OPS - 1, g)) for g in best_gene]

    best_point = scoredPoint(api, best_gene, dataset, hp, flops_min, flops_max,
                             alpha=alpha)
    # The search fitness is on validation; the reported f1 is on test.
    best_point['fitness'] = round(
        alpha * (1.0 - best_point['valid_acc'])
        + (1 - alpha) * best_point['norm_flops'], 6)

    print(f"→ valid={best_point['valid_acc']:.4f}  "
          f"test={best_point['test_acc']:.4f}  "
          f"flops={best_point['flops_M']:.1f}M  "
          f"fit={best_point['fitness']:.4f}  t={t_total:.0f}s")

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_results = {
        'algorithm':   alg_name,
        'benchmark':   'NAS-Bench-201',
        'timestamp':   datetime.now().isoformat(),
        'config': {
            'dataset':       dataset,
            'hp':            hp,
            'alpha':         alpha,
            'n_generations': n_generations,
            'pop_size':      pop_size,
            'seed':          seed,
            'flops_min':     flops_min,
            'flops_max':     flops_max,
            'ref_point':     list(REF_POINT),
        },
        'best':             best_point,
        'hypervolume':      round(computeHypervolume([[best_point['f1'], best_point['f2']]]), 8),
        'hv_history':       [],    # not available for single-obj algorithms
        'n_evaluations':    counter['n'],
        'n_unique_architectures': len({e['arch_index'] for e in eval_log}),
        'total_time_s':     round(t_total, 1),
        'eval_log':         eval_log,
    }

    json_path = os.path.join(
        results_dir, f'{alg_name}_a{int(alpha*10):02d}_{ts}_seed{seed}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)

    df = pd.DataFrame(eval_log)
    csv_path = json_path.replace('.json', '_evals.csv')
    df.to_csv(csv_path, index=False)

    gc.collect()   # the run's eval log is dead by here; reclaim it
    return best_point, run_results


# ─────────────────────────────────────────────────────────────────────────────
# Runner: scalarised sweep (all alphas, one seed per alpha)
# ─────────────────────────────────────────────────────────────────────────────

def runMONASScalarizedSweep(
    algorithm,
    api,
    dataset       = 'cifar10',
    hp            = '200',
    alpha_values  = None,
    n_generations = 50,
    pop_size      = 20,
    seed          = None,
    algo_params   = None,
    results_dir   = None,
    verbose       = False,
):
    """Sweep over alpha values to approximate the Pareto front.

    Runs :func:`runMONASScalarized` for each alpha in alpha_values.
    The aggregate of all returned best points (filtered for non-dominance)
    forms the front approximation for this algorithm.

    Parameters
    ----------
    alpha_values : list of floats in [0, 1]. Defaults to ALPHA_SWEEP.
    seed : base seed; each alpha uses seed + i to ensure reproducibility.
    results_dir : defaults to 'results/mo_nas/<alg_name_lower>/<dataset>'.

    Returns
    -------
    (front, all_points) where front is the non-dominated subset and
    all_points is the full list of best points from each alpha run.
    """
    if alpha_values is None:
        alpha_values = ALPHA_SWEEP
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)

    alg_name = algorithm.upper()
    if results_dir is None:
        results_dir = f'results/mo_nas/{alg_name.lower()}/{dataset}'

    print(f"\n{'='*62}")
    print(f"  MO-NAS SCALARIZED SWEEP: {alg_name}  ({dataset}, hp={hp})")
    print(f"  Alphas: {alpha_values}  |  Base seed: {seed}")
    print(f"{'='*62}")

    # Precompute FLOPs range once, shared across all alpha runs
    flops_min, flops_max = computeFlopsRange(api, dataset, hp, verbose=True)

    all_points = []
    for i, alpha in enumerate(alpha_values):
        pt, _ = runMONASScalarized(
            alg_name, api,
            dataset=dataset, hp=hp,
            alpha=alpha,
            n_generations=n_generations, pop_size=pop_size,
            seed=seed + i,
            algo_params=algo_params,
            flops_min=flops_min, flops_max=flops_max,
            results_dir=results_dir,
            verbose=verbose,
        )
        all_points.append(pt)

    # Filter non-dominated
    F = np.array([[p['f1'], p['f2']] for p in all_points])
    nd_idx = getNonDominated(F)
    front  = [all_points[i] for i in nd_idx]
    front.sort(key=lambda p: p['f1'])

    final_hv = computeHypervolume([[p['f1'], p['f2']] for p in front])
    print(f"\n  {alg_name} sweep done — "
          f"|front|={len(front)}  HV={final_hv:.6f}")

    gc.collect()   # the run's eval log is dead by here; reclaim it
    return front, all_points


# ─────────────────────────────────────────────────────────────────────────────
# Runner: scalarised Bayesian Optimisation — single alpha, single seed
# ─────────────────────────────────────────────────────────────────────────────

def runBOMONASScalarized(
    api,
    dataset     = 'cifar10',
    hp          = '200',
    alpha       = 0.5,
    n_iter      = 200,
    init_points = 20,
    xi          = 0.01,
    seed        = None,
    flops_min   = None,
    flops_max   = None,
    results_dir = 'results/mo_nas/bo',
    verbose     = False,
):
    """Bayesian Optimisation with a scalarised objective for one alpha and one seed.

    Mirrors :func:`runMONASScalarized` but uses GP + Expected Improvement
    instead of a mealpy algorithm. Designed to be called from the notebook
    in a ``for seed in SEEDS: for alpha in ALPHA_SWEEP`` loop.

    Parameters
    ----------
    alpha : float in [0, 1] — weight on the accuracy objective.
    n_iter, init_points, xi : BO budget and acquisition parameters.
    flops_min, flops_max : pre-computed FLOPs bounds. Computed if None.

    Returns
    -------
    (best_point_dict, run_results_dict)
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    np.random.seed(seed)

    if flops_min is None or flops_max is None:
        flops_min, flops_max = computeFlopsRange(api, dataset, hp)

    eval_log = []
    counter  = {'n': 0, 'best_fitness': 1e9, 'start': time.time()}
    mo_obj   = makeScalarizedObjective(
        api, dataset=dataset, hp=hp, alpha=alpha,
        flops_min=flops_min, flops_max=flops_max,
        eval_log=eval_log, counter=counter, verbose=verbose,
    )

    # bayes_opt maximises — wrap to minimise the scalarised fitness
    def _bo_obj(e0, e1, e2, e3, e4, e5):
        return -mo_obj([e0, e1, e2, e3, e4, e5])

    # Declare the six edges as integer variables, matching
    # runBONASSearch. The space is discrete either way -- the
    # objective rounds before querying -- but with the type
    # declared, bayes_opt registers the rounded point, so it can
    # see that a proposal repeats an architecture it already
    # evaluated instead of treating two nearby continuous points
    # as distinct observations.
    pbounds = {f'e{j}': (0, N_OPS - 1, int) for j in range(N_EDGES)}
    acq = acquisition.ExpectedImprovement(xi=xi, random_state=seed)
    bo  = BayesianOptimization(
        f=_bo_obj, pbounds=pbounds,
        acquisition_function=acq,
        random_state=seed, verbose=0,
    )

    t0 = time.time()
    bo.maximize(init_points=init_points, n_iter=n_iter)
    t_total = time.time() - t0

    best_params = bo.max['params']
    best_gene   = [max(0, min(N_OPS - 1, int(round(best_params[f'e{j}']))))
                   for j in range(N_EDGES)]
    best_point = scoredPoint(api, best_gene, dataset, hp, flops_min, flops_max,
                             alpha=alpha)
    best_point['fitness'] = round(
        alpha * (1.0 - best_point['valid_acc'])
        + (1 - alpha) * best_point['norm_flops'], 6)

    print(f"  [BO  α={alpha:.1f}  seed={seed}]  {dataset}  "
          f"→ valid={best_point['valid_acc']:.4f}  "
          f"test={best_point['test_acc']:.4f}  "
          f"flops={best_point['flops_M']:.1f}M  t={t_total:.0f}s")

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_results = {
        'algorithm':   'BO',
        'benchmark':   'NAS-Bench-201',
        'timestamp':   datetime.now().isoformat(),
        'config': {
            'dataset': dataset, 'hp': hp, 'alpha': alpha,
            'n_iter': n_iter, 'init_points': init_points,
            'xi': xi, 'seed': seed,
            'flops_min': flops_min, 'flops_max': flops_max,
            'ref_point': list(REF_POINT),
        },
        'best':        best_point,
        'hypervolume': round(computeHypervolume([[best_point['f1'], best_point['f2']]]), 8),
        'hv_history':  [],
        'n_evaluations':          counter['n'],
        'n_unique_architectures': len({e['arch_index'] for e in eval_log}),
        'total_time_s': round(t_total, 1),
        'eval_log':    eval_log,
    }

    json_path = os.path.join(
        results_dir, f'BO_a{int(alpha*10):02d}_{ts}_seed{seed}.json')
    with open(json_path, 'w') as fh:
        json.dump(run_results, fh, indent=2)
    pd.DataFrame(eval_log).to_csv(
        json_path.replace('.json', '_evals.csv'), index=False)

    gc.collect()   # the run's eval log is dead by here; reclaim it
    return best_point, run_results


# ─────────────────────────────────────────────────────────────────────────────
# Runner: scalarised Bayesian Optimisation sweep
# ─────────────────────────────────────────────────────────────────────────────

def runBOMONASScalarizedSweep(
    api,
    dataset      = 'cifar10',
    hp           = '200',
    alpha_values = None,
    n_iter       = 200,
    init_points  = 20,
    xi           = 0.01,
    seed         = None,
    results_dir  = None,
    verbose      = False,
):
    """Bayesian Optimisation with a scalarised objective, sweeping alpha.

    Uses the same GP + Expected Improvement BO as Study 2, applied to the
    weighted-sum objective for each alpha. Returns the aggregate Pareto front.

    Parameters
    ----------
    alpha_values : list of floats. Defaults to ALPHA_SWEEP.
    n_iter, init_points, xi : same as :func:`utils.nas.runBONASSearch`.
    seed : base seed; alpha[i] uses seed + i.
    results_dir : defaults to 'results/mo_nas/bo/<dataset>'.

    Returns
    -------
    (front, all_points)
    """


    if alpha_values is None:
        alpha_values = ALPHA_SWEEP
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    if results_dir is None:
        results_dir = f'results/mo_nas/bo/{dataset}'

    print(f"\n{'='*62}")
    print(f"  MO-NAS BO SWEEP  ({dataset}, hp={hp})")
    print(f"  Alphas: {alpha_values}  |  Base seed: {seed}")
    print(f"{'='*62}")

    flops_min, flops_max = computeFlopsRange(api, dataset, hp, verbose=True)
    os.makedirs(results_dir, exist_ok=True)

    all_points = []
    for i, alpha in enumerate(alpha_values):
        run_seed = seed + i
        np.random.seed(run_seed)

        eval_log = []
        counter  = {'n': 0, 'best_fitness': 1e9, 'start': time.time()}
        mealpy_obj = makeScalarizedObjective(
            api, dataset=dataset, hp=hp, alpha=alpha,
            flops_min=flops_min, flops_max=flops_max,
            eval_log=eval_log, counter=counter, verbose=verbose,
        )

        # bayes_opt maximises, so wrap accordingly
        def _bo_obj(e0, e1, e2, e3, e4, e5):
            fit = mealpy_obj([e0, e1, e2, e3, e4, e5])
            return -fit   # maximise negative fitness

        # Declare the six edges as integer variables, matching
        # runBONASSearch. The space is discrete either way -- the
        # objective rounds before querying -- but with the type
        # declared, bayes_opt registers the rounded point, so it can
        # see that a proposal repeats an architecture it already
        # evaluated instead of treating two nearby continuous points
        # as distinct observations.
        pbounds = {f'e{j}': (0, N_OPS - 1, int) for j in range(N_EDGES)}
        acq     = acquisition.ExpectedImprovement(xi=xi, random_state=run_seed)
        bo      = BayesianOptimization(
            f=_bo_obj, pbounds=pbounds,
            acquisition_function=acq,
            random_state=run_seed, verbose=0,
        )

        t0 = time.time()
        bo.maximize(init_points=init_points, n_iter=n_iter)
        t_total = time.time() - t0

        best_params = bo.max['params']
        best_gene   = [max(0, min(N_OPS - 1, int(round(best_params[f'e{j}']))))
                       for j in range(N_EDGES)]
        best_point = scoredPoint(api, best_gene, dataset, hp,
                                 flops_min, flops_max, alpha=alpha)
        best_point['fitness'] = round(
            alpha * (1.0 - best_point['valid_acc'])
            + (1 - alpha) * best_point['norm_flops'], 6)

        print(f"  [BO  α={alpha:.1f}  seed={run_seed}] "
              f"valid={best_point['valid_acc']:.4f}  "
              f"test={best_point['test_acc']:.4f}  "
              f"flops={best_point['flops_M']:.1f}M  t={t_total:.0f}s")
        all_points.append(best_point)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_results = {
            'algorithm':   'BO',
            'benchmark':   'NAS-Bench-201',
            'timestamp':   datetime.now().isoformat(),
            'config': {
                'dataset': dataset, 'hp': hp, 'alpha': alpha,
                'n_iter': n_iter, 'init_points': init_points,
                'xi': xi, 'seed': run_seed,
                'flops_min': flops_min, 'flops_max': flops_max,
                'ref_point': list(REF_POINT),
            },
            'best':        best_point,
            'hypervolume': round(computeHypervolume([[best_point['f1'], best_point['f2']]]), 8),
            'hv_history':  [],
            'n_evaluations':   counter['n'],
            'n_unique_architectures': len({e['arch_index'] for e in eval_log}),
            'total_time_s': round(t_total, 1),
            'eval_log':    eval_log,
        }
        json_path = os.path.join(
            results_dir, f'BO_a{int(alpha*10):02d}_{ts}_seed{run_seed}.json')
        with open(json_path, 'w') as fh:
            json.dump(run_results, fh, indent=2)
        pd.DataFrame(eval_log).to_csv(
            json_path.replace('.json', '_evals.csv'), index=False)

    # Non-dominated front from all alpha runs
    F      = np.array([[p['f1'], p['f2']] for p in all_points])
    nd_idx = getNonDominated(F)
    front  = sorted([all_points[i] for i in nd_idx], key=lambda p: p['f1'])
    hv     = computeHypervolume([[p['f1'], p['f2']] for p in front])

    print(f"\n  BO sweep done — |front|={len(front)}  HV={hv:.6f}")
    gc.collect()   # the run's eval log is dead by here; reclaim it
    return front, all_points


# ─────────────────────────────────────────────────────────────────────────────
# Runner: qEHVI  (native multi-objective Bayesian Optimisation, BoTorch)
# ─────────────────────────────────────────────────────────────────────────────

def runQEHVINASSearch(
    api,
    dataset       = 'cifar10',
    hp            = '200',
    n_iter        = 100,
    init_points   = 20,
    batch_size    = 1,
    mc_samples    = 128,
    num_restarts  = 10,
    raw_samples   = 128,
    seed          = None,
    flops_min     = None,
    flops_max     = None,
    results_dir   = 'results/mo_nas/qehvi',
    verbose       = False,
):
    """Run qEHVI on NAS-Bench-201 with objectives (1-acc, norm_flops).

    qEHVI — q-Expected Hypervolume Improvement (Daulton et al., 2020) — is the
    native multi-objective Bayesian-optimisation baseline used in the LaMOO
    paper. A Gaussian-process surrogate is fit to the observed (accuracy,
    FLOPs) trade-offs; each iteration the qEHVI acquisition proposes the
    architecture(s) expected to enlarge the dominated hypervolume the most,
    balancing exploration of the GP's uncertainty against exploitation of the
    current Pareto front. Like NSGA-II / GDE3 it returns a single
    non-dominated front per seed (no alpha sweep).

    Requires BoTorch + PyTorch (``pip install botorch``). The import is lazy
    so the rest of ``nas_mo`` keeps working without these heavy dependencies.

    Encoding
    --------
    The 6 edge operations are relaxed to a continuous unit cube [0, 1]^6 for
    the GP and acquisition optimisation, then mapped to {0, …, N_OPS-1} by
    ``round(x * (N_OPS-1))`` when querying the benchmark. This continuous
    relaxation matches the one used by NSGA-II, GDE3 and the scalarised BO, so
    all methods remain directly comparable on hypervolume.

    Budget
    ------
    Total queries = ``init_points + n_iter * batch_size``. To match the EA
    budget of ``(n_generations+1) * pop_size`` (= 1020 in the notebook), use
    e.g. ``init_points=20, n_iter=200, batch_size=5``. qEHVI refits a GP every
    iteration (cost grows with the number of observed points), so spending
    budget through a larger ``batch_size`` (q) is much cheaper than through
    more iterations.

    Returns
    -------
    (pareto_front, run_results)  — same schema as :func:`runNSGAIINASSearch`.
    """
    try:
        import torch
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls.exact_marginal_log_likelihood import (
            ExactMarginalLogLikelihood,
        )
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            FastNondominatedPartitioning,
        )
        from botorch.acquisition.multi_objective.monte_carlo import (
            qExpectedHypervolumeImprovement,
        )
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.optim.optimize import optimize_acqf
        from botorch.utils.sampling import draw_sobol_samples
    except ImportError as e:
        raise ImportError(
            "runQEHVINASSearch requires BoTorch + PyTorch. "
            "Install with:  pip install botorch\n"
            f"  (original import error: {e})"
        )

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if flops_min is None or flops_max is None:
        flops_min, flops_max = computeFlopsRange(api, dataset, hp, verbose=True)

    total_evals = init_points + n_iter * batch_size
    print(f"\n{'='*62}")
    print(f"  MO-NAS: qEHVI  ({dataset}, hp={hp})")
    print(f"  Init: {init_points}  |  BO iters: {n_iter} x q={batch_size}  |  "
          f"Total queries: {total_evals}  |  Seed: {seed}")
    print(f"{'='*62}\n")

    tkwargs = {'dtype': torch.double, 'device': torch.device('cpu')}

    # Unit-cube bounds for the GP / acquisition optimisation.
    std_bounds = torch.stack([
        torch.zeros(N_EDGES, **tkwargs),
        torch.ones(N_EDGES, **tkwargs),
    ])

    # qEHVI MAXIMISES objectives; ours are minimised, so we optimise
    #   Y = [-(1-acc), -norm_flops]
    # and place the reference point below every attainable value. REF_POINT is
    # (1.1, 1.1) in the minimisation (f1, f2) space → (-1.1, -1.1) here.
    ref_point = torch.tensor([-float(REF_POINT[0]), -float(REF_POINT[1])],
                             **tkwargs)

    eval_log     = []
    counter      = {'n': 0, 'start': time.time()}
    hv_history   = []
    hv_indicator = HV(ref_point=REF_POINT)
    all_min_objs = []   # [[f1, f2], …] VALIDATION — drives the GP and the front
    all_test_objs = []  # [[f1, f2], …] TEST — what the HV trace reports
    all_records  = []   # parallel full point dicts

    def _unit_to_gene(x_unit):
        g = [int(round(float(v) * (N_OPS - 1))) for v in x_unit]
        return [max(0, min(N_OPS - 1, v)) for v in g]

    def _evaluate_unit(X_unit):
        """Evaluate a (q, N_EDGES) unit-cube tensor → (q, 2) maximisation tensor."""
        Y = torch.zeros((X_unit.shape[0], 2), **tkwargs)
        for i in range(X_unit.shape[0]):
            gene = _unit_to_gene(X_unit[i].tolist())
            try:
                r     = queryValidTest(api, gene, dataset=dataset, hp=hp)
                acc   = r['valid_acc'] or 0.0   # SEARCH SIGNAL: validation
                flops = r['flops_M']   or 0.0
            except Exception as ex:
                print(f"  [ERROR] query failed: {ex}")
                r = {'index': -1, 'arch_str': '?', 'valid_acc': 0.0,
                     'test_acc': 0.0}
                acc, flops = 0.0, 0.0

            nf     = _normFlops(flops, flops_min, flops_max)
            f1, f2 = 1.0 - acc, nf
            Y[i, 0], Y[i, 1] = -f1, -f2

            counter['n'] += 1
            elapsed = time.time() - counter['start']
            eval_log.append({
                'eval':       counter['n'],
                'arch_index': r.get('index', -1),
                'arch_str':   r.get('arch_str', '?'),
                'gene':       ' '.join(str(g) for g in gene),
                'valid_acc':  round(acc, 6),
                'test_acc':   round(r.get('test_acc', 0.0), 6),
                'flops_M':    round(flops, 4),
                'norm_flops': round(nf, 6),
                'f1':         round(f1, 6),
                'f2':         round(f2, 6),
                'alpha':      None,
                'fitness':    None,
                'elapsed_s':  round(elapsed, 3),
            })
            all_min_objs.append([f1, f2])
            all_test_objs.append([1.0 - (r.get('test_acc') or 0.0), nf])
            all_records.append({
                'gene':       gene,
                'arch_index': r.get('index', -1),
                'arch_str':   r.get('arch_str', '?'),
                'valid_acc':  round(acc, 6),
                'test_acc':   round(r.get('test_acc', 0.0), 6),
                'flops_M':    round(flops, 4),
                'norm_flops': round(nf, 6),
                # reported objectives on TEST; f1_valid is what the GP saw
                'f1':         round(1.0 - r.get('test_acc', 0.0), 6),
                'f2':         round(nf, 6),
                'f1_valid':   round(f1, 6),
            })
            if verbose:
                print(f"  [{counter['n']:>4}] gene={gene}  "
                      f"valid_acc={acc:.4f}  flops={flops:.1f}M")
        return Y

    # ── Initial design (Sobol) ────────────────────────────────────────────
    train_x   = draw_sobol_samples(bounds=std_bounds, n=init_points, q=1, seed=seed).squeeze(1).to(**tkwargs)
    train_obj = _evaluate_unit(train_x)
    hv_history.append(_hvOfArchive(all_test_objs))

    # ── BO loop ───────────────────────────────────────────────────────────
    log_every = max(1, n_iter // 10)
    for it in range(n_iter):
        # Fit a GP on the standardised maximisation objectives.
        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=2))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as ex:
            print(f"  [WARN] GP fit failed at iter {it}: {ex}")

        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_obj)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=std_bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={'batch_limit': 5, 'maxiter': 200},
            sequential=True,
        )

        new_obj   = _evaluate_unit(candidates.detach())
        train_x   = torch.cat([train_x, candidates.detach().to(**tkwargs)], dim=0)
        train_obj = torch.cat([train_obj, new_obj], dim=0)

        cur_hv = _hvOfArchive(all_test_objs)
        hv_history.append(cur_hv)
        if (it + 1) % log_every == 0 or it == n_iter - 1:
            print(f"  iter {it + 1:>3}/{n_iter}  evals={counter['n']:>4}  "
                  f"HV={cur_hv:.6f}")

    # ── Final Pareto front over ALL evaluated architectures ───────────────
    F      = np.asarray(all_min_objs)
    nd_idx = getNonDominated(F)
    pareto_front = sorted([all_records[i] for i in nd_idx], key=lambda p: p['f1'])
    final_hv = (computeHypervolume([[p['f1'], p['f2']] for p in pareto_front])
                if pareto_front else 0.0)
    t_total = time.time() - counter['start']

    print(f"\n{'='*62}")
    print(f"  FINAL — qEHVI  ({dataset})")
    print(f"  |Pareto front|  : {len(pareto_front)} architectures")
    print(f"  Hypervolume     : {final_hv:.6f}")
    if pareto_front:
        print(f"  Best accuracy   : {max(p['test_acc'] for p in pareto_front):.4f}")
        print(f"  Min FLOPs       : {min(p['flops_M'] for p in pareto_front):.1f} M")
    print(f"  Total evals     : {counter['n']}")
    print(f"  Wallclock       : {t_total:.1f}s\n")

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_results = {
        'algorithm':   'qEHVI',
        'benchmark':   'NAS-Bench-201',
        'timestamp':   datetime.now().isoformat(),
        'config': {
            'dataset':      dataset,
            'hp':           hp,
            'n_iter':       n_iter,
            'init_points':  init_points,
            'batch_size':   batch_size,
            'mc_samples':   mc_samples,
            'num_restarts': num_restarts,
            'raw_samples':  raw_samples,
            'seed':         seed,
            'flops_min':    flops_min,
            'flops_max':    flops_max,
            'ref_point':    list(REF_POINT),
        },
        'pareto_front':           pareto_front,
        'hypervolume':            round(final_hv, 8),
        'hv_history':             [round(h, 8) for h in hv_history],
        'n_evaluations':          counter['n'],
        'n_unique_architectures': len({e['arch_index'] for e in eval_log}),
        'total_time_s':           round(t_total, 1),
        'eval_log':               eval_log,
    }

    json_path = os.path.join(results_dir, f'qEHVI_{ts}_seed{seed}.json')
    with open(json_path, 'w') as f:
        json.dump(run_results, f, indent=2)
    pd.DataFrame(eval_log).to_csv(
        os.path.join(results_dir, f'qEHVI_{ts}_seed{seed}_evals.csv'),
        index=False)

    print(f"  Saved: {json_path}")
    gc.collect()   # the run's eval log is dead by here; reclaim it
    return pareto_front, run_results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load saved results and rebuild front
# ─────────────────────────────────────────────────────────────────────────────

def loadFrontFromDir(results_dir):
    """Load all JSON result files from a directory and return the aggregate
    non-dominated front.

    Works for both NSGA-II (pareto_front key) and scalarised (best key) JSON.

    Returns
    -------
    (front, all_points, hv)
    """
    all_points = []
    for p in sorted(Path(results_dir).glob('*.json')):
        with open(p) as f:
            d = json.load(f)
        if 'pareto_front' in d:
            all_points.extend(d['pareto_front'])
        elif 'best' in d:
            all_points.append(d['best'])

    if not all_points:
        return [], [], 0.0

    F      = np.array([[pt['f1'], pt['f2']] for pt in all_points])
    nd_idx = getNonDominated(F)
    front  = sorted([all_points[i] for i in nd_idx], key=lambda p: p['f1'])
    hv     = computeHypervolume([[p['f1'], p['f2']] for p in front])
    return front, all_points, hv
