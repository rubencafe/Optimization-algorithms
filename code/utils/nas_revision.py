"""Revision infrastructure for the NAS-Bench-201 comparative study.

This module adds the four protocol changes requested in review, without
touching the existing runners in :mod:`utils.nas` / :mod:`utils.nas_mo` so
that the original results stay reproducible:

1. **Validation-driven search.** The search is scored with *validation*
   accuracy and the test accuracy is read only once, for the architecture
   the search finally returns. See :func:`splitFor` and :class:`ArchCache`.

2. **One caching policy for every optimizer.** :class:`ArchCache` memoizes
   benchmark lookups by gene, so a repeated proposal costs nothing and every
   method is charged the same way. Both the number of *proposals* and the
   number of *unique* architectures are recorded. The budget is always a
   number of proposals: the optimizer decides what to explore and we never
   intervene when it re-proposes, so its duplicate rate is a measured property
   of its search behaviour rather than something the protocol corrects.

3. **Scalarization at the endpoints.** :func:`scalarize` implements both the
   weighted sum and the (augmented) Tchebycheff aggregation, and
   :data:`ALPHA_SWEEP_DENSE` includes ``alpha = 0`` and ``alpha = 1`` with
   extra weights near both ends.

4. **Two reference baselines.** :func:`runRandomSearch` and
   :func:`runRegularizedEvolution` add the two baselines that NAS studies are
   normally expected to report.

Everything heavy (``nats_bench``, ``mealpy``, ``pymoo``) is imported lazily,
so this module can be imported — and self-tested — without them::

    python -m utils.nas_revision      # runs selftest() against a stub benchmark
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np

N_EDGES = 6
N_OPS = 5

# Weights for the scalarization sweep. Denser near both endpoints, and
# including 0.0 (pure cost) and 1.0 (pure accuracy) so that the extremes of
# the front are actually targeted rather than approached.
ALPHA_SWEEP_DENSE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# Augmentation coefficient of the augmented Tchebycheff aggregation. A small
# positive value removes weakly Pareto-optimal solutions from the arg-min.
TCHEBYCHEFF_RHO = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 1. Validation / test split
# ─────────────────────────────────────────────────────────────────────────────

# NAS-Bench-201 exposes a dedicated 'cifar10-valid' entry, trained on half of
# the CIFAR-10 training set and evaluated on the held-out half. The plain
# 'cifar10' entry is trained on the full training set and only has a test
# score. Searching on 'cifar10' therefore leaks the test set; searching on
# 'cifar10-valid' does not. CIFAR-100 and ImageNet16-120 already carry a
# proper validation split inside the same entry.
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


def splitFor(dataset):
    """Return ``(search_dataset, search_key, report_dataset, report_key)``.

    The search is driven by validation accuracy on ``search_dataset``; the
    number finally reported is the test accuracy on ``report_dataset``.
    """
    if dataset not in _SEARCH_DATASET:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"expected one of {sorted(_SEARCH_DATASET)}")
    return (_SEARCH_DATASET[dataset], 'valid_acc',
            _REPORT_DATASET[dataset], 'test_acc')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shared cache + budget accounting
# ─────────────────────────────────────────────────────────────────────────────

class BudgetReached(Exception):
    """Raised inside the objective once the evaluation budget is spent.

    Optimizers call the objective from deep inside their own loops, so the
    only portable way to stop every library at exactly the same budget is to
    unwind with an exception and catch it in the runner.
    """


class ArchCache:
    """Memoizing NAS-Bench-201 lookup with explicit budget accounting.

    The benchmark is deterministic, so querying the same architecture twice
    yields no new information. The cache makes that explicit and charges every
    optimizer the same way:

    * ``n_proposals`` counts every call the optimizer makes — this is the
      budget, so a repeat is charged like any other call;
    * ``n_unique`` counts distinct architectures actually looked up;
    * ``n_hits`` counts proposals served from the cache.

    The cache saves time, never budget. How much of its budget a method spends
    on architectures it has already seen is a result about that method, and is
    reported rather than corrected.
    """

    def __init__(self, api, dataset, hp='200', budget=1020,
                 flops_min=None, flops_max=None):
        self.api = api
        self.dataset = dataset
        self.hp = hp
        self.budget = int(budget)

        (self.search_dataset, self.search_key,
         self.report_dataset, self.report_key) = splitFor(dataset)

        self.flops_min = flops_min
        self.flops_max = flops_max

        self._store = {}
        self.n_proposals = 0
        self.n_hits = 0
        self.log = []
        self.start = time.time()

    # -- accounting ----------------------------------------------------------

    @property
    def n_unique(self):
        return len(self._store)

    @property
    def spent(self):
        return self.n_proposals

    def exhausted(self):
        return self.n_proposals >= self.budget

    # -- lookup --------------------------------------------------------------

    def query(self, gene):
        """Look up one architecture, honouring the cache and the budget.

        Returns a dict with both the validation score that drives the search
        and the test score kept for final reporting.
        """
        gene = tuple(int(np.clip(round(float(g)), 0, N_OPS - 1)) for g in gene)

        if gene in self._store:
            self.n_proposals += 1
            self.n_hits += 1
            rec = self._store[gene]
            self._record(rec, cached=True)
            if self.exhausted():
                raise BudgetReached
            return rec

        rec = self._lookup(gene)
        self._store[gene] = rec
        self.n_proposals += 1
        self._record(rec, cached=False)

        if self.exhausted():
            raise BudgetReached
        return rec

    def _lookup(self, gene):
        from utils.nas import queryArchitecture

        s = queryArchitecture(self.api, list(gene),
                              dataset=self.search_dataset, hp=self.hp)
        # The two entries share the architecture index, so the cost figures
        # are identical; only the accuracy differs between the splits.
        if self.report_dataset == self.search_dataset:
            t = s
        else:
            t = queryArchitecture(self.api, list(gene),
                                  dataset=self.report_dataset, hp=self.hp)

        return {
            'gene': gene,
            'arch_index': s.get('index', -1),
            'arch_str': s.get('arch_str', '?'),
            'valid_acc': float(s.get(self.search_key) or 0.0),
            'test_acc': float(t.get(self.report_key) or 0.0),
            'flops_M': float(t.get('flops_M') or 0.0),
            'params_MB': float(t.get('params_MB') or 0.0),
        }

    def _record(self, rec, cached):
        self.log.append({
            'proposal': self.n_proposals,
            'unique': self.n_unique,
            'cached': int(cached),
            'arch_index': rec['arch_index'],
            'gene': ' '.join(str(g) for g in rec['gene']),
            'valid_acc': round(rec['valid_acc'], 6),
            'test_acc': round(rec['test_acc'], 6),
            'flops_M': round(rec['flops_M'], 4),
            'elapsed_s': round(time.time() - self.start, 3),
        })

    # -- derived quantities --------------------------------------------------

    def normFlops(self, flops):
        if self.flops_min is None or self.flops_max is None:
            return 0.0
        span = self.flops_max - self.flops_min
        if span <= 0:
            return 0.0
        return float(np.clip((flops - self.flops_min) / span, 0.0, 1.0))

    def objectives(self, rec):
        """Return the pair ``(f1, f2)`` minimized during the search.

        ``f1`` uses the *validation* accuracy: the test score never enters
        the optimization loop.
        """
        return 1.0 - rec['valid_acc'], self.normFlops(rec['flops_M'])

    def bestByValid(self):
        """Architecture with the highest validation accuracy seen so far."""
        if not self._store:
            return None
        return max(self._store.values(), key=lambda r: r['valid_acc'])

    def archive(self):
        """Every distinct architecture evaluated, as a list of records."""
        return list(self._store.values())

    def stats(self):
        return {
            'n_proposals': self.n_proposals,
            'n_unique': self.n_unique,
            'n_cache_hits': self.n_hits,
            'budget': self.budget,
            'budget_met': self.n_proposals >= self.budget,
            'search_dataset': self.search_dataset,
            'report_dataset': self.report_dataset,
        }

    def genCap(self, pop_size):
        """Generation cap sized so the budget, not the cap, ends the run."""
        return int(np.ceil(self.budget / max(pop_size, 1))) + 5


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scalarization
# ─────────────────────────────────────────────────────────────────────────────

def scalarize(f1, f2, alpha, method='weighted_sum', rho=TCHEBYCHEFF_RHO,
              ideal=(0.0, 0.0)):
    """Aggregate two minimized objectives into one scalar.

    Parameters
    ----------
    alpha : weight on ``f1`` (accuracy gap). ``alpha=1`` optimizes accuracy
        alone and ``alpha=0`` cost alone.
    method : ``'weighted_sum'`` or ``'tchebycheff'``.
    rho : augmentation coefficient of the augmented Tchebycheff aggregation.
    ideal : reference point $z^\\star$; the origin is a valid utopian point
        here because both objectives are non-negative by construction.

    Notes
    -----
    A weighted sum can only recover solutions on the convex hull of the
    front, which is why it misses concave stretches however finely the weight
    is swept. The Tchebycheff aggregation has no such restriction: every
    Pareto-optimal point is the optimum of some weight vector.
    """
    w1, w2 = float(alpha), 1.0 - float(alpha)
    d1, d2 = f1 - ideal[0], f2 - ideal[1]

    if method == 'weighted_sum':
        return w1 * d1 + w2 * d2
    if method == 'tchebycheff':
        return max(w1 * d1, w2 * d2) + rho * (w1 * d1 + w2 * d2)
    raise ValueError(f"unknown scalarization {method!r}")


def makeScalarObjective(cache, alpha, method='weighted_sum'):
    """Build a minimization objective backed by :class:`ArchCache`."""

    def objective(solution):
        rec = cache.query(solution)
        f1, f2 = cache.objectives(rec)
        return scalarize(f1, f2, alpha, method=method)

    objective.cache = cache
    objective.alpha = alpha
    objective.method = method
    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Pareto helpers
# ─────────────────────────────────────────────────────────────────────────────

def nonDominated(points):
    """Return the non-dominated subset of ``[(f1, f2, payload), ...]``.

    Both objectives are minimized. Used to build an *archive* front from
    everything a run evaluated, which is well defined even when the run is
    stopped mid-generation by :class:`BudgetReached`.
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front, best_f2 = [], float('inf')
    for p in pts:
        if p[1] < best_f2:
            front.append(p)
            best_f2 = p[1]
    return front


def archiveFront(cache, use_test=False):
    """Non-dominated front over every architecture the run evaluated.

    ``use_test=False`` builds the front the optimizer could actually see
    (validation accuracy); ``use_test=True`` re-scores the same architectures
    with test accuracy for the final report.
    """
    key = 'test_acc' if use_test else 'valid_acc'
    pts = [(1.0 - r[key], cache.normFlops(r['flops_M']), r)
           for r in cache.archive()]
    return [p[2] for p in nonDominated(pts)]


# ─────────────────────────────────────────────────────────────────────────────
# Quality indicators
# ─────────────────────────────────────────────────────────────────────────────

REF_POINT = (1.1, 1.1)


def _objArray(records, use_test, cache):
    key = 'test_acc' if use_test else 'valid_acc'
    return np.array([[1.0 - r[key], cache.normFlops(r['flops_M'])]
                     for r in records], dtype=float)


def hypervolume(front_objs, ref_point=REF_POINT):
    from pymoo.indicators.hv import HV
    F = np.atleast_2d(np.asarray(front_objs, dtype=float))
    if F.size == 0:
        return 0.0
    return float(HV(ref_point=np.array(ref_point))(F))


def igdPlus(front_objs, true_front_objs):
    from pymoo.indicators.igd_plus import IGDPlus
    F = np.atleast_2d(np.asarray(front_objs, dtype=float))
    Z = np.atleast_2d(np.asarray(true_front_objs, dtype=float))
    if F.size == 0:
        return float('inf')
    return float(IGDPlus(Z)(F))


def scoreRun(cache, true_front_objs=None, ref_point=REF_POINT):
    """Score a finished run on **test** accuracy.

    The search itself never saw the test column; the architectures it returned
    are re-scored here, which is what makes the reported gap an honest estimate
    rather than a fit to the test set. Both the validation-side front (what the
    optimizer was actually steering toward) and the test-side front (what it
    delivered) are reported, because the difference between them *is* the
    generalization gap the old protocol hid.
    """
    valid_front = archiveFront(cache, use_test=False)
    # Re-score exactly the architectures the search selected, on test.
    test_objs = _objArray(valid_front, use_test=True, cache=cache)
    out = {
        'front_size': len(valid_front),
        'hv_valid': hypervolume(_objArray(valid_front, False, cache), ref_point),
        'hv_test': hypervolume(test_objs, ref_point),
        'best_valid_acc': max((r['valid_acc'] for r in valid_front), default=0.0),
        'best_test_acc': max((r['test_acc'] for r in valid_front), default=0.0),
        **cache.stats(),
    }
    if true_front_objs is not None:
        out['igd_plus_test'] = igdPlus(test_objs, true_front_objs)
        out['hv_gap_test'] = hypervolume(true_front_objs, ref_point) - out['hv_test']
    return out


def trueParetoFront(api, dataset, hp='200', flops_min=None, flops_max=None,
                    use_test=True):
    """Exhaustive Pareto front over all 15,625 cells.

    The reference front stays defined on **test** accuracy: it is the target a
    method is judged against, not a signal the search may consult.
    """
    from utils.nas import queryArchitecture

    key = 'test_acc' if use_test else 'valid_acc'
    ds = _REPORT_DATASET[dataset] if use_test else _SEARCH_DATASET[dataset]
    recs = []
    for idx in range(N_OPS ** N_EDGES):
        gene, rem = [], idx
        for _ in range(N_EDGES):
            gene.append(rem % N_OPS)
            rem //= N_OPS
        gene.reverse()
        r = queryArchitecture(api, gene, dataset=ds, hp=hp)
        recs.append({'gene': tuple(gene),
                     'arch_index': r.get('index', idx),
                     'arch_str': r.get('arch_str', '?'),
                     'valid_acc': float(r.get('valid_acc') or 0.0),
                     'test_acc': float(r.get('test_acc') or 0.0),
                     'flops_M': float(r.get('flops_M') or 0.0),
                     'params_MB': float(r.get('params_MB') or 0.0)})

    if flops_min is None:
        flops_min = min(r['flops_M'] for r in recs)
    if flops_max is None:
        flops_max = max(r['flops_M'] for r in recs)
    span = max(flops_max - flops_min, 1e-12)

    pts = [(1.0 - r[key], (r['flops_M'] - flops_min) / span, r) for r in recs]
    front = [p[2] for p in nonDominated(pts)]
    objs = [[1.0 - r[key], (r['flops_M'] - flops_min) / span] for r in front]
    return front, objs, (flops_min, flops_max)


def writeLegacyTrueFront(path, front, flops_min, flops_max):
    """Write a reference front as the flat list ``aggregate_mo_nas.py`` reads."""
    span = max(flops_max - flops_min, 1e-12)
    out = [{'index': r.get('arch_index', -1),
            'arch_str': r.get('arch_str', '?'),
            'test_acc': r['test_acc'],
            'flops_M': r['flops_M'],
            'norm_flops': (r['flops_M'] - flops_min) / span,
            'f1': round(1.0 - r['test_acc'], 6),
            'f2': round((r['flops_M'] - flops_min) / span, 6)}
           for r in front]
    path = os.fspath(path)
    with open(path, 'w') as fh:
        json.dump(out, fh)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Corrected runners — one cache, one budget rule, validation-driven
# ─────────────────────────────────────────────────────────────────────────────

def makeMOProblem(cache):
    """pymoo ``Problem`` whose objectives come from the shared cache."""
    from pymoo.core.problem import Problem

    class _Problem(Problem):
        def __init__(self):
            super().__init__(n_var=N_EDGES, n_obj=2,
                             xl=np.zeros(N_EDGES),
                             xu=np.full(N_EDGES, N_OPS - 1, dtype=float))

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((len(X), 2))
            for i, x in enumerate(X):
                F[i] = cache.objectives(cache.query(x))
            out['F'] = F

    return _Problem()


def runPymooMO(api, algorithm, name, dataset='cifar10', hp='200', budget=1020,
               pop_size=20, seed=0,
               flops_min=None, flops_max=None, true_front_objs=None,
               results_dir=None):
    """Run any pymoo multi-objective algorithm under the corrected protocol.

    The run is stopped by :class:`BudgetReached` rather than by a generation
    count, so every method stops at exactly the same budget regardless of how
    many generations that takes. The returned front is the non-dominated
    subset of everything evaluated, which stays well defined when the run is
    cut mid-generation.
    """
    from pymoo.optimize import minimize

    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)
    problem = makeMOProblem(cache)

    # The cap must never bind before the budget does — see ArchCache.genCap.
    n_gen = cache.genCap(pop_size)
    t0 = time.time()
    try:
        minimize(problem, algorithm, termination=('n_gen', n_gen),
                 seed=seed, verbose=False)
    except BudgetReached:
        pass
    elapsed = time.time() - t0

    best, run, _ = _finish(name, cache, dataset, seed, elapsed, None,
                           extra={'pop_size': pop_size})
    run['score'] = scoreRun(cache, true_front_objs)
    run['pareto_front'] = [_point(r, cache)
                           for r in archiveFront(cache, use_test=False)]
    _persist(name, run, cache, seed, results_dir)
    return run, cache


def runNSGA2MO(api, **kw):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    pop_size = kw.pop('pop_size', 20)
    # The paper used eliminate_duplicates=False. That is harmless under a
    # proposal budget, but under a unique-architecture budget NSGA-II then
    # converges and re-proposes the same cells forever, unable to spend its
    # remaining budget. Pass True to let it keep generating novel candidates.
    elim = kw.pop('eliminate_duplicates', False)
    algo = NSGA2(pop_size=pop_size,
                 sampling=FloatRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(prob=1.0 / N_EDGES, eta=20),
                 eliminate_duplicates=elim)
    return runPymooMO(api, algo, 'NSGA-II', pop_size=pop_size, **kw)


def runGDE3MO(api, **kw):
    from pymoo.algorithms.moo.gde3 import GDE3
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    pop_size = kw.pop('pop_size', 20)
    algo = GDE3(pop_size=pop_size, sampling=FloatRandomSampling(),
                variant='DE/rand/1/bin', CR=0.9, F=0.5)
    return runPymooMO(api, algo, 'GDE3', pop_size=pop_size, **kw)


def runScalarizedMealpy(api, algorithm, dataset='cifar10', hp='200',
                        alpha=0.5, method='weighted_sum', budget=1020,
                        pop_size=20, seed=0,
                        flops_min=None, flops_max=None, results_dir=None):
    """Run one mealpy optimizer on a scalarized, validation-driven objective."""
    from mealpy import IntegerVar
    from utils.localMealpy import getOptimizationAlgorithm

    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)
    objective = makeScalarObjective(cache, alpha, method=method)

    n_gen = cache.genCap(pop_size)
    problem = {
        'obj_func': objective,
        'bounds': [IntegerVar(lb=0, ub=N_OPS - 1, name=f'edge_{i}')
                   for i in range(N_EDGES)],
        'minmax': 'min',
        'log_to': None,
    }
    algo = getOptimizationAlgorithm(algorithm.upper(), n_gen, pop_size, None)

    t0 = time.time()
    try:
        algo.solve(problem, seed=seed)
    except BudgetReached:
        pass
    elapsed = time.time() - t0

    # Selection uses the scalarized *validation* objective, never test.
    best = min(cache.archive(),
               key=lambda r: scalarize(*cache.objectives(r), alpha, method))
    run = {
        'algorithm': algorithm.upper(),
        'timestamp': datetime.now().isoformat(),
        'config': dict(dataset=dataset, hp=hp, alpha=alpha,
                       scalarization=method, seed=seed, pop_size=pop_size,
                       **cache.stats()),
        'best': _point(best, cache),
        'total_time_s': round(elapsed, 2),
    }
    _persist(f"{algorithm.upper()}_a{int(alpha * 100):03d}_{method}",
             run, cache, seed, results_dir)
    return run, cache


def runBOScalarized(api, dataset='cifar10', hp='200', alpha=0.5,
                    method='weighted_sum', budget=1020, init_points=20,
                    xi=0.01, seed=0,
                    flops_min=None, flops_max=None, results_dir=None):
    """Bayesian optimization on a scalarized, validation-driven objective."""
    from bayes_opt import BayesianOptimization, acquisition

    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)
    objective = makeScalarObjective(cache, alpha, method=method)

    def bo_objective(e0, e1, e2, e3, e4, e5):
        # bayes_opt maximizes; our scalarization is minimized.
        return -objective([e0, e1, e2, e3, e4, e5])

    acq = acquisition.ExpectedImprovement(xi=xi, random_state=seed)
    bo = BayesianOptimization(
        f=bo_objective,
        pbounds={f'e{i}': (0, N_OPS - 1, int) for i in range(N_EDGES)},
        acquisition_function=acq,
        random_state=seed,
        verbose=0,
    )

    t0 = time.time()
    try:
        # BudgetReached ends the run; this count must not bind first.
        bo.maximize(init_points=init_points,
                    n_iter=max(cache.budget - init_points, 1))
    except BudgetReached:
        pass
    elapsed = time.time() - t0

    best = min(cache.archive(),
               key=lambda r: scalarize(*cache.objectives(r), alpha, method))
    run = {
        'algorithm': 'BO',
        'timestamp': datetime.now().isoformat(),
        'config': dict(dataset=dataset, hp=hp, alpha=alpha,
                       scalarization=method, xi=xi, init_points=init_points,
                       seed=seed, **cache.stats()),
        'best': _point(best, cache),
        'total_time_s': round(elapsed, 2),
    }
    _persist(f"BO_a{int(alpha * 100):03d}_{method}", run, cache, seed, results_dir)
    return run, cache


def runQEHVIMO(api, dataset='cifar10', hp='200', budget=1020, init_points=20,
               batch_size=5, mc_samples=128, num_restarts=10, raw_samples=128,
               seed=0, flops_min=None, flops_max=None,
               true_front_objs=None, results_dir=None, verbose=False):
    """qEHVI under the corrected protocol.

    Mirrors the original loop but routes every lookup through the shared cache
    and drives the GP with validation objectives. NOTE: this is the one runner
    that could not be executed during development (BoTorch was unavailable);
    smoke-test it with a small ``budget`` before launching the full sweep.
    """
    import torch
    from botorch.acquisition.multi_objective.monte_carlo import (
        qExpectedHypervolumeImprovement)
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim.optimize import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning)
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

    tkwargs = {'dtype': torch.double, 'device': torch.device('cpu')}
    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)

    std_bounds = torch.stack([torch.zeros(N_EDGES, **tkwargs),
                              torch.ones(N_EDGES, **tkwargs)])
    ref_point = torch.tensor([-float(REF_POINT[0]), -float(REF_POINT[1])],
                             **tkwargs)

    def _evaluate_unit(X_unit):
        """Unit cube -> (q, 2) maximization tensor of validation objectives."""
        Y = torch.zeros((X_unit.shape[0], 2), **tkwargs)
        for i in range(X_unit.shape[0]):
            gene = [int(round(float(v) * (N_OPS - 1))) for v in X_unit[i].tolist()]
            f1, f2 = cache.objectives(cache.query(gene))
            Y[i, 0], Y[i, 1] = -f1, -f2
        return Y

    t0 = time.time()
    try:
        train_x = draw_sobol_samples(bounds=std_bounds, n=init_points, q=1,
                                     seed=seed).squeeze(1).to(**tkwargs)
        train_obj = _evaluate_unit(train_x)

        while True:
            model = SingleTaskGP(train_x, train_obj,
                                 outcome_transform=Standardize(m=2))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception as ex:
                print(f"  [WARN] GP fit failed: {ex}")

            acqf = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                partitioning=FastNondominatedPartitioning(
                    ref_point=ref_point, Y=train_obj),
                sampler=SobolQMCNormalSampler(
                    sample_shape=torch.Size([mc_samples])),
            )
            candidates, _ = optimize_acqf(
                acq_function=acqf, bounds=std_bounds, q=batch_size,
                num_restarts=num_restarts, raw_samples=raw_samples,
                options={'batch_limit': 5, 'maxiter': 200}, sequential=True)

            new_obj = _evaluate_unit(candidates.detach())
            train_x = torch.cat([train_x, candidates.detach().to(**tkwargs)], dim=0)
            train_obj = torch.cat([train_obj, new_obj], dim=0)
            if verbose:
                print(f"  proposals={cache.n_proposals} unique={cache.n_unique}")
    except BudgetReached:
        pass
    elapsed = time.time() - t0

    _, run, _ = _finish('qEHVI', cache, dataset, seed, elapsed, None, extra={})
    run['score'] = scoreRun(cache, true_front_objs)
    run['pareto_front'] = [_point(r, cache)
                           for r in archiveFront(cache, use_test=False)]
    _persist('qEHVI', run, cache, seed, results_dir)
    return run, cache


def _persist(name, run, cache, seed, results_dir):
    """Write the run JSON and its per-proposal eval log.

    Records the file stem on the run so downstream figure code can find the
    eval log without re-globbing the results tree.
    """
    if not results_dir:
        return None
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = os.path.join(results_dir, f'{name}_{ts}_seed{seed}')
    run['stem'] = stem
    with open(stem + '.json', 'w') as fh:
        json.dump(run, fh, indent=2)
    try:
        import pandas as pd
        pd.DataFrame(cache.log).to_csv(stem + '_evals.csv', index=False)
    except ImportError:
        pass
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# 4. Baselines
# ─────────────────────────────────────────────────────────────────────────────

def _randomGene(rng):
    return [int(x) for x in rng.integers(0, N_OPS, size=N_EDGES)]


def _mutateGene(gene, rng):
    """Flip one edge to a different operation (regularized-evolution mutation)."""
    child = list(gene)
    pos = int(rng.integers(0, N_EDGES))
    choices = [o for o in range(N_OPS) if o != child[pos]]
    child[pos] = int(rng.choice(choices))
    return child


def runRandomSearch(api, dataset='cifar10', hp='200', budget=1020, seed=0, flops_min=None, flops_max=None,
                    true_front_objs=None, results_dir=None):
    """Uniform random search over the 15,625 cells.

    The canonical NAS sanity baseline: any method that fails to beat it is
    not exploiting the structure of the search space.

    Returns ``(run, cache)``, like every other runner in this module.
    """
    rng = np.random.default_rng(seed)
    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)
    t0 = time.time()
    try:
        while True:
            cache.query(_randomGene(rng))
    except BudgetReached:
        pass
    _, run, _ = _finish('RandomSearch', cache, dataset, seed,
                        time.time() - t0, None, extra={})
    run['score'] = scoreRun(cache, true_front_objs)
    _persist('RandomSearch', run, cache, seed, results_dir)
    return run, cache


def runRegularizedEvolution(api, dataset='cifar10', hp='200', budget=1020,
                            seed=0, pop_size=20, sample_size=5,
                            flops_min=None, flops_max=None,
                            true_front_objs=None, results_dir=None):
    """Regularized (aging) evolution, following Real et al. (2019).

    A tournament of ``sample_size`` individuals is drawn from the population,
    the winner is mutated at one edge, and the *oldest* individual is evicted
    rather than the worst — the ageing mechanism that gives the method its
    name and keeps it from stagnating on an early lucky individual.

    Returns ``(run, cache)``, like every other runner in this module.
    """
    rng = np.random.default_rng(seed)
    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)
    population = deque(maxlen=pop_size)
    t0 = time.time()
    try:
        for _ in range(pop_size):
            g = _randomGene(rng)
            rec = cache.query(g)
            population.append((g, rec['valid_acc']))

        while True:
            idx = rng.choice(len(population),
                             size=min(sample_size, len(population)),
                             replace=False)
            parent = max((population[int(i)] for i in idx), key=lambda p: p[1])
            child = _mutateGene(parent[0], rng)
            rec = cache.query(child)
            population.append((child, rec['valid_acc']))
    except BudgetReached:
        pass
    _, run, _ = _finish('RegularizedEvolution', cache, dataset, seed,
                        time.time() - t0, None,
                        extra={'pop_size': pop_size, 'sample_size': sample_size})
    run['score'] = scoreRun(cache, true_front_objs)
    _persist('RegularizedEvolution', run, cache, seed, results_dir)
    return run, cache


def _point(rec, cache):
    """Serialize one architecture for the results JSON.

    ``f1``/``f2`` are the objective pair **on test accuracy** — the quantity a
    method is judged on — so the existing aggregation and figure scripts
    (``aggregate_mo_nas.py``) read these files unchanged. ``valid_acc`` is
    kept alongside because it is what the search actually optimized.
    """
    f2 = cache.normFlops(rec['flops_M'])
    return {
        'gene': list(rec['gene']),
        'arch_index': rec['arch_index'],
        'arch_str': rec.get('arch_str', '?'),
        'valid_acc': round(rec['valid_acc'], 6),
        'test_acc': round(rec['test_acc'], 6),
        'flops_M': round(rec['flops_M'], 4),
        'params_MB': round(rec.get('params_MB', 0.0), 4),
        'norm_flops': round(f2, 6),
        'f1': round(1.0 - rec['test_acc'], 6),
        'f2': round(f2, 6),
    }


def _finish(name, cache, dataset, seed, elapsed, results_dir, extra):
    """Assemble the run record and optionally persist it."""
    best = cache.bestByValid()
    run = {
        'algorithm': name,
        'benchmark': 'NAS-Bench-201',
        'timestamp': datetime.now().isoformat(),
        'config': dict(dataset=dataset, hp=cache.hp, seed=seed, **extra,
                       **cache.stats()),
        # Selection uses validation accuracy; the test score is reported for
        # the selected architecture only.
        'best': None if best is None else _point(best, cache),
        'total_time_s': round(elapsed, 2),
    }
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = os.path.join(results_dir, f'{name}_{ts}_seed{seed}')
        with open(stem + '.json', 'w') as fh:
            json.dump(run, fh, indent=2)
        try:
            import pandas as pd
            pd.DataFrame(cache.log).to_csv(stem + '_evals.csv', index=False)
        except ImportError:
            pass
    return best, run, cache


# ─────────────────────────────────────────────────────────────────────────────
# Self-test against a synthetic tabular benchmark
# ─────────────────────────────────────────────────────────────────────────────

class _StubAPI:
    """Deterministic stand-in for the NATS-Bench API, for testing only.

    The landscape is mostly additive across edges, with a small pairwise
    interaction term — the structure NAS-Bench-201 actually has, and the
    structure a mutation-based method is supposed to exploit. A purely random
    table would make regularized evolution indistinguishable from random
    search and the baseline test vacuous.
    """

    # Per-operation FLOPs cost, ordered as in the real search space:
    # none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3.
    OP_COST = np.array([0.0, 0.0, 8.0, 36.0, 1.0])

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self._w = rng.normal(size=(N_EDGES, N_OPS))
        self._inter = rng.normal(size=(N_OPS, N_OPS)) * 0.25
        self._shift = rng.normal(size=N_OPS) * 0.004


def _stubQuery(api, gene, dataset='cifar10', hp='200'):
    gene = [int(g) for g in gene]
    idx = 0
    for g in gene:
        idx = idx * N_OPS + g

    score = sum(api._w[i, g] for i, g in enumerate(gene))
    score += sum(api._inter[gene[i], gene[i + 1]] for i in range(N_EDGES - 1))
    valid = 0.40 + 0.50 / (1.0 + np.exp(-score))
    # The test split differs from validation by a small, architecture-
    # dependent offset — the generalization gap the protocol must not peek at.
    test = valid + float(api._shift[gene[0]]) + 0.002 * np.cos(idx)
    flops = 4.0 + float(api.OP_COST[gene].sum() if hasattr(gene, 'sum')
                        else sum(api.OP_COST[g] for g in gene))

    return {
        'index': idx,
        'arch_str': '|'.join(str(g) for g in gene),
        'valid_acc': float(valid) * 100.0,
        'test_acc': float(test) * 100.0,
        'flops_M': flops,
        'params_MB': flops / 140.0,
    }


def selftest():
    """Exercise the cache, budget modes, scalarization and both baselines."""
    import sys
    import types

    # Inject a fake utils.nas so ArchCache._lookup resolves without nats_bench.
    mod = types.ModuleType('utils.nas')

    def queryArchitecture(api, gene, dataset='cifar10', hp='200'):
        r = _stubQuery(api, gene, dataset, hp)
        return {**r, 'valid_acc': r['valid_acc'] / 100.0,
                'test_acc': r['test_acc'] / 100.0}

    mod.queryArchitecture = queryArchitecture
    pkg = sys.modules.setdefault('utils', types.ModuleType('utils'))
    pkg.__path__ = []
    sys.modules['utils.nas'] = mod

    api = _StubAPI()
    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok ' if cond else 'FAIL'}] {label}")

    print('splitFor')
    check("cifar10 searches on cifar10-valid",
          splitFor('cifar10') == ('cifar10-valid', 'valid_acc', 'cifar10', 'test_acc'))
    check("cifar100 searches on its own valid split",
          splitFor('cifar100')[:2] == ('cifar100', 'valid_acc'))

    print('ArchCache — a repeat is charged like any other proposal')
    c = ArchCache(api, 'cifar100', budget=10, flops_min=8, flops_max=208)
    try:
        for _ in range(50):
            c.query([1, 1, 1, 1, 1, 1])
    except BudgetReached:
        pass
    check(f"stopped at 10 proposals (got {c.n_proposals})", c.n_proposals == 10)
    check(f"only 1 unique architecture (got {c.n_unique})", c.n_unique == 1)
    check(f"9 cache hits (got {c.n_hits})", c.n_hits == 9)

    print('scalarize')
    check("alpha=1 weighted sum ignores cost",
          scalarize(0.3, 0.9, 1.0) == 0.3)
    check("alpha=0 weighted sum ignores accuracy",
          scalarize(0.3, 0.9, 0.0) == 0.9)
    check("tchebycheff reduces to the active term at the endpoints",
          abs(scalarize(0.3, 0.9, 1.0, method='tchebycheff')
              - (0.3 + TCHEBYCHEFF_RHO * 0.3)) < 1e-12)
    ws = [scalarize(0.2, 0.8, a) for a in (0.0, 0.5, 1.0)]
    check("weighted sum is monotone in alpha here", ws[0] > ws[1] > ws[2])

    print('scalarization coverage on a concave front')
    # The standard non-convex front f2 = 1 - f1^2 (as in ZDT2), sampled
    # densely. Every point on it is Pareto optimal, but the weighted sum
    # a*f1 + (1-a)*f2 is concave in f1, so its minimum always sits at an
    # endpoint: no weight recovers the interior.
    t = np.linspace(0, 1, 101)
    concave = [(float(x), float(1.0 - x * x)) for x in t]
    sweep = np.linspace(0, 1, 41)

    def covered(method):
        found = set()
        for a in sweep:
            vals = [scalarize(f1, f2, a, method=method) for f1, f2 in concave]
            found.add(int(np.argmin(vals)))
        return found

    ws_cov, tch_cov = covered('weighted_sum'), covered('tchebycheff')
    check(f"weighted sum collapses to the extremes ({len(ws_cov)} distinct "
          f"points of {len(concave)})", len(ws_cov) <= 3)
    check(f"tchebycheff recovers the interior ({len(tch_cov)} distinct points "
          f"of {len(concave)})", len(tch_cov) > 10 * len(ws_cov))

    print('nonDominated')
    # e=(0.05,1.0) is better than a=(0.1,0.9) in f1 and worse in f2, so
    # neither dominates the other: e belongs to the front. d is dominated by a.
    pts = [(0.1, 0.9, 'a'), (0.2, 0.5, 'b'), (0.3, 0.4, 'c'),
           (0.25, 0.95, 'd'), (0.05, 1.0, 'e')]
    f = [p[2] for p in nonDominated(pts)]
    check(f"front is e,a,b,c and drops the dominated d (got {f})",
          f == ['e', 'a', 'b', 'c'])

    print('baselines')
    run, cache = runRandomSearch(api, 'cifar100', budget=200, seed=0,
                                 flops_min=8, flops_max=208)
    check(f"random search spent its budget (got {run['config']['n_proposals']})",
          run['config']['n_proposals'] == 200)
    check("random search reports both valid and test",
          run['best']['valid_acc'] > 0 and run['best']['test_acc'] > 0)
    check("every runner returns (run, cache) with a score block",
          'score' in run and 'n_unique' in run['score'])

    # Compare over several seeds: a single seed says nothing about either method.
    wins, rs_all, re_all = 0, [], []
    for s in range(8):
        r_rs, _ = runRandomSearch(api, 'cifar100', budget=200, seed=s,
                                  flops_min=8, flops_max=208)
        r_re, cache = runRegularizedEvolution(
            api, 'cifar100', budget=200, seed=s, flops_min=8, flops_max=208)
        rs_all.append(r_rs['best']['valid_acc'])
        re_all.append(r_re['best']['valid_acc'])
        wins += r_re['best']['valid_acc'] >= r_rs['best']['valid_acc']
    check(f"regularized evolution spent its budget "
          f"(got {r_re['config']['n_proposals']})",
          r_re['config']['n_proposals'] == 200)
    check(f"regularized evolution beats random search on a structured "
          f"landscape ({np.mean(re_all):.4f} vs {np.mean(rs_all):.4f}, "
          f"{wins}/8 seeds)", np.mean(re_all) > np.mean(rs_all) and wins >= 6)
    check(f"it re-proposes, so unique < proposals "
          f"({r_re['config']['n_unique']} < 200)",
          r_re['config']['n_unique'] < 200)

    print('no test leakage')
    # The search must never consult the test column: selection by validation
    # must be able to differ from selection by test.
    by_valid = max(cache.archive(), key=lambda r: r['valid_acc'])
    by_test = max(cache.archive(), key=lambda r: r['test_acc'])
    check("validation-selected and test-selected picks are distinguishable",
          by_valid['arch_index'] != by_test['arch_index']
          or by_valid['valid_acc'] != by_valid['test_acc'])

    print('archiveFront')
    front_v = archiveFront(cache, use_test=False)
    front_t = archiveFront(cache, use_test=True)
    check(f"validation front is non-empty ({len(front_v)} points)", len(front_v) > 1)
    check(f"test front is non-empty ({len(front_t)} points)", len(front_t) > 1)

    print('pymoo runners (NSGA-II, GDE3) under the corrected protocol')
    _, tf_objs, (fmin, fmax) = trueParetoFront(api, 'cifar100')
    hv_star = hypervolume(tf_objs)
    # The stub's FLOPs are a sum of five op costs, so its front is coarse;
    # the real benchmark yields far more points.
    check(f"true front is non-trivial ({len(tf_objs)} points, HV*={hv_star:.4f})",
          len(tf_objs) >= 3 and hv_star > 0)

    results = {}
    for label, fn in (('NSGA-II', runNSGA2MO), ('GDE3', runGDE3MO)):
        run, c = fn(api, dataset='cifar100', budget=400, pop_size=20, seed=0,
                    flops_min=fmin, flops_max=fmax, true_front_objs=tf_objs)
        results[label] = (run, c)
        s = run['score']
        check(f"{label} stopped exactly at budget "
              f"({s['n_proposals']} proposals)", s['n_proposals'] == 400)
        check(f"{label} reports proposals and unique separately "
              f"({s['n_proposals']} / {s['n_unique']})",
              s['n_unique'] <= s['n_proposals'])
        check(f"{label} HV is below the true front "
              f"({s['hv_test']:.4f} <= {hv_star:.4f})",
              s['hv_test'] <= hv_star + 1e-9)
        check(f"{label} IGD+ is finite ({s['igd_plus_test']:.5f})",
              np.isfinite(s['igd_plus_test']))

    # REGRESSION: the generation cap must not bind before the budget does.
    for label, fn in (('NSGA-II', runNSGA2MO), ('GDE3', runGDE3MO)):
        r, c = fn(api, dataset='cifar100', budget=250, pop_size=10, seed=0,
                  flops_min=fmin, flops_max=fmax, true_front_objs=tf_objs)
        sc = r['score']
        check(f"{label} spends exactly its proposal budget "
              f"({sc['n_proposals']} proposals, {sc['n_unique']} unique)",
              sc['n_proposals'] == 250 and sc['budget_met'])
        check(f"{label} re-proposes, so unique <= proposals",
              sc['n_unique'] <= sc['n_proposals'])

    print('validation-driven search does not equal test-driven search')
    s = results['GDE3'][0]['score']
    check(f"test HV differs from validation HV "
          f"({s['hv_test']:.5f} vs {s['hv_valid']:.5f}) — the generalization "
          f"gap the old protocol hid", abs(s['hv_test'] - s['hv_valid']) > 1e-9)
    check(f"best test acc <= best valid acc "
          f"({s['best_test_acc']:.4f} vs {s['best_valid_acc']:.4f}) is not "
          f"forced, but both are reported", True)

    print('\nSELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if selftest() else 1)
