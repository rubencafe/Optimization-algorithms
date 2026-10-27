from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime

import numpy as np

N_EDGES = 6
N_OPS = 5

ALPHA_SWEEP_DENSE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

TCHEBYCHEFF_RHO = 0.05


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
    """Return (search_dataset, search_key, report_dataset, report_key): the
    search uses validation accuracy, the report test accuracy.

    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    """
    if dataset not in _SEARCH_DATASET:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"expected one of {sorted(_SEARCH_DATASET)}")
    return (_SEARCH_DATASET[dataset], 'valid_acc',
            _REPORT_DATASET[dataset], 'test_acc')


class BudgetReached(Exception):
    """Raised by the objective once the budget is spent, so every optimizer
    stops at the same point.
    """


class ArchCache:
    """Memoized benchmark lookups with the budget accounting shared by every
    optimizer.
    """

    def __init__(self, api, dataset, hp='200', budget=1020,
                 flops_min=None, flops_max=None):
        """Create an empty cache for one run.

        api: API from getApi.
        dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
        hp: training epochs of the benchmark lookup, '12' or '200'.
        budget: number of proposals allowed.
        flops_min, flops_max: FLOPs range used to normalize the cost objective.
        """
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

    @property
    def n_unique(self):
        """Number of distinct architectures evaluated."""
        return len(self._store)

    @property
    def spent(self):
        """Number of proposals made so far."""
        return self.n_proposals

    def exhausted(self):
        """Return True once the budget is spent."""
        return self.n_proposals >= self.budget

    def query(self, gene):
        """Count one proposal and return the cell's record (validation and test
        accuracy, FLOPs); raise BudgetReached when the budget is spent.

        gene: 6 values, rounded and clipped to operation indices.
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
        """Query the benchmark for one cell on the search and report datasets.

        gene: tuple of 6 operation indices.
        """
        from utils.nas import queryArchitecture

        s = queryArchitecture(self.api, list(gene),
                              dataset=self.search_dataset, hp=self.hp)
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
        """Append one proposal to the evaluation log.

        rec: architecture record.
        cached: True if the cell had already been evaluated.
        """
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

    def normFlops(self, flops):
        """Normalize FLOPs to [0, 1] with the cache's FLOPs range.

        flops: FLOPs in millions.
        """
        if self.flops_min is None or self.flops_max is None:
            return 0.0
        span = self.flops_max - self.flops_min
        if span <= 0:
            return 0.0
        return float(np.clip((flops - self.flops_min) / span, 0.0, 1.0))

    def objectives(self, rec):
        """Return the minimized pair (1 - validation accuracy, normalized
        FLOPs).

        rec: architecture record.
        """
        return 1.0 - rec['valid_acc'], self.normFlops(rec['flops_M'])

    def bestByValid(self):
        """Return the evaluated architecture with the highest validation
        accuracy.
        """
        if not self._store:
            return None
        return max(self._store.values(), key=lambda r: r['valid_acc'])

    def archive(self):
        """Return every distinct architecture evaluated."""
        return list(self._store.values())

    def stats(self):
        """Return the budget counters (proposals, unique cells, cache hits) and
        the datasets used.
        """
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
        """Return a generation cap large enough that the budget, not the cap,
        ends the run.

        pop_size: population size.
        """
        return int(np.ceil(self.budget / max(pop_size, 1))) + 5


def scalarize(f1, f2, alpha, method='weighted_sum', rho=TCHEBYCHEFF_RHO,
              ideal=(0.0, 0.0)):
    """Combine the two minimized objectives into one value.

    f1: accuracy gap (1 - accuracy).
    f2: normalized FLOPs.
    alpha: weight of f1; 1 optimizes accuracy only, 0 cost only.
    method: 'weighted_sum' or 'tchebycheff'.
    rho: augmentation coefficient of the Tchebycheff aggregation.
    ideal: ideal point of the Tchebycheff aggregation.
    """
    w1, w2 = float(alpha), 1.0 - float(alpha)
    d1, d2 = f1 - ideal[0], f2 - ideal[1]

    if method == 'weighted_sum':
        return w1 * d1 + w2 * d2
    if method == 'tchebycheff':
        return max(w1 * d1, w2 * d2) + rho * (w1 * d1 + w2 * d2)
    raise ValueError(f"unknown scalarization {method!r}")


def makeScalarObjective(cache, alpha, method='weighted_sum'):
    """Build a minimization objective that queries the cache and returns the
    scalarized value.

    cache: ArchCache of the run.
    alpha: weight of the accuracy objective.
    method: 'weighted_sum' or 'tchebycheff'.
    """

    def objective(solution):
        """Return the scalarized objective of one solution.

        solution: 6 values, rounded to operation indices.
        """
        rec = cache.query(solution)
        f1, f2 = cache.objectives(rec)
        return scalarize(f1, f2, alpha, method=method)

    objective.cache = cache
    objective.alpha = alpha
    objective.method = method
    return objective


def nonDominated(points):
    """Return the non-dominated tuples of a list, both objectives minimized.

    points: list of (f1, f2, payload).
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front, best_f2 = [], float('inf')
    for p in pts:
        if p[1] < best_f2:
            front.append(p)
            best_f2 = p[1]
    return front


def archiveFront(cache, use_test=False):
    """Return the non-dominated front of every architecture the run evaluated.

    cache: ArchCache of the run.
    use_test: score the front with test accuracy instead of validation
        accuracy.
    """
    key = 'test_acc' if use_test else 'valid_acc'
    pts = [(1.0 - r[key], cache.normFlops(r['flops_M']), r)
           for r in cache.archive()]
    return [p[2] for p in nonDominated(pts)]


REF_POINT = (1.1, 1.1)


def _objArray(records, use_test, cache):
    """Return the (n, 2) objective array of a list of records.

    records: architecture records.
    use_test: use test accuracy instead of validation accuracy.
    cache: ArchCache that normalizes FLOPs.
    """
    key = 'test_acc' if use_test else 'valid_acc'
    return np.array([[1.0 - r[key], cache.normFlops(r['flops_M'])]
                     for r in records], dtype=float)


def hypervolume(front_objs, ref_point=REF_POINT):
    """Return the hypervolume of a front.

    front_objs: (n, 2) minimized objectives.
    ref_point: reference point.
    """
    from pymoo.indicators.hv import HV
    F = np.atleast_2d(np.asarray(front_objs, dtype=float))
    if F.size == 0:
        return 0.0
    return float(HV(ref_point=np.array(ref_point))(F))


def igdPlus(front_objs, true_front_objs):
    """Return the IGD+ distance from a front to the exact front.

    front_objs: (n, 2) objectives of the front.
    true_front_objs: (m, 2) objectives of the exact Pareto front.
    """
    from pymoo.indicators.igd_plus import IGDPlus
    F = np.atleast_2d(np.asarray(front_objs, dtype=float))
    Z = np.atleast_2d(np.asarray(true_front_objs, dtype=float))
    if F.size == 0:
        return float('inf')
    return float(IGDPlus(Z)(F))


def scoreRun(cache, true_front_objs=None, ref_point=REF_POINT):
    """Score a finished run on test accuracy: front size, hypervolume, best
    accuracies, budget counters and, with a reference front, IGD+ and the
    hypervolume gap.

    cache: ArchCache of the run.
    true_front_objs: objectives of the exact Pareto front, or None.
    ref_point: hypervolume reference point.
    """
    valid_front = archiveFront(cache, use_test=False)
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
    """Enumerate all 15,625 cells and return (front, objectives, (flops_min,
    flops_max)) of the exact Pareto front.

    api: API from getApi.
    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the benchmark lookup, '12' or '200'.
    flops_min, flops_max: FLOPs range; taken from the benchmark when None.
    use_test: define the front on test accuracy instead of validation accuracy.
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
    """Write a reference front as the flat JSON list read by
    aggregate_mo_nas.py, and return the path.

    path: output file.
    front: records of the front.
    flops_min, flops_max: FLOPs range used for normalization.
    """
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


def makeMOProblem(cache):
    """Build a pymoo problem whose two objectives come from the cache.

    cache: ArchCache of the run.
    """
    from pymoo.core.problem import Problem

    class _Problem(Problem):
        """Two-objective problem over the 6 cell edges."""
        def __init__(self):
            """Declare the 6 variables, their bounds and the 2 objectives."""
            super().__init__(n_var=N_EDGES, n_obj=2,
                             xl=np.zeros(N_EDGES),
                             xu=np.full(N_EDGES, N_OPS - 1, dtype=float))

        def _evaluate(self, X, out, *args, **kwargs):
            """Evaluate a population and write its objectives to out['F'].

            X: population, one row per solution.
            out: pymoo output dict.
            """
            F = np.zeros((len(X), 2))
            for i, x in enumerate(X):
                F[i] = cache.objectives(cache.query(x))
            out['F'] = F

    return _Problem()


def runPymooMO(api, algorithm, name, dataset='cifar10', hp='200', budget=1020,
               pop_size=20, seed=0,
               flops_min=None, flops_max=None, true_front_objs=None,
               results_dir=None):
    """Run a pymoo multi-objective algorithm until the budget is spent and
    return (run, cache).

    api: API from getApi.
    algorithm: configured pymoo algorithm.
    name: algorithm name saved with the results.
    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the benchmark lookup, '12' or '200'.
    budget: number of proposals.
    pop_size: population size.
    seed: random seed.
    flops_min, flops_max: FLOPs range used to normalize the cost objective.
    true_front_objs: objectives of the exact Pareto front used to score the
        run, or None.
    results_dir: folder for the result JSON and evaluation log CSV, or None to
        skip saving.
    """
    from pymoo.optimize import minimize

    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)
    problem = makeMOProblem(cache)

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
    """Run NSGA-II through runPymooMO.

    api: API from getApi.
    kw: arguments of runPymooMO, plus pop_size and eliminate_duplicates.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    pop_size = kw.pop('pop_size', 20)
    elim = kw.pop('eliminate_duplicates', False)
    algo = NSGA2(pop_size=pop_size,
                 sampling=FloatRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(prob=1.0 / N_EDGES, eta=20),
                 eliminate_duplicates=elim)
    return runPymooMO(api, algo, 'NSGA-II', pop_size=pop_size, **kw)


def runGDE3MO(api, **kw):
    """Run GDE3 through runPymooMO.

    api: API from getApi.
    kw: arguments of runPymooMO, plus pop_size.
    """
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
    """Run a mealpy optimizer on the scalarized objective until the budget is
    spent and return (run, cache).

    api: API from getApi.
    algorithm: 'GA', 'SADE', 'AIW_PSO' or 'GWO'.
    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the benchmark lookup, '12' or '200'.
    alpha: weight of the accuracy objective; 1 is the accuracy-only
        formulation.
    method: 'weighted_sum' or 'tchebycheff'.
    budget: number of proposals.
    pop_size: population size.
    seed: random seed.
    flops_min, flops_max: FLOPs range used to normalize the cost objective.
    results_dir: folder for the result JSON and evaluation log CSV, or None to
        skip saving.
    """
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
    """Run Bayesian Optimization on the scalarized objective until the budget
    is spent and return (run, cache).

    api: API from getApi.
    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the benchmark lookup, '12' or '200'.
    alpha: weight of the accuracy objective; 1 is the accuracy-only
        formulation.
    method: 'weighted_sum' or 'tchebycheff'.
    budget: number of proposals.
    init_points: random points evaluated before the GP starts proposing.
    xi: exploration parameter of Expected Improvement.
    seed: random seed.
    flops_min, flops_max: FLOPs range used to normalize the cost objective.
    results_dir: folder for the result JSON and evaluation log CSV, or None to
        skip saving.
    """
    from bayes_opt import BayesianOptimization, acquisition

    cache = ArchCache(api, dataset, hp=hp, budget=budget,
                      flops_min=flops_min, flops_max=flops_max)
    objective = makeScalarObjective(cache, alpha, method=method)

    def bo_objective(e0, e1, e2, e3, e4, e5):
        """Return the negated scalarized objective of one BO point (BO
        maximizes).

        e0, e1, e2, e3, e4, e5: continuous values, rounded to operation
            indices.
        """
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
    """Run qEHVI (BoTorch) on the two objectives until the budget is spent and
    return (run, cache).

    api: API from getApi.
    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the benchmark lookup, '12' or '200'.
    budget: number of proposals.
    init_points: Sobol points evaluated before the GP starts proposing.
    batch_size: candidates proposed per iteration (q).
    mc_samples: Monte Carlo samples of the acquisition function.
    num_restarts, raw_samples: settings of the acquisition optimizer.
    seed: random seed.
    flops_min, flops_max: FLOPs range used to normalize the cost objective.
    true_front_objs: objectives of the exact Pareto front used to score the
        run, or None.
    results_dir: folder for the result JSON and evaluation log CSV, or None to
        skip saving.
    verbose: print progress.
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
        """Evaluate points of the unit cube and return their negated validation
        objectives.

        X_unit: (q, 6) tensor in [0, 1].
        """
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
    """Save the run JSON and its evaluation log CSV, and return the file stem
    (None when not saving).

    name: algorithm name.
    run: run record.
    cache: ArchCache of the run.
    seed: random seed.
    results_dir: output folder, or None.
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


def _randomGene(rng):
    """Return a uniformly random gene.

    rng: numpy random generator.
    """
    return [int(x) for x in rng.integers(0, N_OPS, size=N_EDGES)]


def _mutateGene(gene, rng):
    """Return a copy of the gene with one edge changed to a different
    operation.

    gene: 6 operation indices.
    rng: numpy random generator.
    """
    child = list(gene)
    pos = int(rng.integers(0, N_EDGES))
    choices = [o for o in range(N_OPS) if o != child[pos]]
    child[pos] = int(rng.choice(choices))
    return child


def runRandomSearch(api, dataset='cifar10', hp='200', budget=1020, seed=0, flops_min=None, flops_max=None,
                    true_front_objs=None, results_dir=None):
    """Sample cells uniformly at random until the budget is spent and return
    (run, cache).

    api: API from getApi.
    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the benchmark lookup, '12' or '200'.
    budget: number of proposals.
    seed: random seed.
    flops_min, flops_max: FLOPs range used to normalize the cost objective.
    true_front_objs: objectives of the exact Pareto front used to score the
        run, or None.
    results_dir: folder for the result JSON and evaluation log CSV, or None to
        skip saving.
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
    """Run regularized (aging) evolution (Real et al., 2019) until the budget
    is spent and return (run, cache).

    api: API from getApi.
    dataset: 'cifar10', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the benchmark lookup, '12' or '200'.
    budget: number of proposals.
    seed: random seed.
    pop_size: population size.
    sample_size: tournament size.
    flops_min, flops_max: FLOPs range used to normalize the cost objective.
    true_front_objs: objectives of the exact Pareto front used to score the
        run, or None.
    results_dir: folder for the result JSON and evaluation log CSV, or None to
        skip saving.
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
    """Serialize one architecture for the results JSON, with f1 and f2 computed
    on test accuracy.

    rec: architecture record.
    cache: ArchCache that normalizes FLOPs.
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
    """Assemble the run record, optionally save it, and return (best, run,
    cache).

    name: algorithm name.
    cache: ArchCache of the run.
    dataset: dataset name.
    seed: random seed.
    elapsed: run time in seconds.
    results_dir: output folder, or None to skip saving.
    extra: extra fields stored in the run configuration.
    """
    best = cache.bestByValid()
    run = {
        'algorithm': name,
        'benchmark': 'NAS-Bench-201',
        'timestamp': datetime.now().isoformat(),
        'config': dict(dataset=dataset, hp=cache.hp, seed=seed, **extra,
                       **cache.stats()),
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
