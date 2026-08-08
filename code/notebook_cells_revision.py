"""Paste-ready notebook cells for the revised (leakage-free) protocol.

SUPERSEDED by NAS-revision-pipeline.ipynb, which runs all of this in order
with checkpointing and figure generation. Kept only for splicing individual
steps into the original notebooks; prefer the notebook.

Each block below marked `# %% CELL` replaces or follows a cell in the two
multi-objective notebooks. Nothing here mutates the existing runners, so the
original results stay reproducible from the old cells.

Cells 1-3 are shared setup. Cells 4-7 go in `NAS-mo-pipeline.ipynb`,
cells 8-9 in `NAS-mo-pipeline-gde3-qehvi.ipynb`.

Reminder: `NAS-pipeline.ipynb` (Case Study 1, accuracy-only) also searched on
`test_acc` and needs the same treatment — cell 10 covers it.
"""

# %% CELL 1 — imports and configuration (replaces the existing config cell)
CELL_1 = r'''
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT if (ROOT / 'utils').exists() else ROOT.parent))

from utils.nas import getApi
from utils.nas_revision import (
    ALPHA_SWEEP_DENSE, REF_POINT,
    trueParetoFront, hypervolume, igdPlus, scoreRun,
    runNSGA2MO, runGDE3MO, runQEHVIMO,
    runScalarizedMealpy, runBOScalarized,
    runRandomSearch, runRegularizedEvolution,
)

NAS_PATH = ROOT / 'utils' / 'nas_records' / 'NATS-tss-v1_0' / 'NATS-tss-v1_0'
api = getApi(NAS_PATH)

DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
HP       = '200'
BUDGET   = 1020
POP_SIZE = 20
SEEDS    = [0, 1, 2]

# The budget is a count of objective calls; a repeat is charged like any other
# call. The optimizer decides what to explore and we never intervene when it
# re-proposes, so its duplicate rate is a reported result, not a correction.

RESULTS = ROOT / 'colab' / 'results' / 'MO_NAS_rev'
RESULTS.mkdir(parents=True, exist_ok=True)

print('search protocol : validation accuracy (test read only at the end)')
print('budget          :', BUDGET, 'proposals per run')
print('alpha sweep     :', ALPHA_SWEEP_DENSE)
'''

# %% CELL 2 — reference fronts (test-side; the search never sees these)
CELL_2 = r'''
# The reference front stays defined on TEST accuracy: it is what a method is
# judged against, not a signal the search may consult. Enumerating all 15,625
# cells takes a couple of minutes per dataset, so cache it to disk.
true_fronts, flops_ranges, hv_star = {}, {}, {}

for ds in DATASETS:
    cache_file = RESULTS / f'true_front_{ds}.json'
    if cache_file.exists():
        blob = json.loads(cache_file.read_text())
    else:
        front, objs, (fmin, fmax) = trueParetoFront(api, ds, hp=HP)
        blob = {'objs': objs, 'flops_min': fmin, 'flops_max': fmax,
                'front': [{k: (list(v) if k == 'gene' else v)
                           for k, v in r.items()} for r in front]}
        cache_file.write_text(json.dumps(blob))
    true_fronts[ds]  = blob['objs']
    flops_ranges[ds] = (blob['flops_min'], blob['flops_max'])
    hv_star[ds]      = hypervolume(blob['objs'], REF_POINT)
    print(f'{ds:16} front={len(blob["objs"]):4}  HV*={hv_star[ds]:.4f}')
'''

# %% CELL 3 — helper to collect one run's row
CELL_3 = r'''
def row(algo, ds, seed, run, extra=None):
    s = run['score'] if 'score' in run else {}
    r = {'algorithm': algo, 'dataset': ds, 'seed': seed,
         'n_proposals': s.get('n_proposals'), 'n_unique': s.get('n_unique'),
         'n_cache_hits': s.get('n_cache_hits'),
         'hv_test': s.get('hv_test'), 'hv_gap_test': s.get('hv_gap_test'),
         'igd_plus_test': s.get('igd_plus_test'),
         'best_valid_acc': s.get('best_valid_acc'),
         'best_test_acc': s.get('best_test_acc'),
         'time_s': run.get('total_time_s')}
    if run.get('best'):
        r['sel_valid_acc'] = run['best']['valid_acc']
        r['sel_test_acc']  = run['best']['test_acc']
        r['sel_flops_M']   = run['best']['flops_M']
    return {**r, **(extra or {})}

rows = []
'''

# %% CELL 4 — NAS-mo-pipeline.ipynb: native NSGA-II
CELL_4 = r'''
for ds in DATASETS:
    fmin, fmax = flops_ranges[ds]
    for seed in SEEDS:
        run, _ = runNSGA2MO(api, dataset=ds, hp=HP, budget=BUDGET,
                            pop_size=POP_SIZE, seed=seed,
                            flops_min=fmin, flops_max=fmax,
                            true_front_objs=true_fronts[ds],
                            results_dir=str(RESULTS / 'nsga2' / ds))
        rows.append(row('NSGA-II', ds, seed, run))
        print(f"NSGA-II {ds:16} seed={seed}  "
              f"unique={run['score']['n_unique']:4}  "
              f"HV_test={run['score']['hv_test']:.4f}")
'''

# %% CELL 5 — NAS-mo-pipeline.ipynb: scalarized EAs, both aggregations
CELL_5 = r'''
# 4 optimizers x 13 weights x 2 aggregations x 3 seeds x 3 datasets.
# All evolutionary, so the whole block is minutes, not hours.
for method in ('weighted_sum', 'tchebycheff'):
    for algo in ('GA', 'SADE', 'GWO', 'AIW_PSO'):
        for ds in DATASETS:
            fmin, fmax = flops_ranges[ds]
            for seed in SEEDS:
                pts = []
                for i, alpha in enumerate(ALPHA_SWEEP_DENSE):
                    run, cache = runScalarizedMealpy(
                        api, algo, dataset=ds, hp=HP, alpha=alpha,
                        method=method, budget=BUDGET, pop_size=POP_SIZE,
                        seed=seed * 100 + i,
                        flops_min=fmin, flops_max=fmax,
                        results_dir=str(RESULTS / method / algo.lower() / ds))
                    pts.append(run['best'])
                # Assemble the scalarized front from the per-alpha winners,
                # scored on test.
                objs = [[1.0 - p['test_acc'],
                         (p['flops_M'] - fmin) / max(fmax - fmin, 1e-12)]
                        for p in pts]
                rows.append({'algorithm': algo, 'dataset': ds, 'seed': seed,
                             'scalarization': method, 'native': False,
                             'hv_test': hypervolume(objs, REF_POINT),
                             'hv_gap_test': hv_star[ds] - hypervolume(objs, REF_POINT),
                             'igd_plus_test': igdPlus(objs, true_fronts[ds]),
                             'best_test_acc': max(p['test_acc'] for p in pts)})
            print(f'{method:14} {algo:8} {ds:16} done')
'''

# %% CELL 6 — NAS-mo-pipeline.ipynb: scalarized BO (the expensive one)
CELL_6 = r'''
# BO is ~2700 s per run. 13 weights x 3 seeds x 3 datasets = 117 runs ~= 88 h
# for ONE aggregation. Keep BO on the weighted sum only: the Tchebycheff-vs-
# weighted-sum comparison is about the aggregation, not the optimizer, and the
# four EAs above already establish it.
BO_METHOD = 'weighted_sum'
BO_ALPHAS = ALPHA_SWEEP_DENSE          # or a coarser subset to cut cost

for ds in DATASETS:
    fmin, fmax = flops_ranges[ds]
    for seed in SEEDS:
        pts = []
        for i, alpha in enumerate(BO_ALPHAS):
            run, _ = runBOScalarized(
                api, dataset=ds, hp=HP, alpha=alpha, method=BO_METHOD,
                budget=BUDGET, init_points=POP_SIZE, xi=0.01,
                seed=seed * 100 + i,
                flops_min=fmin, flops_max=fmax,
                results_dir=str(RESULTS / BO_METHOD / 'bo' / ds))
            pts.append(run['best'])
        objs = [[1.0 - p['test_acc'],
                 (p['flops_M'] - fmin) / max(fmax - fmin, 1e-12)] for p in pts]
        rows.append({'algorithm': 'BO', 'dataset': ds, 'seed': seed,
                     'scalarization': BO_METHOD, 'native': False,
                     'hv_test': hypervolume(objs, REF_POINT),
                     'hv_gap_test': hv_star[ds] - hypervolume(objs, REF_POINT),
                     'igd_plus_test': igdPlus(objs, true_fronts[ds]),
                     'best_test_acc': max(p['test_acc'] for p in pts)})
        print(f'BO {ds:16} seed={seed} done')
'''

# %% CELL 7 — new baselines (both notebooks / Case Study 1)
CELL_7 = r'''
for ds in DATASETS:
    fmin, fmax = flops_ranges[ds]
    for seed in SEEDS:
        for label, fn in (('RandomSearch', runRandomSearch),
                          ('RegularizedEvolution', runRegularizedEvolution)):
            run, _ = fn(api, dataset=ds, hp=HP, budget=BUDGET, seed=seed,
                        flops_min=fmin, flops_max=fmax,
                        true_front_objs=true_fronts[ds],
                        results_dir=str(RESULTS / label.lower() / ds))
            rows.append(row(label, ds, seed, run))
            print(f"{label:22} {ds:16} seed={seed}  "
                  f"test_acc={run['best']['test_acc']:.4f}")
'''

# %% CELL 8 — NAS-mo-pipeline-gde3-qehvi.ipynb: GDE3
CELL_8 = r'''
for ds in DATASETS:
    fmin, fmax = flops_ranges[ds]
    for seed in SEEDS:
        run, _ = runGDE3MO(api, dataset=ds, hp=HP, budget=BUDGET,
                           pop_size=POP_SIZE, seed=seed,
                           flops_min=fmin, flops_max=fmax,
                           true_front_objs=true_fronts[ds],
                           results_dir=str(RESULTS / 'gde3' / ds))
        rows.append(row('GDE3', ds, seed, run))
        print(f"GDE3 {ds:16} seed={seed}  "
              f"unique={run['score']['n_unique']:4}  "
              f"HV_test={run['score']['hv_test']:.4f}")
'''

# %% CELL 9 — NAS-mo-pipeline-gde3-qehvi.ipynb: qEHVI
CELL_9 = r'''
# SMOKE-TEST FIRST. This runner was written against the original loop but
# could not be executed during development (BoTorch was not installed), so
# verify it on a small budget before committing hours to it:
#
#   run, c = runQEHVIMO(api, dataset='cifar10', budget=60, init_points=10,
#                       batch_size=5, seed=0, flops_min=flops_ranges['cifar10'][0],
#                       flops_max=flops_ranges['cifar10'][1],
#                       true_front_objs=true_fronts['cifar10'], verbose=True)
#   print(run['score'])
#
for ds in DATASETS:
    fmin, fmax = flops_ranges[ds]
    for seed in SEEDS:
        run, _ = runQEHVIMO(api, dataset=ds, hp=HP, budget=BUDGET,
                            init_points=POP_SIZE, batch_size=5,
                            mc_samples=128, num_restarts=10, raw_samples=128,
                            seed=seed,
                            flops_min=fmin, flops_max=fmax,
                            true_front_objs=true_fronts[ds],
                            results_dir=str(RESULTS / 'qehvi' / ds))
        rows.append(row('qEHVI', ds, seed, run))
        print(f"qEHVI {ds:16} seed={seed}  "
              f"HV_test={run['score']['hv_test']:.4f}")
'''

# %% CELL 10 — NAS-pipeline.ipynb (Case Study 1, accuracy-only)
CELL_10 = r'''
# The accuracy-only study has the same leak: FITNESS_KEY was 'test_acc'.
# Single-objective search is the alpha=1 case of the scalarization, so the
# same runner covers it — alpha=1 puts zero weight on FLOPs.
cs1_rows = []
for ds in DATASETS:
    fmin, fmax = flops_ranges[ds]
    for algo in ('GA', 'SADE', 'GWO', 'AIW_PSO'):
        for seed in [0, 1, 2, 3, 4]:
            run, _ = runScalarizedMealpy(
                api, algo, dataset=ds, hp=HP, alpha=1.0,
                method='weighted_sum', budget=BUDGET, pop_size=POP_SIZE,
                seed=seed,
                flops_min=fmin, flops_max=fmax,
                results_dir=str(RESULTS / 'cs1' / algo.lower() / ds))
            cs1_rows.append(row(algo, ds, seed, run))
    for seed in [0, 1, 2, 3, 4]:
        run, _ = runBOScalarized(
            api, dataset=ds, hp=HP, alpha=1.0, method='weighted_sum',
            budget=BUDGET, init_points=POP_SIZE, xi=0.01, seed=seed,
            flops_min=fmin, flops_max=fmax,
            results_dir=str(RESULTS / 'cs1' / 'bo' / ds))
        cs1_rows.append(row('BO', ds, seed, run))

df_cs1 = pd.DataFrame(cs1_rows)
# The headline table: selection on validation, reporting on test.
print(df_cs1.groupby(['dataset', 'algorithm'])[
    ['sel_valid_acc', 'sel_test_acc', 'n_proposals', 'n_unique']
].mean().round(4).to_string())
'''

# %% CELL 11 — aggregate and save
CELL_11 = r'''
df = pd.DataFrame(rows)
df.to_csv(RESULTS / 'mo_nas_rev_per_run.csv', index=False)

print('\n=== proposals vs unique (the reviewer\'s point, now measured) ===')
print(df.groupby('algorithm')[['n_proposals', 'n_unique', 'n_cache_hits']]
        .mean().round(1).to_string())

print('\n=== quality on TEST, by algorithm and dataset ===')
print(df.groupby(['dataset', 'algorithm'])[['hv_test', 'hv_gap_test',
                                            'igd_plus_test', 'best_test_acc']]
        .agg(['mean', 'std']).round(5).to_string())
'''

if __name__ == '__main__':
    cells = {k: v for k, v in sorted(globals().items())
             if k.startswith('CELL_')}
    print(f'{len(cells)} cells defined')
    for name, body in cells.items():
        print(f'  {name:10} {len(body.splitlines())} lines')
