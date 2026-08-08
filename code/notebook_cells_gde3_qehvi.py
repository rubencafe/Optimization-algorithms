"""
notebook_cells_gde3_qehvi.py
============================
Paste-ready cells to add GDE3 and qEHVI to NAS-mo-pipeline.ipynb.

GDE3 and qEHVI are *native* multi-objective methods (like NSGA-II): one
non-dominated front per seed, no alpha sweep. So their cells mirror the
NSGA-II cell, not the scalarised ones.

Copy each block below into its own notebook cell, in order. They assume the
existing setup cells (imports, config, api, flops_ranges, true_fronts) have
already run.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL — extend imports (add to the existing `from utils.nas_mo import (...)`)
# ─────────────────────────────────────────────────────────────────────────────
from utils.nas_mo import runGDE3NASSearch, runQEHVINASSearch   # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# CELL — extra config for the two native methods
# ─────────────────────────────────────────────────────────────────────────────
# GDE3 — same population budget as NSGA-II so they are directly comparable.
GDE3_CR      = 0.9            # DE crossover rate
GDE3_F       = 0.5           # DE scale factor
GDE3_VARIANT = 'DE/rand/1/bin'

# qEHVI — total queries = QEHVI_INIT + QEHVI_N_ITER * QEHVI_BATCH.
# Match the EA budget N_QUERIES = (N_GENERATIONS+1)*POP_SIZE = 1020:
#   20 + 200 * 5 = 1020.  qEHVI refits a GP every iteration, so we spend the
#   budget through a batch (q) of 5 rather than 1000 single-point iterations.
QEHVI_INIT   = POP_SIZE      # = 20
QEHVI_BATCH  = 5
QEHVI_N_ITER = (N_QUERIES - QEHVI_INIT) // QEHVI_BATCH   # = 200
QEHVI_MC     = 128           # MC samples for the acquisition


# ─────────────────────────────────────────────────────────────────────────────
# CELL — GDE3  (native, multi-objective Differential Evolution)
# ─────────────────────────────────────────────────────────────────────────────
gde3_summary = []

for dataset in DATASETS_USED:
    print(f'\n{"#"*70}')
    print(f'#  GDE3  —  {dataset}')
    print(f'{"#"*70}')

    fmin, fmax = flops_ranges[dataset]
    out_dir = MO_ROOT / 'gde3' / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f'\n  seed={seed}')
        front, run = runGDE3NASSearch(
            api           = api,
            dataset       = dataset,
            hp            = HP,
            n_generations = N_GENERATIONS,
            pop_size      = POP_SIZE,
            seed          = seed,
            CR            = GDE3_CR,
            F             = GDE3_F,
            variant       = GDE3_VARIANT,
            flops_min     = fmin,
            flops_max     = fmax,
            results_dir   = str(out_dir),
            verbose       = False,
        )
        tf_hv = computeHypervolume([[p['f1'], p['f2']] for p in true_fronts[dataset]])
        gde3_summary.append({
            'dataset':    dataset,
            'algorithm':  'GDE3',
            'seed':       seed,
            'hv':         run['hypervolume'],
            'hv_gap':     tf_hv - run['hypervolume'],
            'front_size': len(front),
            'best_acc':   max(p['test_acc'] for p in front),
            'min_flops':  min(p['flops_M']  for p in front),
            'n_evals':    run['n_evaluations'],
        })

df_gde3 = pd.DataFrame(gde3_summary)
print('\nGDE3 summary:')
print(df_gde3.groupby('dataset')[['hv', 'front_size', 'best_acc']]
      .agg(['mean', 'std']).round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CELL — qEHVI  (native, multi-objective Bayesian Optimisation; needs botorch)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: qEHVI refits a Gaussian process every iteration, so this is the
# slowest method here — expect minutes per seed. Reduce QEHVI_N_ITER for a
# quick smoke run.
qehvi_summary = []

for dataset in DATASETS_USED:
    print(f'\n{"#"*70}')
    print(f'#  qEHVI  —  {dataset}')
    print(f'{"#"*70}')

    fmin, fmax = flops_ranges[dataset]
    out_dir = MO_ROOT / 'qehvi' / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f'\n  seed={seed}')
        front, run = runQEHVINASSearch(
            api          = api,
            dataset      = dataset,
            hp           = HP,
            n_iter       = QEHVI_N_ITER,
            init_points  = QEHVI_INIT,
            batch_size   = QEHVI_BATCH,
            mc_samples   = QEHVI_MC,
            seed         = seed,
            flops_min    = fmin,
            flops_max    = fmax,
            results_dir  = str(out_dir),
            verbose      = False,
        )
        tf_hv = computeHypervolume([[p['f1'], p['f2']] for p in true_fronts[dataset]])
        qehvi_summary.append({
            'dataset':    dataset,
            'algorithm':  'qEHVI',
            'seed':       seed,
            'hv':         run['hypervolume'],
            'hv_gap':     tf_hv - run['hypervolume'],
            'front_size': len(front),
            'best_acc':   max(p['test_acc'] for p in front),
            'min_flops':  min(p['flops_M']  for p in front),
            'n_evals':    run['n_evaluations'],
        })

df_qehvi = pd.DataFrame(qehvi_summary)
print('\nqEHVI summary:')
print(df_qehvi.groupby('dataset')[['hv', 'front_size', 'best_acc']]
      .agg(['mean', 'std']).round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CELL — REPLACE the combined-summary concat in section 11 with this one
# ─────────────────────────────────────────────────────────────────────────────
all_runs = pd.concat(
    [df_nsga2, df_gde3, df_qehvi, df_ga, df_sade, df_gwo, df_pso, df_bo],
    ignore_index=True,
)
# ...rest of section 11 (the groupby/agg/save) stays exactly the same.


# ─────────────────────────────────────────────────────────────────────────────
# CELL — for the quick-plot cell (section 12), add these entries to the maps
# ─────────────────────────────────────────────────────────────────────────────
# ALGO_COLORS['GDE3']  = '#8c564b';  ALGO_COLORS['qEHVI'] = '#17becf'
# ALGO_MARKERS['GDE3'] = 'X';        ALGO_MARKERS['qEHVI'] = '*'
# ALGO_DIRS['GDE3']    = 'gde3';     ALGO_DIRS['qEHVI']   = 'qehvi'
# HV_MEANS['GDE3']     = df_gde3;    HV_MEANS['qEHVI']    = df_qehvi
#
# The publication figures in section 13 (aggregate_mo_nas.py) already include
# GDE3 and qEHVI — they were registered in ALGORITHMS / ALGO_FOLDERS there.
