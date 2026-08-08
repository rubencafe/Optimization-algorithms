"""Publication figures for the revised NAS-Bench-201 study.

Regenerates every figure the paper uses, from the results the revised pipeline
writes. One function per figure; :func:`makeAll` runs the lot.

    from utils.nas_figures import makeAll
    makeAll(RESULTS, DATASETS, out_dir=RESULTS / 'figures')

Figure names match the ``\\includegraphics`` keys in ``article.tex``:

    cs1_gap_grid                 distance to the exhaustive upper bound
    cs1_scalar_fronts_<method>   scalarized non-dominated sets vs exact front
    cs1_alpha_sweep_<method>     accuracy-cost trajectory as alpha is swept
    cs2_fronts_nat               native Pareto fronts vs exact front
    cs2_hv_convergence_native    hypervolume vs evaluations
    proposals_vs_unique          budget spent on distinct architectures  (NEW)
    unique_per_dataset           the same, split by dataset              (NEW)

Everything is driven by the per-proposal eval logs, so the search is scored on
validation accuracy while every plotted quantity is reported on test accuracy.

Self-test (synthetic results tree, no benchmark needed)::

    python -m utils.nas_figures
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REF_POINT = (1.1, 1.1)

ALGO_COLORS = {
    'GA': '#1f77b4', 'SADE': '#d62728', 'AIW_PSO': '#2ca02c',
    'AIW-PSO': '#2ca02c', 'GWO': '#9467bd', 'BO': '#ff7f0e',
    'NSGA-II': '#e377c2', 'NSGA2': '#e377c2', 'GDE3': '#8c564b',
    'qEHVI': '#17becf', 'RandomSearch': '#7f7f7f',
    'RegularizedEvolution': '#bcbd22',
}
ALGO_MARKERS = {
    'GA': 'o', 'SADE': 's', 'AIW_PSO': '^', 'AIW-PSO': '^', 'GWO': 'D',
    'BO': 'v', 'NSGA-II': 'P', 'GDE3': 'X', 'qEHVI': '*',
    'RandomSearch': '.', 'RegularizedEvolution': 'h',
}
DS_LABEL = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100',
            'ImageNet16-120': 'ImageNet16-120'}


def _plt():
    import matplotlib
    if not os.environ.get('DISPLAY') and os.name != 'nt':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ('pdf', 'png'):
        p = out_dir / f'{name}.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        paths.append(p)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def loadManifest(results_dir):
    p = Path(results_dir) / 'manifest.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def loadRefs(results_dir, datasets):
    """Reference fronts, FLOPs ranges and accuracy upper bounds per dataset."""
    refs = {}
    for ds in datasets:
        p = Path(results_dir) / f'true_front_{ds}.json'
        if not p.exists():
            continue
        blob = json.loads(p.read_text())
        refs[ds] = {
            'objs': np.asarray(blob['objs'], dtype=float),
            'flops_min': blob['flops_min'], 'flops_max': blob['flops_max'],
            'best_test_acc': blob['best_test_acc'],
            'front': blob.get('front', []),
        }
    return refs


def loadEvals(stem):
    p = Path(str(stem) + '_evals.csv')
    return pd.read_csv(p) if p.exists() else None


def _normFlops(flops, ref):
    span = max(ref['flops_max'] - ref['flops_min'], 1e-12)
    return np.clip((np.asarray(flops) - ref['flops_min']) / span, 0.0, 1.0)


def _nonDominated(F):
    """Indices of the non-dominated rows of an (n, 2) minimization array."""
    F = np.asarray(F, dtype=float)
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best = [], np.inf
    for i in order:
        if F[i, 1] < best:
            keep.append(i)
            best = F[i, 1]
    return np.array(keep, dtype=int)


def _hv(F, ref_point=REF_POINT):
    from pymoo.indicators.hv import HV
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if F.size == 0:
        return 0.0
    return float(HV(ref_point=np.array(ref_point))(F))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — cs1_gap_grid
# ─────────────────────────────────────────────────────────────────────────────

def figGapGrid(results_dir, datasets, refs, out_dir, pop_size=20,
               study='CS1', name='cs1_gap_grid'):
    """Distance from the running incumbent to the exhaustive upper bound.

    The incumbent is whichever architecture has the best *validation* accuracy
    so far — what the search can actually see — and the curve plots that
    architecture's *test* accuracy against the test upper bound. The floor is
    therefore a genuine generalization gap, not an artifact of having
    optimized the test column.
    """
    plt = _plt()
    man = loadManifest(results_dir)
    if not len(man) or 'stem' not in man:
        return None
    man = man[man['study'] == study]
    if not len(man):
        return None

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ref = refs.get(ds)
        sub = man[man['dataset'] == ds]
        for algo in sorted(sub['algorithm'].unique()):
            curves = []
            for stem in sub[sub['algorithm'] == algo]['stem'].dropna():
                ev = loadEvals(stem)
                if ev is None or 'valid_acc' not in ev:
                    continue
                inc = ev['valid_acc'].cummax()
                # test accuracy of the validation-selected incumbent
                pick = ev['valid_acc'].expanding().apply(np.argmax, raw=True)
                test_of_inc = ev['test_acc'].to_numpy()[pick.to_numpy().astype(int)]
                gap = np.maximum(ref['best_test_acc'] - test_of_inc, 1e-6)
                gen = np.arange(1, len(gap) + 1) / pop_size
                curves.append(pd.Series(gap, index=np.ceil(gen).astype(int))
                              .groupby(level=0).min())
            if not curves:
                continue
            m = pd.concat(curves, axis=1).mean(axis=1)
            ax.plot(m.index, m.to_numpy(), label=algo.replace('_', '-'),
                    color=ALGO_COLORS.get(algo), lw=1.8)
        ax.set_yscale('log')
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=11)
        ax.set_xlabel('generation')
        ax.grid(alpha=0.25, which='both')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('test-accuracy gap to upper bound')
    axes[0][-1].legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    return _save(fig, out_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 2/3 — scalarized fronts and the alpha sweep
# ─────────────────────────────────────────────────────────────────────────────

def loadScalarPoints(results_dir, method):
    """Per-alpha winners saved by the scalarized stage."""
    d = Path(results_dir) / 'scalar_points'
    rows = []
    for p in sorted(d.glob(f'{method}_*.json')) if d.is_dir() else []:
        blob = json.loads(p.read_text())
        for pt in blob['points']:
            rows.append({'algorithm': blob['algorithm'],
                         'dataset': blob['dataset'], 'seed': blob['seed'],
                         'method': blob['method'], **pt})
    return pd.DataFrame(rows)


def figScalarFronts(results_dir, datasets, refs, out_dir, method='weighted_sum',
                    name=None):
    """Non-dominated set assembled from each optimizer's alpha runs."""
    plt = _plt()
    df = loadScalarPoints(results_dir, method)
    if not len(df):
        return None
    name = name or f'cs1_scalar_fronts_{method}'
    algos = sorted(df['algorithm'].unique())
    ncol = min(3, len(algos))
    nrow = int(np.ceil(len(algos) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow),
                             squeeze=False)
    for k, algo in enumerate(algos):
        ax = axes[k // ncol][k % ncol]
        for ds in datasets:
            ref = refs.get(ds)
            if ref is None:
                continue
            tf = np.asarray(ref['objs'])
            ax.plot(tf[:, 1], 1 - tf[:, 0], '--', color='0.6', lw=1,
                    zorder=1, label='exact front' if ds == datasets[0] else None)
            sel = df[(df['algorithm'] == algo) & (df['dataset'] == ds)]
            if not len(sel):
                continue
            F = np.column_stack([1 - sel['test_acc'].to_numpy(),
                                 _normFlops(sel['flops_M'], ref)])
            nd = _nonDominated(F)
            ax.scatter(F[nd, 1], 1 - F[nd, 0], s=26, zorder=3,
                       color=ALGO_COLORS.get(algo), marker=ALGO_MARKERS.get(algo, 'o'),
                       edgecolor='white', linewidth=0.4,
                       label=DS_LABEL.get(ds, ds))
        ax.set_title(algo.replace('_', '-'), fontsize=11)
        ax.set_xlabel('normalized #FLOPs')
        ax.set_ylabel('test accuracy')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    for k in range(len(algos), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    axes[0][0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle(f'Scalarized fronts — {method.replace("_", " ")}', fontsize=12)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def figAlphaSweep(results_dir, datasets, refs, out_dir, method='weighted_sum',
                  name=None):
    """Accuracy-cost trajectory as the weight alpha is swept end to end."""
    plt = _plt()
    df = loadScalarPoints(results_dir, method)
    if not len(df):
        return None
    name = name or f'cs1_alpha_sweep_{method}'

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df['dataset'] == ds]
        for algo in sorted(sub['algorithm'].unique()):
            g = (sub[sub['algorithm'] == algo]
                 .groupby('alpha')[['test_acc', 'flops_M']].mean()
                 .sort_index())
            ax.plot(g['flops_M'], g['test_acc'], '-',
                    marker=ALGO_MARKERS.get(algo, 'o'), ms=4, lw=1.4,
                    color=ALGO_COLORS.get(algo), label=algo.replace('_', '-'))
            if len(g):   # mark the alpha = 1 endpoint the reviewer asked for
                ax.scatter([g['flops_M'].iloc[-1]], [g['test_acc'].iloc[-1]],
                           s=90, facecolor='none',
                           edgecolor=ALGO_COLORS.get(algo, 'k'), lw=1.4, zorder=4)
        if ds in refs:
            ax.axhline(refs[ds]['best_test_acc'], color='black', ls='--', lw=1,
                       label='upper bound')
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=11)
        ax.set_xlabel('#FLOPs (M)')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('test accuracy')
    axes[0][-1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle(f'Alpha sweep — {method.replace("_", " ")} '
                 f'(circled marker = alpha 1, accuracy only)', fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — cs2_fronts_nat
# ─────────────────────────────────────────────────────────────────────────────

def figNativeFronts(results_dir, datasets, refs, out_dir, name='cs2_fronts_nat'):
    plt = _plt()
    man = loadManifest(results_dir)
    if not len(man) or 'stem' not in man:
        return None
    nat = man[man['study'] == 'CS2']
    if not len(nat):
        return None
    algos = sorted(nat['algorithm'].unique())

    fig, axes = plt.subplots(1, len(algos), figsize=(4.8 * len(algos), 4.2),
                             squeeze=False)
    for ax, algo in zip(axes[0], algos):
        for ds in datasets:
            ref = refs.get(ds)
            if ref is None:
                continue
            tf = np.asarray(ref['objs'])
            ax.plot(tf[:, 1], 1 - tf[:, 0], '--', color='0.6', lw=1, zorder=1)
            pts = []
            for stem in nat[(nat['algorithm'] == algo)
                            & (nat['dataset'] == ds)]['stem'].dropna():
                blob_p = Path(str(stem) + '.json')
                if not blob_p.exists():
                    continue
                for r in json.loads(blob_p.read_text()).get('pareto_front', []):
                    pts.append((r['test_acc'], r['flops_M']))
            if not pts:
                continue
            acc, fl = np.array(pts).T
            F = np.column_stack([1 - acc, _normFlops(fl, ref)])
            nd = _nonDominated(F)
            ax.scatter(F[nd, 1], 1 - F[nd, 0], s=24, zorder=3,
                       color=ALGO_COLORS.get(algo),
                       marker=ALGO_MARKERS.get(algo, 'o'),
                       edgecolor='white', linewidth=0.4,
                       label=DS_LABEL.get(ds, ds))
        ax.set_title(algo, fontsize=11)
        ax.set_xlabel('normalized #FLOPs')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('test accuracy')
    axes[0][-1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    return _save(fig, out_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — cs2_hv_convergence_native
# ─────────────────────────────────────────────────────────────────────────────

def figHVConvergence(results_dir, datasets, refs, out_dir, every=20,
                     name='cs2_hv_convergence_native'):
    """Hypervolume of the evaluated archive against the number of proposals."""
    plt = _plt()
    man = loadManifest(results_dir)
    if not len(man) or 'stem' not in man:
        return None
    nat = man[man['study'] == 'CS2']
    if not len(nat):
        return None

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ref = refs.get(ds)
        if ref is None:
            continue
        sub = nat[nat['dataset'] == ds]
        for algo in sorted(sub['algorithm'].unique()):
            series = []
            for stem in sub[sub['algorithm'] == algo]['stem'].dropna():
                ev = loadEvals(stem)
                if ev is None or 'test_acc' not in ev:
                    continue
                F = np.column_stack([1 - ev['test_acc'].to_numpy(),
                                     _normFlops(ev['flops_M'], ref)])
                xs = list(range(every, len(F) + 1, every))
                series.append(pd.Series([_hv(F[:k][_nonDominated(F[:k])])
                                         for k in xs], index=xs))
            if not series:
                continue
            M = pd.concat(series, axis=1)
            mean, sd = M.mean(axis=1), M.std(axis=1).fillna(0.0)
            c = ALGO_COLORS.get(algo)
            ax.plot(mean.index, mean.to_numpy(), color=c, lw=1.8,
                    label=algo.replace('_', '-'))
            ax.fill_between(mean.index, mean - sd, mean + sd, color=c, alpha=0.18)
        ax.axhline(_hv(ref['objs']), color='black', ls='--', lw=1.2,
                   label='true front')
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=11)
        ax.set_xlabel('evaluations')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].set_ylabel('hypervolume (test)')
    axes[0][-1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    return _save(fig, out_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 6/7 — duplicate evaluations
# ─────────────────────────────────────────────────────────────────────────────

def figProposalsUnique(results_dir, out_dir, budget=1020,
                       name='proposals_vs_unique'):
    """Budget spent on distinct architectures, per optimizer.

    Every method gets the same number of objective calls and is left alone to
    spend it. The filled bar is how much of that budget went to architectures
    it had not already seen — a measurement of search behaviour, not a defect
    to be corrected.
    """
    plt = _plt()
    from matplotlib.patches import Patch

    man = loadManifest(results_dir)
    if not len(man):
        return None
    d = man.dropna(subset=['n_proposals', 'n_unique'])
    if not len(d):
        return None

    agg = (d.groupby('algorithm')
             .agg(proposals=('n_proposals', 'mean'), unique=('n_unique', 'mean'),
                  sd=('n_unique', 'std'), runs=('n_unique', 'size'))
             .fillna(0.0).sort_values('unique'))
    agg['pct'] = 100.0 * agg['unique'] / agg['proposals']

    fig, ax = plt.subplots(figsize=(9, 0.52 * len(agg) + 2.0))
    y = np.arange(len(agg))
    ax.barh(y, agg['proposals'], color='#e8e8e8', edgecolor='#bdbdbd',
            height=0.62, zorder=1)
    ax.barh(y, agg['unique'], height=0.62, zorder=2, xerr=agg['sd'],
            error_kw=dict(ecolor='#444', lw=1, capsize=3),
            color=[ALGO_COLORS.get(a, '#4c72b0') for a in agg.index],
            edgecolor='none')
    for i, (_, r) in enumerate(agg.iterrows()):
        ax.text(r['unique'] + r['proposals'] * 0.012, i,
                f"{r['unique']:.0f}  ({r['pct']:.0f}%)", va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([a.replace('_', '-') for a in agg.index])
    ax.set_xlabel('architectures')
    ax.set_xlim(0, agg['proposals'].max() * 1.20)
    ax.set_title('Budget spent on distinct architectures\n'
                 f'(mean over runs; budget = {budget} proposals per run)',
                 fontsize=11)
    ax.legend(handles=[Patch(facecolor='#e8e8e8', edgecolor='#bdbdbd',
                             label='proposals (budget)'),
                       Patch(facecolor='#4c72b0', label='unique architectures')],
              loc='lower right', frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25, zorder=0)
    fig.tight_layout()
    agg.round(2).to_csv(Path(results_dir) / 'proposals_vs_unique.csv')
    return _save(fig, out_dir, name)


def figUniquePerDataset(results_dir, datasets, out_dir, budget=1020,
                        name='unique_per_dataset'):
    plt = _plt()
    man = loadManifest(results_dir)
    d = man.dropna(subset=['n_unique']) if len(man) else man
    if not len(d) or d['dataset'].nunique() < 2:
        return None

    order = d.groupby('algorithm')['n_unique'].mean().sort_values().index
    piv = (d.pivot_table(index='algorithm', columns='dataset',
                         values='n_unique', aggfunc='mean').reindex(order))
    piv = piv[[c for c in datasets if c in piv.columns]]

    fig, ax = plt.subplots(figsize=(9, 0.52 * len(piv) + 2.0))
    y = np.arange(len(piv))
    h = 0.8 / max(len(piv.columns), 1)
    shades = ['#4c72b0', '#dd8452', '#55a868']
    for k, ds in enumerate(piv.columns):
        ax.barh(y + (k - (len(piv.columns) - 1) / 2) * h, piv[ds], height=h,
                color=shades[k % len(shades)], label=DS_LABEL.get(ds, ds),
                edgecolor='none')
    ax.axvline(budget, color='black', ls='--', lw=1.2, label=f'budget ({budget})')
    ax.set_yticks(y)
    ax.set_yticklabels([a.replace('_', '-') for a in piv.index])
    ax.set_xlabel('unique architectures')
    ax.set_title('Unique architectures per dataset', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def makeAll(results_dir, datasets, out_dir=None, budget=1020, pop_size=20,
            methods=('weighted_sum', 'tchebycheff'), verbose=True):
    """Regenerate every paper figure that the available results support."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir or results_dir / 'figures')
    refs = loadRefs(results_dir, datasets)

    jobs = [
        ('cs1_gap_grid', lambda: figGapGrid(results_dir, datasets, refs,
                                            out_dir, pop_size=pop_size)),
        ('cs2_fronts_nat', lambda: figNativeFronts(results_dir, datasets, refs,
                                                   out_dir)),
        ('cs2_hv_convergence_native',
         lambda: figHVConvergence(results_dir, datasets, refs, out_dir)),
        ('proposals_vs_unique',
         lambda: figProposalsUnique(results_dir, out_dir, budget=budget)),
        ('unique_per_dataset',
         lambda: figUniquePerDataset(results_dir, datasets, out_dir,
                                     budget=budget)),
    ]
    for m in methods:
        jobs.append((f'cs1_scalar_fronts_{m}',
                     lambda m=m: figScalarFronts(results_dir, datasets, refs,
                                                 out_dir, method=m)))
        jobs.append((f'cs1_alpha_sweep_{m}',
                     lambda m=m: figAlphaSweep(results_dir, datasets, refs,
                                               out_dir, method=m)))

    made = {}
    for label, fn in jobs:
        try:
            paths = fn()
        except Exception as e:            # a missing stage must not kill the rest
            if verbose:
                print(f'  [skip] {label:30} {type(e).__name__}: {e}')
            continue
        if paths:
            made[label] = paths
            if verbose:
                print(f'  [ok  ] {label:30} -> {paths[0].name}')
        elif verbose:
            print(f'  [skip] {label:30} no data yet')
    return made


# ─────────────────────────────────────────────────────────────────────────────
# Self-test against a synthetic results tree
# ─────────────────────────────────────────────────────────────────────────────

def _synth(tmp, datasets=('cifar10', 'cifar100'), seeds=(0, 1), budget=120):
    rng = np.random.default_rng(0)
    tmp = Path(tmp)
    (tmp / 'scalar_points').mkdir(parents=True, exist_ok=True)
    rows = []

    for ds in datasets:
        fmin, fmax = 8.0, 220.0
        acc = rng.uniform(0.5, 0.94, 400)
        fl = rng.uniform(fmin, fmax, 400)
        F = np.column_stack([1 - acc, (fl - fmin) / (fmax - fmin)])
        nd = _nonDominated(F)
        (tmp / f'true_front_{ds}.json').write_text(json.dumps({
            'objs': F[nd].tolist(), 'flops_min': fmin, 'flops_max': fmax,
            'best_test_acc': float(acc.max()), 'front': []}))

        for study, algos in (('CS1', ['GA', 'SADE']),
                             ('CS2', ['NSGA-II', 'GDE3'])):
            for algo in algos:
                for seed in seeds:
                    d = tmp / algo.lower() / ds
                    d.mkdir(parents=True, exist_ok=True)
                    stem = str(d / f'{algo}_{seed}')
                    n = budget
                    a = np.clip(rng.normal(0.85, 0.05, n) +
                                np.linspace(0, 0.06, n), 0, 0.95)
                    f = rng.uniform(fmin, fmax, n)
                    pd.DataFrame({'proposal': np.arange(1, n + 1),
                                  'valid_acc': a + rng.normal(0, .002, n),
                                  'test_acc': a, 'flops_M': f}).to_csv(
                        stem + '_evals.csv', index=False)
                    Fr = np.column_stack([1 - a, (f - fmin) / (fmax - fmin)])
                    ndr = _nonDominated(Fr)
                    Path(stem + '.json').write_text(json.dumps({
                        'algorithm': algo,
                        'pareto_front': [{'test_acc': float(a[i]),
                                          'valid_acc': float(a[i]),
                                          'flops_M': float(f[i])} for i in ndr]}))
                    uniq = int(rng.integers(int(n * .15), n))
                    rows.append({'key': f'{study}|{algo}|{ds}|{seed}',
                                 'study': study, 'algorithm': algo,
                                 'dataset': ds, 'seed': seed,
                                 'n_proposals': n, 'n_unique': uniq,
                                 'stem': stem})

        for method in ('weighted_sum', 'tchebycheff'):
            for algo in ('GA', 'SADE'):
                for seed in seeds:
                    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
                    pts = [{'alpha': al,
                            'test_acc': float(0.60 + 0.30 * al + rng.normal(0, .01)),
                            'valid_acc': float(0.60 + 0.30 * al),
                            'flops_M': float(10 + 200 * al)} for al in alphas]
                    (tmp / 'scalar_points' /
                     f'{method}_{algo}_{ds}_seed{seed}.json').write_text(
                        json.dumps({'algorithm': algo, 'dataset': ds,
                                    'seed': seed, 'method': method,
                                    'points': pts}))

    pd.DataFrame(rows).to_csv(tmp / 'manifest.csv', index=False)
    return tmp


def selftest():
    import shutil
    import tempfile
    import matplotlib
    matplotlib.use('Agg')

    tmp = Path(tempfile.mkdtemp(prefix='figtest_'))
    datasets = ['cifar10', 'cifar100']
    _synth(tmp, datasets=datasets)
    print('generating figures from a synthetic results tree...')
    made = makeAll(tmp, datasets, budget=120, pop_size=20)

    expect = {'cs1_gap_grid', 'cs2_fronts_nat', 'cs2_hv_convergence_native',
              'proposals_vs_unique', 'unique_per_dataset',
              'cs1_scalar_fronts_weighted_sum', 'cs1_alpha_sweep_weighted_sum',
              'cs1_scalar_fronts_tchebycheff', 'cs1_alpha_sweep_tchebycheff'}
    missing = expect - set(made)
    empty = [n for n, ps in made.items() if ps[0].stat().st_size < 3000]

    print(f'\n{len(made)}/{len(expect)} figures produced')
    if missing:
        print('  MISSING:', sorted(missing))
    if empty:
        print('  SUSPICIOUSLY SMALL:', empty)
    ok = not missing and not empty
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if selftest() else 1)
