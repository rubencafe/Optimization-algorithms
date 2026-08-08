
"""
irace.py
--------
Automatic algorithm configuration using racing techniques.
Iterated Racing for Automatic Algorithm Configuration

Implemented in Python using Optuna's TPE sampler + Hyperband pruner.

How it works:
    1. For each algorithm, define a space of its own hyperparameters
    
    2. Optuna proposes candidate configurations and races them

    3. Each race runs the algorithm with a reduced budget (fewer iterations)
       and reports val_accuracy back to Optuna

    4. Optuna's Hyperband pruner eliminates poor configurations early

    5. After n_trials races, the best configuration is returned

    6. That configuration is then used in the full runOptimization call

Tunable parameters per algorithm (raced by Optuna's TPE):
    PSO → c1     : cognitive coefficient      [1.0, 3.0]
          c2     : social coefficient         [1.0, 3.0]
          alpha  : w-adaptation amplitude     [0.0, 1.0]
          (AIW_PSO — w is NOT a hyperparameter; it is adapted per-particle
           from fitness rank every iteration — Qin et al., 2006)

    GA  → pc  : crossover probability  [0.6, 1.0]
          pm  : mutation probability   [0.001, 0.1]
          + selection / crossover / mutation operator choices

    DE  → (no configurable parameters — SADE self-adapts F, CR and mutation
           strategy from success history every iteration; Qin & Suganthan, 2005)

    BSA → ff, pff, c1, c2, a1, a2, fc
    GWO → (no configurable parameters)

Usage:
    from Utils.irace import configureOptimizationAlgorithmIRACE, configureAll

    data = getDataset('cifar10')

    # Configure a single algorithm
    best_params = configureOptimizationAlgorithmIRACE('PSO', data, n_trials=20)
    print(best_params)  # {'w': 0.72, 'c1': 1.8, 'c2': 2.1}

    # Use in the main optimization run
    best, history = runOptimization('PSO', data, algo_params=best_params)

    # Configure all algorithms at once
    all_params = configureAll(['PSO', 'GA', 'DE'], data, n_trials=20)
    # all_params = {'PSO': {...}, 'GA': {...}, 'DE': {...}}
"""

import json
import os
import time
import numpy as np
import optuna
from datetime import datetime
import matplotlib.pyplot as plt

from code.utils.localMealpy import getLiteratureDefaults
from code.utils.optimizer import runOptimization

# Suppress Optuna's verbose logging — we handle our own output
optuna.logging.set_verbosity(optuna.logging.WARNING)




# Os hiperparâmetros de um GA/PSO/DE só "se expressam" ao longo de várias gerações. O F do DE, o cognitive/social do PSO,
# a pressão de seleção do GA — tudo isto rege a trajetória da população no tempo. Se tens, digamos, população 10 e 5 gerações (50 avaliações),
# estás a observar essencialmente 5 passos de uma caminhada estocástica a partir de um ponto aleatório. A variância da posição inicial domina quase tudo.
# Um F=0.5 vs F=0.9 precisa de umas dezenas de gerações para produzir uma diferença de média que sobreviva ao ruído entre sementes.
# Depois da transição para variantes auto-adaptativas (PSO→AIW_PSO, DE→SADE),
# os parâmetros "lentos" desapareceram do racing por construção:
# - AIW_PSO adapta `w` por partícula em cada iteração a partir do rank de fitness;
# - SADE adapta F, CR e a estratégia de mutação a partir do histórico de sucessos.
# Portanto nenhum desses escalares precisa de ser nem fixado nem raced.
def _suggestParams(trial, algo_name):
    if algo_name == 'AIW_PSO':
        return {
            'c1':    trial.suggest_float('c1',    1.0, 3.0),
            'c2':    trial.suggest_float('c2',    1.0, 3.0),
            'alpha': trial.suggest_float('alpha', 0.0, 1.0),
        }
    
    elif algo_name == 'GA':
        mutation_multipoints = trial.suggest_categorical('mutation_multipoints', [True, False])
        if mutation_multipoints:
            mutation_choices = ['flip', 'swap']
        else:
            mutation_choices = ['flip', 'swap', 'scramble', 'inversion']
        return {
            'pc':                   trial.suggest_float('pc', 0.6, 1.0),
            'pm':                   trial.suggest_float('pm', 0.001, 0.1),
            'selection':            trial.suggest_categorical('selection', ['roulette', 'tournament', 'random']),
            'k_way':                trial.suggest_float('k_way', 0.1, 0.5),
            'crossover':            trial.suggest_categorical('crossover', ['one_point', 'multi_points', 'uniform', 'arithmetic']),
            'mutation_multipoints': mutation_multipoints,
            'mutation':             trial.suggest_categorical('mutation', mutation_choices),
        }
    elif algo_name == 'BSA':
        return {
            'ff':  trial.suggest_int  ('ff',  5,   20),
            'pff': trial.suggest_float('pff', 0.1,  1.0),
            'c1':  trial.suggest_float('c1',  0.5,  3.0),
            'c2':  trial.suggest_float('c2',  0.5,  3.0),
            'a1':  trial.suggest_float('a1',  0.5,  3.0),
            'a2':  trial.suggest_float('a2',  0.5,  3.0),
            'fc':  trial.suggest_float('fc',  0.1,  1.0),
        }

    elif algo_name == 'SADE':
        # SADE is fully self-adaptive: F, CR, and the mutation strategy are
        # updated every iteration from the success history of previous trials.
        # Nothing for Optuna to tune here.
        return {}

    elif algo_name == 'GWO':
        # Grey Wolf Optimizer has no behavioural hyperparameters exposed.
        return {}

    else:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Single algorithm configuration
# ─────────────────────────────────────────────────────────────────────────────
# O pop_size é um parâmetro de budget computacional, não um parâmetro de comportamento do algoritmo. 
# O que o IRACE/Optuna deve encontrar é como o algoritmo se comporta — 
# a inércia do PSO, a taxa de mutação do GA, o factor de escala do DE. 
# O pop_size afecta quantos agentes exploram o espaço, não como exploram.
def configureOptimizationAlgorithmIRACE(
    algorithm,
    data,
    n_trials        = 15,
    n_generations   = 3,
    pop_size        = 10,
    eval_epochs     = 1,
    n_repeat_trials = 1,
    results_dir     = 'results/irace',
    seed            = 42,
    extra_info      = True,
):
    alg_name       = algorithm.upper()
    fixed_pop_size = pop_size 

    print(f"\n{'='*62}")
    print(f"  IRACE — Configuring {alg_name}")
    print(f"  Trials: {n_trials}  |  Repetitions/trial: {n_repeat_trials}  |  "
          f"pop_size: {fixed_pop_size} (fixed)  |  n_generations: {n_generations}  |  epochs: {eval_epochs}")
    print(f"{'='*62}\n")

    t_start = time.time()

    def optunaObjective(trial):
        # Everything the algorithm needs beyond pop_size/epochs is sampled by
        # Optuna's TPE this trial — after switching to self-adaptive variants
        # (AIW_PSO, SADE) there are no literature constants to merge in.
        algo_params = _suggestParams(trial, alg_name)

        params_str = '  '.join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in algo_params.items()
        )
        print(f"  [Trial {trial.number+1:>3}/{n_trials}]  pop_size={fixed_pop_size}  {params_str}")

        # ── SEED BLOCKING for variance reduction ────────────────────────────
        # Every trial is evaluated on the SAME set of seeds
        # {seed, seed+1, ..., seed+n_repeat_trials-1}. This is the blocking
        # design used by IRACE (López-Ibáñez et al., 2016): if we allowed each
        # trial to use different seeds, the variance between trials would be a
        # mixture of (i) configuration effect and (ii) seed effect — making it
        # impossible to attribute observed differences to the configuration.
        # By blocking on seeds, seed-induced variance cancels across pairs of
        # trials and only configuration effect remains.
        #
        # NOTE: The previous formula `seed + trial.number * 100 + i` assigned
        # a different seed set to every trial — a bug that defeated blocking.
        # Fixed as of Phase 1 of the tuning pipeline refactor.
        # ─────────────────────────────────────────────────────────────────────
        scores = []
        for i in range(n_repeat_trials):
            _, run_results = runOptimization(
                alg_name,
                data,
                eval_epochs   = eval_epochs,
                n_generations = n_generations,
                pop_size      = fixed_pop_size,
                results_dir   = os.path.join(results_dir, 'races'),
                seed          = seed + i,   # BLOCKING: identical seed set per trial
                algo_params   = algo_params,
                extra_info    = extra_info,
            )
            scores.append(run_results['best']['val_accuracy'])

        return float(np.mean(scores))

    study = optuna.create_study(
        direction = 'maximize',
        sampler   = optuna.samplers.TPESampler(
            seed         = seed,
            multivariate = True,   # learns correlations between parameters
            group        = True,   # handles conditional parameters (e.g. mutation)
        ),
    )
    study.optimize(optunaObjective, n_trials=n_trials, show_progress_bar=False)

    t_total = time.time() - t_start
    best_params = study.best_params
    best_value  = study.best_value

    print(f"\n{'='*62}")
    print(f"  IRACE RESULT — {alg_name}")
    print(f"{'='*62}")
    print(f"  Best val_accuracy (race) : {best_value:.4f}")
    print(f"  Configuration time       : {t_total:.0f}s  ({t_total/60:.1f} min)")
    print(f"  Fixed pop_size           : {fixed_pop_size}")
    print(f"  Best parameters found:")
    for k, v in best_params.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<24}  {v_str}")
    print()

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result = {
        'algorithm':    alg_name,
        'timestamp':    datetime.now().isoformat(),
        'n_trials':     n_trials,
        'race_budget':  {'n_generations': n_generations, 'pop_size': fixed_pop_size, 'eval_epochs': eval_epochs},
        'best_params':  best_params,
        'best_val_accuracy_race': round(best_value, 4),
        'config_time_s': round(t_total, 1),
        'all_trials': [
            {
                'trial':        t.number,
                'params':       t.params,
                'val_accuracy': t.value,
            }
            for t in study.trials
        ],
    }
    json_path = os.path.join(results_dir, f'irace_{alg_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved: {json_path}")

    # Return both best_params and the study object
    # study is needed for optuna.visualization plots in notebooks
    return best_params, study



# ─────────────────────────────────────────────────────────────────────────────
# Parameter importance analysis  (fANOVA via Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def analyzeParamImportance(study, algo_name, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Try fANOVA first, fall back to tree-based importance
    try:
        from optuna.importance import get_param_importances, FanovaImportanceEvaluator
        importances = get_param_importances(study, evaluator=FanovaImportanceEvaluator())
        method = 'fANOVA (Hutter et al., 2014)'
    except Exception:
        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(study)
            method = 'Mean Decrease Impurity (tree-based)'
        except Exception as e:
            print(f"  [ERROR] Importance analysis unavailable: {e}")
            return {}

    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  PARAMETER IMPORTANCE — {algo_name.upper()}")
    print(f"  Method: {method}")
    print(sep)
    print(f"  {'Parameter':<22}  {'Score':>8}  Bar")
    print(f"  {'-'*54}")
    for param, score in importances.items():
        bar = '█' * max(1, int(score * 35))
        print(f"  {param:<22}  {score:>8.4f}  {bar}")
    print()
    print("  Interpretation:")
    print("    High score → algorithm is sensitive to this parameter on your problem")
    print("    Low score  → parameter has little effect; literature default is acceptable")
    print()

    # Horizontal bar chart
    names  = list(importances.keys())
    scores = list(importances.values())
    max_s  = max(scores) if scores else 1
    colors = ['#e74c3c' if s == max_s else '#3498db' for s in scores]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.3), 4))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1],
                   alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(
        f'Parameter Importance — {algo_name.upper()}\n'
        f'({method})',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim([0, max_s * 1.25])
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Importance chart saved: {save_path}")
    plt.show()

    return dict(importances)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical validation  (tuned vs. baseline, Mann-Whitney U)
# ─────────────────────────────────────────────────────────────────────────────

def _plotValidationBoxplot(tuned_scores, baseline_scores, algo_name, p_value, save_path):
    """Internal: box plot comparing tuned vs baseline val_accuracy distributions."""

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [baseline_scores, tuned_scores],
        patch_artist=True, notch=False, widths=0.5,
    )
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.set_xticklabels(['Baseline\n(Literature)', 'Tuned\n(IRACE/Optuna)'], fontsize=11)
    ax.set_ylabel('Val Accuracy', fontsize=11)
    sig_str = f'p={p_value:.4f} {"★ p<0.05 significant" if p_value < 0.05 else "(not significant)"}'
    ax.set_title(
        f'Tuned vs. Baseline Configuration — {algo_name}\n{sig_str}',
        fontsize=12, fontweight='bold'
    )
    ax.grid(True, axis='y', alpha=0.3)

    # Overlay individual run points with jitter
    for i, scores in enumerate([baseline_scores, tuned_scores], start=1):
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(scores))
        ax.scatter([i + j for j in jitter], scores,
                   color='black', alpha=0.55, s=22, zorder=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Box plot saved: {save_path}")


def validateConfiguration(
    algo_name,
    tuned_params,
    data,
    baseline_params = None,
    n_runs          = 10,
    n_generations   = 15,
    pop_size        = 25,
    eval_epochs     = 5,
    results_dir     = 'results/irace',
    seed            = 42,
):
    from scipy.stats import mannwhitneyu
    import numpy as np

    algo_name = algo_name.upper()
    os.makedirs(results_dir, exist_ok=True)

    # ── Resolve baseline ──────────────────────────────────────────────────────
    if baseline_params is None:
        lit            = getLiteratureDefaults(algo_name)
        baseline_pop   = lit['pop_size']
        baseline_algo  = lit['params']
        baseline_label = 'Literature defaults'
    else:
        bp             = dict(baseline_params)
        baseline_pop   = bp.pop('pop_size', pop_size)
        baseline_algo  = bp
        baseline_label = 'Custom baseline'

    # ── tuned_params contains only internal algorithm parameters (no pop_size) ─
    tuned_algo = dict(tuned_params)
    tuned_pop  = pop_size

    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  STATISTICAL VALIDATION — {algo_name}")
    print(f"  Runs per configuration : {n_runs}  (independent seeds)")
    print(f"  n_generations={n_generations}  pop_size={pop_size}  eval_epochs={eval_epochs}")
    print(f"  Baseline: {baseline_label}")
    print(sep)

    def _run_config(algo_params, pop_sz, label):
        scores = []
        for i in range(n_runs):
            print(f"  [{label}] run {i+1:>2}/{n_runs} ...", end=' ', flush=True)
            _, run_res = runOptimization(
                algo_name, data,
                eval_epochs  = eval_epochs,
                n_generations  = n_generations,
                pop_size     = pop_sz,
                results_dir  = os.path.join(results_dir, 'validation_races'),
                seed         = seed + i * 100,
                algo_params  = algo_params,
            )
            acc = run_res['best']['val_accuracy']
            scores.append(acc)
            print(f"val_acc={acc:.4f}")
        return scores

    tuned_scores    = _run_config(tuned_algo,   tuned_pop,   'TUNED   ')
    baseline_scores = _run_config(baseline_algo, baseline_pop, 'BASELINE')

    # ── Statistics ────────────────────────────────────────────────────────────
    t = np.array(tuned_scores)
    b = np.array(baseline_scores)

    _, p_value   = mannwhitneyu(t, b, alternative='two-sided')
    significant  = bool(p_value < 0.05)

    if t.mean() > b.mean():
        winner = 'tuned' if significant else 'tie (not significant)'
    elif b.mean() > t.mean():
        winner = 'baseline' if significant else 'tie (not significant)'
    else:
        winner = 'tie'

    print(f"\n{sep}")
    print(f"  RESULTS — {algo_name}")
    print(sep)
    print(f"  {'Config':<14}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*56}")
    print(f"  {'TUNED':<14}  {t.mean():>8.4f}  {t.std():>8.4f}  {t.min():>8.4f}  {t.max():>8.4f}")
    print(f"  {'BASELINE':<14}  {b.mean():>8.4f}  {b.std():>8.4f}  {b.min():>8.4f}  {b.max():>8.4f}")
    print()
    print(f"  Mann-Whitney U p-value       : {p_value:.4f}")
    print(f"  Significant (p < 0.05)       : {'YES ✓' if significant else 'NO'}")
    print(f"  Winner                       : {winner.upper()}")
    print()
    if not significant:
        print("  NOTE: The difference is not statistically significant.")
        print("        Consider increasing n_runs (≥30 for publication) or n_trials.")
    print()

    result = {
        'algorithm':      algo_name,
        'n_runs':         n_runs,
        'baseline_label': baseline_label,
        'tuned': {
            'params':   tuned_algo,
            'pop_size': tuned_pop,
            'scores':   tuned_scores,
            'mean':     float(t.mean()),
            'std':      float(t.std()),
            'min':      float(t.min()),
            'max':      float(t.max()),
        },
        'baseline': {
            'params':   baseline_algo,
            'pop_size': baseline_pop,
            'scores':   baseline_scores,
            'mean':     float(b.mean()),
            'std':      float(b.std()),
            'min':      float(b.min()),
            'max':      float(b.max()),
        },
        'p_value':    float(p_value),
        'significant': significant,
        'winner':      winner,
    }

    # Save JSON
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(results_dir, f'validation_{algo_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Validation JSON saved: {json_path}")

    # Box plot
    plot_path = os.path.join(results_dir, f'validation_{algo_name}_{ts}.png')
    _plotValidationBoxplot(tuned_scores, baseline_scores, algo_name, p_value, plot_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full scientific pipeline
# ─────────────────────────────────────────────────────────────────────────────

def configureAndValidate(
    algo_name,
    data,

    n_trials         = 15,
    irace_generations= 3,
    irace_pop_size   = 10,
    irace_eval_epochs= 3,

    n_runs           = 10,
    n_generations    = 15,
    eval_epochs      = 2,
    
    results_dir      = 'results/irace',
    seed             = 42,
):
    algo_name = algo_name.upper()
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Literature defaults ────────────────────────────────────────────────
    lit = getLiteratureDefaults(algo_name)

    # ── 2. IRACE with reduced budget (racing phase) ────────────────────────────
    best_params, study = configureOptimizationAlgorithmIRACE(
        algo_name,
        data,
        n_trials       = n_trials,
        n_generations  = irace_generations,
        pop_size       = irace_pop_size,
        eval_epochs    = irace_eval_epochs,
        results_dir    = results_dir,
        seed           = seed,
    )

    # ── 3. Parameter importance analysis ─────────────────────────────────────
    importance_path = os.path.join(results_dir, f'importance_{algo_name}.png')
    importances     = analyzeParamImportance(study, algo_name, save_path=importance_path)

    # ── 4. Statistical validation ─────────────────────────────────────────────
    # pop_size comes from literature defaults — both tuned and baseline configs
    # run with the same population size to ensure a fair comparison.
    validation = validateConfiguration(
        algo_name,
        dict(best_params),
        data,
        baseline_params = None, # auto-uses literature defaults
        n_runs          = n_runs,
        n_generations   = n_generations,
        pop_size        = lit['pop_size'],
        eval_epochs     = eval_epochs,
        results_dir     = results_dir,
        seed            = seed,
    )

    # ── 5. Save comprehensive report ──────────────────────────────────────────
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'algorithm':  algo_name,
        'timestamp':  datetime.now().isoformat(),
        'literature_defaults': {
            'params':   lit['params'],
            'pop_size': lit['pop_size'],
        },
        'tuning': {
            'n_trials':        n_trials,
            'race_n_generations': irace_generations,
            'race_eval_epochs': irace_eval_epochs,
            'best_params':     best_params,
            'best_val_acc_race': round(study.best_value, 4),
        },
        'importance': importances,
        'validation': {k: v for k, v in validation.items()},
    }
    report_path = os.path.join(results_dir, f'full_report_{algo_name}_{ts}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  CONFIGURE & VALIDATE COMPLETE — {algo_name}")
    print(sep)
    print(f"  Literature baseline   : {lit['notes'][:60]}...")
    print(f"  Best params (tuned)   : {best_params}")
    print(f"  Most important param  : {next(iter(importances), 'n/a')}")
    print(f"  Validation winner     : {validation['winner'].upper()}")
    print(f"  p-value               : {validation['p_value']:.4f}  "
          f"({'significant' if validation['significant'] else 'not significant'})")
    print(f"\n  Full report saved     : {report_path}")
    print()

    return {
        'literature_defaults': lit,
        'best_params':         best_params,
        'study':               study,
        'importances':         importances,
        'validation':          validation,
    }
