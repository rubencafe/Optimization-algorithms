from mealpy import FloatVar
from mealpy.swarm_based.PSO import AIW_PSO
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.DE import SADE
from mealpy.swarm_based.BSA import OriginalBSA
from mealpy.swarm_based.GWO import OriginalGWO


ALGORITHM_INFO = {
    'AIW_PSO': 'Adaptive Inertia Weight PSO       — w adapts per-particle from fitness rank (Qin et al., 2006)',
    'GA':  'Genetic Algorithm                 — selection, crossover, mutation',
    'SADE':  'Self-Adaptive Differential Evolution (SADE) — F/CR and strategy adapt from success history (Qin & Suganthan, 2005)',
    'BSA': 'Backtracking Search Algorithm     — mutation with historical population memory',
    'GWO': 'Grey Wolf Optimizer               — wolf hierarchy',
    'NSGA2': 'NSGA-II (Non-dominated Sorting GA II) — Pareto-based multi-objective EA (Deb et al., 2002). '
             'Implemented via pymoo; not a mealpy algorithm.',
    'GDE3':  'GDE3 (Generalised Differential Evolution 3) — native multi-objective DE (Kukkonen & '
             'Lampinen, 2005). Implemented via pymoo; not a mealpy algorithm. '
             'Use utils.nas_mo.runGDE3NASSearch().',
    'qEHVI': 'qEHVI (q-Expected Hypervolume Improvement) — native multi-objective Bayesian optimisation '
             '(Daulton et al., 2020). Implemented via BoTorch; not a mealpy algorithm. '
             'Use utils.nas_mo.runQEHVINASSearch().',
}
    
# ─────────────────────────────────────────────────────────────────────────────
# Literature-grounded default parameters
# ─────────────────────────────────────────────────────────────────────────────

# Each entry contains the parameters as they were proposed in the original paper,
# a full bibliographic reference for thesis use, and practical notes.
_LITERATURE_DEFAULTS = {
    # qin2006adaptive
    'PSO': {
        'params': {'c1': 2.05, 'c2': 2.05, 'alpha': 0.4},
        'pop_size': 30,
        'notes': (
            'AIW_PSO — Adaptive Inertia Weight PSO (Qin, Yu, Shi & Wang, 2006). '
            'The inertia weight `w` is NOT a hyperparameter: every iteration, each '
            'particle receives its own w computed from the rank of its fitness within '
            'the current swarm — fit particles get a low w (exploit around their pbest), '
            'unfit particles get a high w (keep exploring). `alpha` controls the '
            'amplitude of this adaptation. Contrary to schedule-based variants such as '
            'LDW_PSO or HPSO-TVAC, AIW_PSO does not require long epoch budgets, because '
            'the adaptation reacts to the state of the swarm rather than to the '
            'iteration number. Defaults c1=c2=2.05, alpha=0.4 follow the mealpy '
            'implementation of the original paper.'
        ),
    },
    # goldberg1989genetic, dejong1975analysis, grefenstette1986optimization
    'GA': {
        'params': {
            'pc': 0.8, 'pm': 0.01,
            'selection': 'tournament', 'k_way': 0.2,
            'crossover': 'one_point',
            'mutation_multipoints': False, 'mutation': 'flip',
        },
        'pop_size': 50,
        'notes': (
            'pc ∈ [0.6, 0.9]: crossover probability; higher values preserve more genetic '
            'material. pm ≈ 1/L where L is chromosome length (here L≈100 → pm≈0.01); '
            'too-high pm turns GA into random search. Tournament selection with k_way=0.2 '
            '(20%% of population) is robust across problem types and avoids premature '
            'convergence better than roulette. One-point crossover is the canonical baseline.'
        ),
    },
    # qin2005self
    'DE': {
        'params': {},
        'pop_size': 50,
        'notes': (
            'SADE — Self-Adaptive Differential Evolution (Qin & Suganthan, 2005). '
            'F (scale factor), CR (crossover rate) and the mutation strategy are all '
            'self-adapted from the success history of previous trial vectors: strategies '
            'that produced better offspring get higher sampling probability; CR adapts '
            'around the running mean of successful values; F is drawn from a Cauchy/normal '
            'distribution. Unlike time-scheduled adaptive variants (e.g. L-SHADE), SADE\'s '
            'adaptation is success-driven and therefore works at the short epoch budgets '
            'used in HPO. There are no user-settable parameters beyond pop_size.'
        ),
    },
    # civicioglu2013backtracking
    'BSA': {
        'params': {'ff': 10, 'pff': 0.8, 'c1': 2.0, 'c2': 2.0, 'a1': 1.5, 'a2': 1.5, 'fc': 0.5},
        'pop_size': 30,
        'notes': (
            'ff (mixrate) ∈ [1, problem_dim]: controls how many dimensions are mutated '
            'per individual (typically 5–20). pff ∈ [0.1, 1.0]: probability of using the '
            'historical population for the mutation direction. c1, c2 are velocity scaling '
            'factors; a1, a2 are amplitude factors; fc is a form factor controlling step size. '
            'All values are as recommended in the original paper.'
        ),
    },
    # mirjalili2014grey
    'GWO': {
        'params': {},
        'pop_size': 30,
        'notes': (
            'GWO has no configurable internal hyperparameters — the hunting strategy '
            'coefficients a, A, C are computed deterministically from the iteration counter. '
            'Population size (typically 20–50) is the only tunable parameter. '
            'The original paper used pop_size=30 across benchmark functions.'
        ),
    },
}

def getLiteratureDefaults(algo_name):
    name = algo_name.upper()
    if name not in _LITERATURE_DEFAULTS:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(_LITERATURE_DEFAULTS)}")

    entry = _LITERATURE_DEFAULTS[name]

    print(f"\n{'='*62}")
    print(f"  LITERATURE DEFAULTS — {name}")
    print(f"{'='*62}")
    print(f"  Notes    : {entry['notes']}")
    print(f"\n  pop_size : {entry['pop_size']}")
    print(f"  Parameters:")
    if entry['params']:
        for k, v in entry['params'].items():
            print(f"    {k:<10}  {v}")
    else:
        print(f"    (none — algorithm has no internal hyperparameters)")
    print()

    return {
        'params':   dict(entry['params']),
        'pop_size': entry['pop_size'],
        'notes':    entry['notes'],
    }

def getOptimizationAlgorithm(name, n_generations, pop_size, algo_params=None):
    name = name.upper()
    p    = algo_params or {}

    if name == 'AIW_PSO':
        return AIW_PSO(
            epoch=n_generations,
            pop_size=pop_size,
            c1=p.get('c1', 2.05),
            c2=p.get('c2', 2.05),
            alpha=p.get('alpha', 0.4),
        )

    elif name == 'GA':
        return BaseGA(
            epoch=n_generations,
            pop_size=pop_size,
            pc=p.get('pc', 0.9),
            pm=p.get('pm', 0.05),
            selection=p.get('selection', 'tournament'),
            k_way=p.get('k_way', 0.2),
            crossover=p.get('crossover', 'uniform'),
            mutation_multipoints=p.get('mutation_multipoints', True),
            mutation=p.get('mutation', 'flip')
        )

    elif name == 'SADE':
        # SADE — Self-Adaptive Differential Evolution (Qin & Suganthan, 2005).
        # Neither F (scale factor), CR (crossover rate), nor the mutation
        # strategy are user-set: they are updated every iteration from the
        # success history of previous trials. IRACE has nothing to tune here.
        return SADE(
            epoch=n_generations,
            pop_size=pop_size,
        )

    elif name == 'BSA':
        return OriginalBSA(
            epoch=n_generations,
            pop_size=pop_size,
            ff=p.get('ff', 10),
            pff=p.get('pff', 0.8),
            c_couples=[p.get('c1', 2.0), p.get('c2', 2.0)],
            a_couples=[p.get('a1', 1.5), p.get('a2', 1.5)],
            fc=p.get('fc', 0.5)
        )

    elif name == 'GWO':
        return OriginalGWO(
            epoch=n_generations,
            pop_size=pop_size
        )

    elif name == 'NSGA2':
        # NSGA-II is implemented via pymoo (not mealpy).
        # Use utils.nas_mo.runNSGAIINASSearch() instead of this function.
        raise NotImplementedError(
            "NSGA-II is not a mealpy algorithm. "
            "Use utils.nas_mo.runNSGAIINASSearch() directly."
        )

    else:
        available = list(ALGORITHM_INFO.keys())
        raise ValueError(f"Algorithm '{name}' not available. Available: {available}")