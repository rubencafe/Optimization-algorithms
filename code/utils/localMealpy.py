from mealpy.swarm_based.PSO import AIW_PSO
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.DE import SADE
from mealpy.swarm_based.GWO import OriginalGWO


ALGORITHM_INFO = {
    'AIW_PSO': 'Adaptive Inertia Weight PSO       — w adapts per-particle from fitness rank (Qin et al., 2006)',
    'GA':  'Genetic Algorithm                 — selection, crossover, mutation',
    'SADE':  'Self-Adaptive Differential Evolution (SADE) — F/CR and strategy adapt from success history (Qin & Suganthan, 2005)',
    'GWO': 'Grey Wolf Optimizer               — wolf hierarchy',
}


def getOptimizationAlgorithm(name, n_generations, pop_size, algo_params=None):
    """Return a configured mealpy optimizer.

    name: 'AIW_PSO', 'GA', 'SADE' or 'GWO'.
    n_generations: number of generations (mealpy epochs).
    pop_size: population size.
    algo_params: optional overrides of the algorithm parameters (pc, pm, ...
        for GA; c1, c2, alpha for AIW_PSO).
    """
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
        return SADE(
            epoch=n_generations,
            pop_size=pop_size,
        )

    elif name == 'GWO':
        return OriginalGWO(
            epoch=n_generations,
            pop_size=pop_size
        )

    else:
        available = list(ALGORITHM_INFO.keys())
        raise ValueError(f"Algorithm '{name}' not available. Available: {available}")
