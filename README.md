# Improving Machine Learning Models with Artificial Intelligence Techniques

*Melhoria de Modelos de Aprendizagem Automática com Técnicas de Inteligência Artificial*

Experimental code for the master's thesis.

| | |
|---|---|
| **Author** | Ruben Café |
| **Programme** | Master's in Informatics and Multimedia Engineering (MEIM) |
| **Institution** | Instituto Superior de Engenharia de Lisboa (ISEL), Instituto Politécnico de Lisboa |
| **Department** | Department of Electronics, Telecommunications and Computer Engineering (DEETC) |
| **Supervisors** | Nuno Leite, Artur Ferreira |
| **Academic year** | 2025/2026 |

## Abstract

The thesis studies optimization algorithms, with a focus on evolutionary
machine learning (EML), to improve machine learning models through automated
hyperparameter optimization (HPO) and neural architecture search (NAS). Four
evolutionary and swarm metaheuristics, the Genetic Algorithm (GA),
Self-Adaptive Differential Evolution (SADE), Adaptive Inertia Weight Particle
Swarm Optimization (AIW-PSO) and Grey Wolf Optimization (GWO), are compared
against Bayesian Optimization (BO) in two studies, under identical budgets and
seeds:

1. **Study 1 — HPO:** optimization of 7 categorical hyperparameters of a CNN
   with a fixed template (648 configurations), on MNIST and CIFAR-10.
2. **Study 2 — NAS:** cell search on NAS-Bench-201 (CIFAR-10, CIFAR-100 and
   ImageNet16-120), with an accuracy-only, a cost-aware and a native
   multi-objective formulation, for which NSGA-II, GDE3 and qEHVI are added.

**Keywords:** Evolutionary Algorithms; Evolutionary Machine Learning;
Hyperparameter Optimization; Neural Architecture Search; Optimization
Algorithms.

## Modules

All the code is in `code/utils/`:

| Module | Contents |
|---|---|
| `datasets.py` | MNIST and CIFAR-10 loading with a train/validation/test split |
| `models.py` | Study 1 CNN template (`buildTemplateCNN`) |
| `localMealpy.py` | GA, SADE, AIW-PSO and GWO instances (`mealpy`) |
| `optimizer.py` | Study 1 search space and HPO loop with the metaheuristics |
| `optimizer_bo.py` | Study 1 Bayesian Optimization baseline |
| `worker.py` | Evaluation of a configuration in an isolated subprocess (`runOptimization(..., use_worker=True)`) |
| `nas.py` | Cell encoding and NAS-Bench-201 queries |
| `nas_revision.py` | Study 2: search protocol (validation during search, test for reporting), optimizers and metrics |
| `nas_figures.py` | Study 2 figures from the saved results |

## Installation

```bash
pip install "keras>=3.3" torch tensorflow-cpu mealpy==3.0.3 scikit-learn pandas psutil
```

```bash
pip install bayesian-optimization nats_bench pymoo botorch matplotlib
```

The first command is enough for Study 1. The second adds BO, NAS-Bench-201
and the multi-objective optimizers. The modules are imported as `utils.*`, so
the code runs from the `code/` folder. Training uses Keras with the PyTorch
backend (environment variable `KERAS_BACKEND=torch`); TensorFlow is only used
to load the data.

Study 2 also needs the benchmark file, `NATS-tss-v1_0-3ffb9-simple` (~2 GB,
available from [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)). Its path is
passed to `getApi()` or set in the `NATS_BENCH_PATH` environment variable.

## Usage

### Study 1 — CNN HPO

```python
from utils.datasets import getDataset
from utils.models import buildTemplateCNN
from utils.optimizer import decodeSolution, runOptimization
from utils.optimizer_bo import runBOOptimization

data = getDataset('cifar10', val_split=0.2, seed=42)   # or 'mnist'

# A solution is a vector of 7 indices (batch, filters, blocks, dense,
# optimizer, pooling, kernel); decodeSolution turns it into a configuration.
config = decodeSolution([1, 1, 1, 1, 0, 0, 0])
model = buildTemplateCNN(config, data['input_shape'], data['num_classes'])

# Search with a metaheuristic: 'GA', 'SADE', 'AIW_PSO' or 'GWO'.
# Budget = (n_generations + 1) * pop_size = 220 evaluations.
best_config, run = runOptimization('GWO', data, eval_epochs=1,
                                   n_generations=10, pop_size=20, seed=0)
print(run['best']['val_accuracy'])

# BO baseline with the same budget (20 random points + 200 iterations).
best_bo, run_bo = runBOOptimization(data, eval_epochs=1,
                                    init_points=20, n_iter=200, seed=0)

# Full training of the best configuration and evaluation on the test set.
model = buildTemplateCNN(best_config, data['input_shape'], data['num_classes'])
model.fit(data['x_train'], data['y_train'],
          validation_data=(data['x_val'], data['y_val']),
          epochs=50, batch_size=best_config['batch_size'])
test_loss, test_acc = model.evaluate(data['x_test'], data['y_test'])
```

`runOptimization` and `runBOOptimization` save the result (JSON) and the log
of every evaluation (CSV) under `results/`.

### Study 2 — NAS on NAS-Bench-201

```python
from utils.nas import getApi, geneToArchStr, queryArchitecture
from utils.nas_revision import (
    trueParetoFront, runScalarizedMealpy, runBOScalarized,
    runNSGA2MO, runGDE3MO, runQEHVIMO,
    runRandomSearch, runRegularizedEvolution,
)

api = getApi('path/to/NATS-tss-v1_0-3ffb9-simple')   # or $NATS_BENCH_PATH

# A cell is a vector of 6 integers, one operation (0-4) per edge.
gene = [3, 3, 1, 3, 1, 3]
print(geneToArchStr(gene))
print(queryArchitecture(api, gene, dataset='cifar10'))   # valid_acc, test_acc, flops_M, ...

# Exact Pareto front (test accuracy vs. FLOPs), by enumerating all 15,625
# cells; it is the reference front and fixes the FLOPs normalization.
front, true_objs, (fmin, fmax) = trueParetoFront(api, 'cifar10')
common = dict(dataset='cifar10', budget=1020, seed=0,
              flops_min=fmin, flops_max=fmax)

# Accuracy-only formulation (alpha=1) with a metaheuristic.
run, cache = runScalarizedMealpy(api, 'SADE', alpha=1.0, pop_size=20, **common)
print(run['best'])   # selected by valid_acc, reported with test_acc

# Cost-aware formulation: alpha < 1, weighted sum or Tchebycheff.
run, cache = runScalarizedMealpy(api, 'GWO', alpha=0.5,
                                 method='tchebycheff', pop_size=20, **common)
run, cache = runBOScalarized(api, alpha=0.5, **common)

# Native multi-objective formulation.
run, cache = runGDE3MO(api, pop_size=20, true_front_objs=true_objs, **common)
print(run['score']['hv_test'], run['score']['igd_plus_test'])
# runNSGA2MO has the same interface; runQEHVIMO takes init_points and
# batch_size instead of pop_size.

# Reference baselines.
run, cache = runRandomSearch(api, **common)
run, cache = runRegularizedEvolution(api, pop_size=20, **common)
```

Every runner accepts `results_dir=` to save the result (JSON) and the
evaluation log (CSV). `utils.nas_figures.makeAll(results_dir, datasets)`
regenerates the Study 2 figures from a folder holding those results, a
`manifest.csv` with one row per run and the reference fronts
`true_front_<dataset>.json`.
