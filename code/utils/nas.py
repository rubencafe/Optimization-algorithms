import os

import numpy as np


OPS = [
    'none',
    'skip_connect',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3',
]

N_EDGES = 6
N_OPS   = len(OPS)


def geneToArchStr(gene):
    """Convert a cell gene into the NAS-Bench-201 architecture string.

    gene: 6 operation indices in [0, 4], one per edge, in the NATS-Bench edge
        order.
    """
    g = [int(np.round(x)) for x in gene]
    if len(g) != N_EDGES:
        raise ValueError(f"gene must have length {N_EDGES}, got {len(g)}")
    for i, v in enumerate(g):
        if not (0 <= v < N_OPS):
            raise ValueError(f"gene[{i}]={v} out of range [0, {N_OPS-1}]")
    return (
        f"|{OPS[g[0]]}~0|+"
        f"|{OPS[g[1]]}~0|{OPS[g[2]]}~1|+"
        f"|{OPS[g[3]]}~0|{OPS[g[4]]}~1|{OPS[g[5]]}~2|"
    )


_API_CACHE = {}


def getApi(file_path=None, fast_mode=True, verbose=False):
    """Load the NATS-Bench topology search space API, cached per path.

    file_path: benchmark file or folder; defaults to the NATS_BENCH_PATH
        environment variable.
    fast_mode: load results lazily instead of all at once.
    verbose: library verbosity.
    """
    from nats_bench import create

    raw = file_path if file_path is not None else os.environ.get('NATS_BENCH_PATH')
    path = str(raw) if raw is not None else None

    key = (path, fast_mode)
    if key in _API_CACHE:
        return _API_CACHE[key]

    api = create(path, 'tss', fast_mode=fast_mode, verbose=verbose)
    _API_CACHE[key] = api
    return api


def queryArchitecture(api, gene, dataset='cifar10', hp='200', is_random=False):
    """Look up one cell and return its accuracies (in [0, 1]), losses, training
    time and cost.

    api: API from getApi.
    gene: 6 operation indices in [0, 4].
    dataset: 'cifar10', 'cifar10-valid', 'cifar100' or 'ImageNet16-120'.
    hp: training epochs of the lookup, '12' or '200'.
    is_random: False averages the training seeds, True picks one at random, an
        int picks that seed.
    """
    arch_str = geneToArchStr(gene)
    index    = api.query_index_by_arch(arch_str)
    info     = api.get_more_info(index, dataset, hp=hp, is_random=is_random)
    cost     = api.get_cost_info(index, dataset, hp=hp)

    def _pct(x):
        """Convert a percentage into a fraction, keeping None.

        x: value in percent, or None.
        """
        return None if x is None else float(x) / 100.0

    return {
        'index':           int(index),
        'arch_str':        arch_str,
        'train_acc':       _pct(info.get('train-accuracy')),
        'valid_acc':       _pct(info.get('valid-accuracy')),
        'test_acc':        _pct(info.get('test-accuracy')),
        'train_loss':      info.get('train-loss'),
        'valid_loss':      info.get('valid-loss'),
        'test_loss':       info.get('test-loss'),
        'train_all_time':  info.get('train-all-time'),
        'train_per_time':  info.get('train-per-time'),
        'params_MB':       cost.get('params'),
        'flops_M':         cost.get('flops'),
        'latency':         cost.get('latency'),
    }
