"""Locate the memory growth in the NAS pipeline, using the REAL benchmark API.

Runs a handful of scalarized searches and reports, after each one:

  * process RSS (the number that actually matters)
  * the size of every dict/list attribute hanging off the NATS-Bench api
  * our own memoization cache
  * which Python types grew the most

Run it standalone so Jupyter's output history is not part of the picture:

    cd code && python diagnose_memory.py
"""

import ctypes
import gc
import io
import contextlib
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── process RSS, without needing psutil ─────────────────────────────────────
def rssMiB():
    if os.name == 'nt':
        import ctypes.wintypes as wt          # Windows-only; keep it local

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 2**20
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float('nan')


def apiContainers(api, top=8):
    """Sizes of the biggest container attributes on the api object."""
    out = []
    for name in dir(api):
        if name.startswith('__'):
            continue
        try:
            v = getattr(api, name)
        except Exception:
            continue
        if isinstance(v, (dict, list, set, tuple)) and len(v):
            out.append((name, type(v).__name__, len(v)))
    return sorted(out, key=lambda x: -x[2])[:top]


def typeCounts():
    c = Counter()
    for o in gc.get_objects():
        c[type(o).__name__] += 1
    return c


def main():
    from utils.nas import getApi
    import utils.nas as nas
    import utils.nas_mo as mo

    root = os.path.dirname(os.path.abspath(__file__))
    nas_path = os.path.join(root, 'utils', 'nas_records',
                            'NATS-tss-v1_0', 'NATS-tss-v1_0')
    print('benchmark :', nas_path)
    print('RSS antes de carregar a API :', f'{rssMiB():.0f} MiB')

    api = getApi(nas_path)
    gc.collect()
    print('RSS depois de carregar a API:', f'{rssMiB():.0f} MiB')
    print('containers da API           :', apiContainers(api))

    has_clear = hasattr(api, 'clear_params')
    print('api.clear_params existe?    :', has_clear)
    print()

    fmin, fmax = 7.78305, 220.11969
    base_types = typeCounts()
    prev = rssMiB()

    print(f'{"run":>4} | {"RSS MiB":>8} | {"delta":>7} | '
          f'{"nossa cache":>11} | maiores containers da API')
    print('-' * 100)

    for k in range(1, 9):
        with contextlib.redirect_stdout(io.StringIO()):
            mo.runMONASScalarized(
                'GA', api, dataset='cifar10', hp='200', alpha=0.5,
                n_generations=50, pop_size=20, seed=k,
                flops_min=fmin, flops_max=fmax,
                results_dir=os.path.join(root, '_diag_out'), verbose=False)
        gc.collect()
        cur = rssMiB()
        conts = apiContainers(api, top=3)
        print(f'{k:>4} | {cur:8.0f} | {cur - prev:+7.0f} | '
              f'{len(nas._QUERY_CACHE):>11} | {conts}')
        prev = cur

    print('\n=== tipos que mais cresceram ===')
    now = typeCounts()
    grew = sorted(((now[t] - base_types.get(t, 0), t) for t in now),
                  reverse=True)[:12]
    for d, t in grew:
        if d > 0:
            print(f'  {d:>9,}  {t}')

    print('\n=== leitura ===')
    print('  Se a RSS sobe e um container da API cresce junto -> e a API.')
    print('  Se a RSS sobe e nada cresce -> fragmentacao ou memoria nativa')
    print('  (torch/gpytorch/numpy) fora do alcance do gc.')

    # ── e o despejo? sonda controlada, para nao termos de adivinhar depois ──
    print('\n=== e possivel despejar a cache interna da API? ===')
    print('  classe da API :', ' -> '.join(c.__name__ for c in type(api).__mro__[:3]))
    for name in ('arch2infos_dict', 'evaluated_indexes', 'arch2infos_full',
                 'arch2infos_less', '_all_infos'):
        v = getattr(api, name, None)
        print(f'  {name:20} {type(v).__name__ if v is not None else "-":<12}'
              f'{len(v) if isinstance(v, (dict, list, set)) else ""}')

    store = getattr(api, 'arch2infos_dict', None)
    if not isinstance(store, dict) or not store:
        print('\n  -> arch2infos_dict nao encontrado/vazio. Diz-me qual dos')
        print('     containers acima cresceu na tabela e ataco esse.')
        return

    victim = next(iter(store))
    before = len(store)
    ev = getattr(api, 'evaluated_indexes', None)
    store.pop(victim, None)
    if isinstance(ev, set):
        ev.discard(victim)
    elif isinstance(ev, list) and victim in ev:
        ev.remove(victim)
    print(f'\n  despejei o indice {victim}: {before} -> {len(store)} entradas')

    try:
        from utils.nas import clearQueryCache, queryValidTest
        clearQueryCache()
        gene = [0, 0, 0, 0, 0, 0]
        r = queryValidTest(api, gene, dataset='cifar10', hp='200')
        ok = r.get('test_acc') is not None
        print(f'  consulta depois do despejo: {"OK" if ok else "FALHOU"} '
              f'(test_acc={r.get("test_acc")})')
        print(f'  store voltou a {len(store)} entradas (recarregou sob procura)')
        print('\n  -> despejo VIAVEL. Posso limitar a memoria da API a um run.')
    except Exception as e:
        print(f'  consulta depois do despejo REBENTOU: {type(e).__name__}: {e}')
        print('\n  -> despejo NAO viavel por esta via; uso recriacao periodica')
        print('     da API (limpar _API_CACHE) em vez disto.')


if __name__ == '__main__':
    main()
