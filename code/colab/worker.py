

from __future__ import annotations

import multiprocessing as mp
import os
import numpy as np


def _getActivationLayer(activation):
    from keras import layers
    if activation == 'leaky_relu':
        return layers.LeakyReLU()
    return layers.Activation(activation)


def _getOptimizer(name, learning_rate):
    from keras.optimizers import SGD, AdamW, RMSprop
    name = name.lower()
    if name == 'sgd':
        return SGD(learning_rate=learning_rate, nesterov=True)
    if name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    return AdamW(learning_rate=learning_rate)


def buildTemplateCNN(config, input_shape, num_classes):
    # Imports are function-local so merely importing `worker` in the
    # parent does not drag TensorFlow/Keras into parent memory.
    from keras import layers, models

    k       = config['kernel_size']
    pool_fn = layers.AveragePooling2D if config['pooling'] == 'average' else layers.MaxPooling2D
    base    = config['base_filters']

    # VGG doubling rule — design principle, NOT a search dimension.
    # With n_blocks=3 and base=64 -> [64, 128, 256].
    n_blocks = config['n_blocks']
    filters  = [base * (2 ** i) for i in range(n_blocks)]

    model = models.Sequential(name='Template_CNN')
    model.add(layers.Input(shape=input_shape))

    # Batch normalization is ALWAYS applied after every convolution:
    # design principle, not a search dimension (Ioffe & Szegedy, 2015;
    # Santurkar et al., 2018). See FIXED_BN in optimizer.py for the
    # methodological justification.
    for n_filters in filters:
        # Double conv per block (VGG-style)
        model.add(layers.Conv2D(n_filters, (k, k), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(_getActivationLayer(config['activation']))

        model.add(layers.Conv2D(n_filters, (k, k), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(_getActivationLayer(config['activation']))

        model.add(pool_fn((2, 2)))

    # Classifier head (also topologically fixed)
    model.add(layers.Flatten())
    model.add(layers.Dense(config['dense_units']))
    model.add(_getActivationLayer(config['activation']))
    if config['dropout_rate'] > 0:
        model.add(layers.Dropout(config['dropout_rate']))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=_getOptimizer(
            config['optimizer'],
            config['learning_rate'],
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset serialisation (called once in the parent)
# ─────────────────────────────────────────────────────────────────────────────

def prepareDataFile(data, dir_path: str = '/tmp/hpo_data') -> dict:
    os.makedirs(dir_path, exist_ok=True)
    paths = {
        'x_train':     os.path.join(dir_path, 'x_train.npy'),
        'y_train':     os.path.join(dir_path, 'y_train.npy'),
        'x_test':      os.path.join(dir_path, 'x_test.npy'),
        'y_test':      os.path.join(dir_path, 'y_test.npy'),
        'input_shape': tuple(data['input_shape']),
        'num_classes': int(data['num_classes']),
    }
    np.save(paths['x_train'], data['x_train'])
    np.save(paths['y_train'], data['y_train'])
    np.save(paths['x_test'],  data['x_test'])
    np.save(paths['y_test'],  data['y_test'])
    return paths


def cleanupDataFile(paths: dict) -> None:
    for key in ('x_train', 'y_train', 'x_test', 'y_test'):
        path = paths.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    dir_path = os.path.dirname(paths.get('x_train', '') or '')
    if dir_path and os.path.isdir(dir_path):
        try:
            os.rmdir(dir_path)
        except OSError:
            pass


def _childFn(config, data_paths, eval_epochs, seed, queue):
    """Run a single CNN build+fit; return ('status', val_acc) via `queue`."""
    # ── Silence TF / CUDA / absl startup noise ──────────────────────────────
    # These messages ("cuFFT/cuDNN/cuBLAS factory already registered", "XLA
    # service initialized", "Created device", etc.) are emitted from C++
    # *before* absl initialises its Python bindings — the literal warning
    # "All log messages before absl::InitializeLog() is called are written
    # to STDERR" confirms it. That means TF_CPP_MIN_LOG_LEVEL, absl.logging,
    # and tf.get_logger() all arrive too late to catch them.
    #
    # The only reliable fix is to redirect file descriptor 2 (stderr) at
    # the OS level BEFORE importing tensorflow. This also silences any
    # future C++ noise regardless of which library emits it. The trade-off
    # is that genuine C++ stack traces would also disappear — but our
    # Python-level try/except still captures exceptions via the queue,
    # so diagnostic information about eval success/failure is preserved.
    import sys as _sys
    _sys.stderr.flush()
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)   # redirect OS-level stderr → /dev/null
    os.close(_devnull_fd)

    # Belt-and-braces: also set the env vars so any child-of-child that
    # inspects them behaves quietly, and in case a future TF release
    # honours them before first-stderr-write.
    os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'   # 0=all, 3=fatal only
    os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
    os.environ['GRPC_VERBOSITY']         = 'ERROR'
    os.environ['GLOG_minloglevel']       = '2'
    os.environ['AUTOGRAPH_VERBOSITY']    = '0'

    # Deterministic per-child seeding (still before any TF import).
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)

    # Python-side absl logger — handles the W0000/E0000/I0000 lines.
    try:
        import logging
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    except Exception:
        pass

    # Import TF inside the child — parent never touches it.
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    tf.random.set_seed(seed)
    tf.config.optimizer.set_jit(False)
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Already configured — harmless.
            pass

    try:
        # Load preprocessed arrays from disk. Linux page cache makes
        # subsequent loads effectively free across sibling processes.
        x_train     = np.load(data_paths['x_train'])
        y_train     = np.load(data_paths['y_train'])
        x_test      = np.load(data_paths['x_test'])
        y_test      = np.load(data_paths['y_test'])
        input_shape = tuple(data_paths['input_shape'])
        num_classes = int(data_paths['num_classes'])

        # `buildTemplateCNN` is defined in *this* module (above),
        # so no external import is required. This is what makes the
        # file self-contained / drop-in on Kaggle.
        model = buildTemplateCNN(config, input_shape, num_classes)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=eval_epochs,
            batch_size=config['batch_size'],
            verbose=0,
        )
        val_acc = float(max(history.history['val_accuracy']))
        queue.put(('ok', val_acc))

    except tf.errors.ResourceExhaustedError as e:
        # GPU OOM — config is too big for the available VRAM.
        queue.put(('oom', str(e)[:300]))

    except Exception as e:
        # Any other error (graph build failure, NaN loss, etc.).
        queue.put(('error', f"{type(e).__name__}: {str(e)[:300]}"))


# ─────────────────────────────────────────────────────────────────────────────
# Parent-side API
# ─────────────────────────────────────────────────────────────────────────────

def evalWorker(
    config: dict,
    data_paths: dict,
    eval_epochs: int = 1,
    timeout: int = 180,
    seed: int = 42,
    start_method: str = 'spawn',
) -> tuple:
    ctx   = mp.get_context(start_method)
    queue = ctx.Queue()
    proc  = ctx.Process(
        target=_childFn,
        args=(config, data_paths, eval_epochs, seed, queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    # Timeout path: child still running when we expected it to finish.
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        _drainQueue(queue)
        return ('timeout', f'child exceeded {timeout}s')

    # Normal path: child exited; check whether it produced a result.
    try:
        if not queue.empty():
            return queue.get_nowait()
        return ('error', f'child exited (code={proc.exitcode}) with no result')
    finally:
        _drainQueue(queue)


def _drainQueue(q) -> None:
    """Empty a queue silently — avoids resource warnings on close."""
    try:
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break
    except Exception:
        pass
    try:
        q.close()
        q.join_thread()
    except Exception:
        pass
