from __future__ import annotations

import multiprocessing as mp
import os
import numpy as np


def _getActivationLayer(activation):
    """Return the Keras activation layer for a name.

    activation: activation name, e.g. 'relu' or 'leaky_relu'.
    """
    from keras import layers
    if activation == 'leaky_relu':
        return layers.LeakyReLU()
    return layers.Activation(activation)


def _getOptimizer(name, learning_rate):
    """Return the Keras optimizer for a name.

    name: 'adamw', 'sgd' (with Nesterov momentum) or 'rmsprop'.
    learning_rate: optimizer learning rate.
    """
    from keras.optimizers import SGD, AdamW, RMSprop
    name = name.lower()
    if name == 'sgd':
        return SGD(learning_rate=learning_rate, nesterov=True)
    if name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    return AdamW(learning_rate=learning_rate)


def buildTemplateCNN(config, input_shape, num_classes):
    """Build and compile the Study 1 VGG-style CNN for one configuration.

    config: configuration dict from decodeSolution.
    input_shape: shape of one input image.
    num_classes: number of output classes.
    """
    from keras import layers, models

    k       = config['kernel_size']
    pool_fn = layers.AveragePooling2D if config['pooling'] == 'average' else layers.MaxPooling2D
    base    = config['base_filters']

    n_blocks = config['n_blocks']
    filters  = [base * (2 ** i) for i in range(n_blocks)]

    model = models.Sequential(name='Template_CNN')
    model.add(layers.Input(shape=input_shape))

    for n_filters in filters:
        model.add(layers.Conv2D(n_filters, (k, k), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(_getActivationLayer(config['activation']))

        model.add(layers.Conv2D(n_filters, (k, k), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(_getActivationLayer(config['activation']))

        model.add(pool_fn((2, 2)))

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


def prepareDataFile(data, dir_path: str = '/tmp/hpo_data') -> dict:
    """Save the train and test arrays as .npy files for the child processes and
    return their paths.

    data: dataset dict from getDataset.
    dir_path: folder where the files are written.
    """
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


def _childFn(config, data_paths, eval_epochs, seed, queue):
    """Build and train one CNN in a child process and put ('ok', accuracy),
    ('oom', msg) or ('error', msg) on the queue.

    config: configuration dict from decodeSolution.
    data_paths: dict from prepareDataFile.
    eval_epochs: training epochs.
    seed: random seed of the evaluation.
    queue: multiprocessing queue that receives the result.
    """
    import sys as _sys
    _sys.stderr.flush()
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)

    os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
    os.environ['GRPC_VERBOSITY']         = 'ERROR'
    os.environ['GLOG_minloglevel']       = '2'
    os.environ['AUTOGRAPH_VERBOSITY']    = '0'

    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import logging
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    except Exception:
        pass

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    tf.random.set_seed(seed)
    tf.config.optimizer.set_jit(False)
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    try:
        x_train     = np.load(data_paths['x_train'])
        y_train     = np.load(data_paths['y_train'])
        x_test      = np.load(data_paths['x_test'])
        y_test      = np.load(data_paths['y_test'])
        input_shape = tuple(data_paths['input_shape'])
        num_classes = int(data_paths['num_classes'])

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
        queue.put(('oom', str(e)[:300]))

    except Exception as e:
        queue.put(('error', f"{type(e).__name__}: {str(e)[:300]}"))


def evalWorker(
    config: dict,
    data_paths: dict,
    eval_epochs: int = 1,
    timeout: int = 180,
    seed: int = 42,
    start_method: str = 'spawn',
) -> tuple:
    """Evaluate one configuration in a fresh subprocess and return its (status,
    value) pair.

    config: configuration dict from decodeSolution.
    data_paths: dict from prepareDataFile.
    eval_epochs: training epochs.
    timeout: seconds before the child process is killed.
    seed: random seed of the evaluation.
    start_method: multiprocessing start method.
    """
    ctx   = mp.get_context(start_method)
    queue = ctx.Queue()
    proc  = ctx.Process(
        target=_childFn,
        args=(config, data_paths, eval_epochs, seed, queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        _drainQueue(queue)
        return ('timeout', f'child exceeded {timeout}s')

    try:
        if not queue.empty():
            return queue.get_nowait()
        return ('error', f'child exited (code={proc.exitcode}) with no result')
    finally:
        _drainQueue(queue)


def _drainQueue(q) -> None:
    """Empty a queue so it closes without resource warnings.

    q: multiprocessing queue.
    """
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
