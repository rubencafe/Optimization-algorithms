import numpy as np
import keras
from sklearn.model_selection import train_test_split

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

MNIST_CLASSES = [
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9'
]


def loadCifar10(val_split=0.2, seed=42):
    """Load CIFAR-10 as a dict of train, validation and test arrays (pixels in
    [0, 1], one-hot labels).

    val_split: fraction of the official training set kept for validation.
    seed: random seed of the train/validation split.
    """
    num_classes = 10

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train_full = x_train_full.astype('float32') / 255.0
    x_test       = x_test.astype('float32')       / 255.0

    y_train_cat = keras.utils.to_categorical(y_train_full, num_classes)
    y_test_cat  = keras.utils.to_categorical(y_test,       num_classes)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_cat,
        test_size=val_split, random_state=seed, shuffle=True
    )

    print(f"  CIFAR-10 loaded:")
    print(f"    Train      : {x_train.shape[0]:>6} samples")
    print(f"    Val        : {x_val.shape[0]:>6} samples  (val_split={val_split})")
    print(f"    Test       : {x_test.shape[0]:>6} samples")
    print(f"    Shape      : {x_train.shape[1:]}")
    print(f"    Classes    : {CIFAR10_CLASSES}")

    return {
        'x_train':     x_train,
        'y_train':     y_train,
        'x_val':       x_val,
        'y_val':       y_val,
        'x_test':      x_test,
        'y_test':      y_test_cat,
        'class_names': CIFAR10_CLASSES,
        'num_classes': num_classes,
        'input_shape': x_train.shape[1:],
    }


def loadMnist(val_split=0.2, seed=42):
    """Load MNIST as a dict of train, validation and test arrays (pixels in [0,
    1], one-hot labels).

    val_split: fraction of the official training set kept for validation.
    seed: random seed of the train/validation split.
    """
    num_classes = 10

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_full = x_train_full.astype('float32') / 255.0
    x_test       = x_test.astype('float32')       / 255.0

    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test       = np.expand_dims(x_test,       axis=-1)

    y_train_cat = keras.utils.to_categorical(y_train_full, num_classes)
    y_test_cat  = keras.utils.to_categorical(y_test,       num_classes)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_cat,
        test_size=val_split, random_state=seed, shuffle=True
    )

    print(f"  MNIST loaded:")
    print(f"    Train      : {x_train.shape[0]:>6} samples")
    print(f"    Val        : {x_val.shape[0]:>6} samples  (val_split={val_split})")
    print(f"    Test       : {x_test.shape[0]:>6} samples")
    print(f"    Shape      : {x_train.shape[1:]}")
    print(f"    Classes    : {MNIST_CLASSES}")

    return {
        'x_train':     x_train,
        'y_train':     y_train,
        'x_val':       x_val,
        'y_val':       y_val,
        'x_test':      x_test,
        'y_test':      y_test_cat,
        'class_names': MNIST_CLASSES,
        'num_classes': num_classes,
        'input_shape': x_train.shape[1:],
    }


_DATASET_REGISTRY = {
    'cifar10': loadCifar10,
    'mnist':   loadMnist,
}


def getDataset(name='cifar10', **kwargs):
    """Load a dataset by name.

    name: 'cifar10' or 'mnist'.
    kwargs: arguments passed to the loader (val_split, seed).
    """
    name = name.lower()
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not available. "
            f"Availables: {list(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[name](**kwargs)
