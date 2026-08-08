import numpy as np
import keras
import tensorflow as tf
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
    """
    Carrega e pre-processa o dataset CIFAR-10.

    Parametros:
        val_split (float): Fracao do treino usada para validacao. Default: 0.2
        seed (int)       : Seed para reprodutibilidade. Default: 42

    Retorna:
        dict com:
            x_train, y_train         — imagens normalizadas + labels one-hot (treino)
            x_val,   y_val           — subconjunto de validacao (split de x_train)
            x_test,  y_test          — dados de teste (intocados)
            class_names              — lista dos 10 nomes de classes
            num_classes              — 10
            input_shape              — (32, 32, 3)
    """
    num_classes = 10

    # 50 000 imagens de treino + 10 000 de teste (split oficial)
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalizar pixeis para [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test       = x_test.astype('float32')       / 255.0

    y_train_cat = keras.utils.to_categorical(y_train_full, num_classes)
    y_test_cat  = keras.utils.to_categorical(y_test,       num_classes)

    # Split treino -> treino + validacao
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
        'input_shape': x_train.shape[1:],   # (32, 32, 3)
    }


def loadMnist(val_split=0.2, seed=42):
    """
    Carrega e pre-processa o dataset MNIST.

    Parametros:
        val_split (float): Fracao do treino usada para validacao. Default: 0.2
        seed (int)       : Seed para reprodutibilidade. Default: 42

    Retorna:
        dict com:
            x_train, y_train         — imagens normalizadas + labels one-hot (treino)
            x_val,   y_val           — subconjunto de validacao (split de x_train)
            x_test,  y_test          — dados de teste (intocados)
            class_names              — lista dos 10 nomes de classes (digitos 0-9)
            num_classes              — 10
            input_shape              — (28, 28, 1)
    """
    num_classes = 10

    # 60 000 imagens de treino + 10 000 de teste (split oficial)
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalizar pixeis para [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test       = x_test.astype('float32')       / 255.0

    # Adicionar dimensao do canal: (N, 28, 28) -> (N, 28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test       = np.expand_dims(x_test,       axis=-1)

    y_train_cat = keras.utils.to_categorical(y_train_full, num_classes)
    y_test_cat  = keras.utils.to_categorical(y_test,       num_classes)

    # Split treino -> treino + validacao
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
        'input_shape': x_train.shape[1:],   # (28, 28, 1)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Factory — adiciona novos datasets aqui no futuro
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_REGISTRY = {
    'cifar10': loadCifar10,
    'mnist':   loadMnist,
    # 'cifar100': load_cifar100,   # futuro
    # 'imagenet': load_imagenet,   # futuro
}


def getDataset(name='cifar10', **kwargs):
    """
    Carrega um dataset pelo nome.

    Parametros:
        name (str) : Nome do dataset. Disponiveis: ['cifar10', 'mnist']
        **kwargs   : Argumentos passados ao loader (val_split, seed, ...)

    Retorna:
        dict com os dados prontos para treino/teste.

    Exemplo:
        data = getDataset('cifar10', val_split=0.2, seed=42)
        data['x_train'].shape   # (40000, 32, 32, 3)
    """
    name = name.lower()
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not available. "
            f"Availables: {list(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[name](**kwargs)


def listDatasets():
    """Retorna lista dos datasets disponiveis."""
    return list(_DATASET_REGISTRY.keys())
