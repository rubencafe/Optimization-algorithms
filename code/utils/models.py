from keras import layers, models
from keras.optimizers import SGD, AdamW, RMSprop


def _getActivationLayer(activation):
    if activation == 'leaky_relu':
        return layers.LeakyReLU()
    return layers.Activation(activation)


def _getOptimizer(name, learning_rate):
    name   = name.lower()
    if name == 'sgd':
        return SGD(learning_rate=learning_rate, nesterov=True)
    if name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)    
    return AdamW(learning_rate=learning_rate)


def buildTemplateCNN(config, input_shape, num_classes):
    k       = config['kernel_size']
    pool_fn = layers.AveragePooling2D if config['pooling'] == 'average' else layers.MaxPooling2D
    base    = config['base_filters']

    # VGG doubling rule — design principle, NOT a search dimension.
    # With n_blocks=3 and base=64 -> [64, 128, 256].
    n_blocks = config['n_blocks']
    filters = [base * (2 ** i) for i in range(n_blocks)]

    model = models.Sequential(name='Template_CNN')
    model.add(layers.Input(shape=input_shape))

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