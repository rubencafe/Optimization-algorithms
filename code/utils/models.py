from keras import layers, models
from keras.optimizers import SGD, AdamW, RMSprop


def _getActivationLayer(activation):
    """Return the Keras activation layer for a name.

    activation: activation name, e.g. 'relu' or 'leaky_relu'.
    """
    if activation == 'leaky_relu':
        return layers.LeakyReLU()
    return layers.Activation(activation)


def _getOptimizer(name, learning_rate):
    """Return the Keras optimizer for a name.

    name: 'adamw', 'sgd' (with Nesterov momentum) or 'rmsprop'.
    learning_rate: optimizer learning rate.
    """
    name   = name.lower()
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
    k       = config['kernel_size']
    pool_fn = layers.AveragePooling2D if config['pooling'] == 'average' else layers.MaxPooling2D
    base    = config['base_filters']

    n_blocks = config['n_blocks']
    filters = [base * (2 ** i) for i in range(n_blocks)]

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
