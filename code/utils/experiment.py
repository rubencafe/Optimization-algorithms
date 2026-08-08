"""
experiment.py
-------------
Runner de experimentos: carrega dados, treina modelo, avalia e guarda resultados.

Funcao principal:
    run_experiment() — executa o pipeline completo num unico comando.

Uso rapido:
    from experiment import run_experiment
    results = run_experiment('CNN_M', epochs=20)

Uso com configuracao completa:
    results = run_experiment(
        model_name          = 'CNN_L_BN',
        dataset_name        = 'cifar10',
        epochs              = 30,
        batch_size          = 64,
        dropout_rate        = 0.3,
        learning_rate       = 1e-3,
        early_stopping      = True,
        patience            = 5,
        save                = True,
    )

Comparar varios modelos:
    from experiment import run_multiple
    results = run_multiple(['CNN_S', 'CNN_M', 'CNN_L'], epochs=20)
    # -> gera automaticamente grafico de comparacao
"""

import time
from datetime import datetime

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # reduz logs TF

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from code.utils.datasets   import getDataset
from code.utils.models     import getModel, listModels, buildDynamicCNN
from code.utils.evaluation import (
    evaluate_model, print_metrics_table,
    save_results_json, save_metrics_csv, save_history_csv,
    plot_training_history, plot_confusion_matrix,
    plot_per_class_metrics, plot_model_comparison
)


# ─────────────────────────────────────────────────────────────────────────────
# Experimento unico
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_name,
    dataset_name        = 'cifar10',
    epochs              = 20,
    batch_size          = 64,
    dropout_rate        = 0.0,
    learning_rate       = 1e-3,
    val_split           = 0.2,
    seed                = 42,
    early_stopping      = True,
    patience            = 5,
    save                = True,
    results_dir         = 'results',
    augment             = False,
    show_plots          = True,
    verbose             = 1,
):
    bar = '=' * 62
    print(f"\n{bar}")
    print(f"  EXPERIMENTO: {model_name.upper()}  |  Dataset: {dataset_name.upper()}")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")
    print(f"  Dropout: {dropout_rate}  |  Early Stopping: {early_stopping}  |  Augmentation: {augment}")
    print(bar)

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    print("\n  [1/4] A carregar dataset...")
    data = getDataset(dataset_name, val_split=val_split, seed=seed)

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    print(f"\n  [2/4] A construir modelo {model_name.upper()}...")
    model = getmodel(
        model_name,
        input_shape   = data['input_shape'],
        num_classes   = data['num_classes'],
        dropout_rate  = dropout_rate,
        learning_rate = learning_rate,
    )
    if verbose > 0:
        model.summary()


    train_data = data['x_train']
    val_data   = (data['x_val'], data['y_val'])

    # ── 4. Treino ─────────────────────────────────────────────────────────────
    print(f"\n  [3/4] A treinar...\n")

    callbacks = []
    if early_stopping:
        # will stop as soon as val_accuracy stops improving for 'patience' epochs, and will restore the best weights found during training.
        callbacks.append(EarlyStopping(
            monitor            = 'val_accuracy',
            patience           = patience,
            restore_best_weights = True,
            verbose            = 1,
        ))
        # will reduce the learning rate by a factor of 0.5 if val_loss does not improve for 'patience//2' epochs, down to a minimum of 1e-6.
        callbacks.append(ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = max(2, patience // 2),
            min_lr   = 1e-6,
            verbose  = 1,
        ))

    start_time = time.time()
    fit_kwargs = dict(
        validation_data = val_data,
        epochs          = epochs,
        callbacks       = callbacks,
        verbose         = verbose,
    )
    # batch_size so e necessario quando nao se usa tf.data (augment=False)
    if not augment:
        fit_kwargs['batch_size'] = batch_size
        history = model.fit(data['x_train'], data['y_train'], **fit_kwargs)
    else:
        history = model.fit(train_data, **fit_kwargs)
    train_time = time.time() - start_time

    # ── 4. Avaliacao ──────────────────────────────────────────────────────────
    print(f"\n  [4/4] A avaliar no conjunto de teste...")
    metrics = evaluate_model(
        model,
        data['x_test'], data['y_test'],
        data['y_test_raw'], data['class_names']
    )

    # ── Resultados ────────────────────────────────────────────────────────────
    epochs_trained = len(history.history['accuracy'])
    results = {
        'model_name':          model_name.upper(),
        'dataset':             dataset_name,
        'timestamp':           datetime.now().isoformat(),
        'config': {
            'epochs_requested': epochs,
            'epochs_trained':   epochs_trained,
            'batch_size':       batch_size,
            'dropout_rate':     dropout_rate,
            'learning_rate':    learning_rate,
            'val_split':        val_split,
            'seed':             seed,
            'early_stopping':   early_stopping,
            'patience':         patience,
            'augment':          augment,
        },
        'train_time_seconds':  round(train_time, 2),
        'history':             {k: [float(v) for v in vals]
                                for k, vals in history.history.items()},
        'metrics':             metrics,
        'class_names':         data['class_names'],
        'model':               model,   # nao serializado, apenas referencia
    }

    # Tabela de metricas no terminal
    print_metrics_table(metrics, model_name.upper())
    print(f"  Tempo de treino : {train_time:.1f}s")
    print(f"  Epochs treinados: {epochs_trained} / {epochs}")
    if early_stopping and epochs_trained < epochs:
        print(f"  (Early stopping ativado na epoch {epochs_trained})")

    # Graficos
    if show_plots:
        plot_training_history(
            results['history'],
            title=f'Historico de Treino — {model_name.upper()}'
        )
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            data['class_names'],
            title=f'Matriz de Confusao — {model_name.upper()}'
        )
        plot_per_class_metrics(
            metrics['per_class'],
            data['class_names'],
            title=f'Metricas por Classe — {model_name.upper()}'
        )

    # Guardar resultados
    if save:
        save_results_json(results, results_dir)                    # JSON completo
        save_metrics_csv(results,                                  # CSV metricas (append)
                         filepath=os.path.join(results_dir, 'metrics.csv'))
        save_history_csv(results)                                  # CSV historico por epoch

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multiplos experimentos
# ─────────────────────────────────────────────────────────────────────────────

def run_multiple(
    model_names,
    compare_metric = 'accuracy',
    show_comparison = True,
    save_comparison = True,
    results_dir    = 'results',
    **kwargs
):
    """
    Corre varios experimentos em sequencia e compara os resultados.

    Parametros:
        model_names      (list) : Lista de nomes de modelos, e.g. ['CNN_S','CNN_M','CNN_L'].
        compare_metric   (str)  : Metrica de comparacao final. Default: 'accuracy'.
        show_comparison  (bool) : Mostrar grafico de comparacao no final. Default: True.
        save_comparison  (bool) : Guardar grafico de comparacao. Default: True.
        results_dir      (str)  : Directoria de resultados. Default: 'results'.
        **kwargs                : Argumentos passados a run_experiment() (epochs, batch_size, ...).

    Retorna:
        list de dicts de resultados, ordenados por compare_metric (decrescente).
    """
    all_results = []
    for name in model_names:
        r = run_experiment(name, results_dir=results_dir, **kwargs)
        all_results.append(r)

    # Ordenar por metrica decrescente
    all_results.sort(key=lambda x: x['metrics'][compare_metric], reverse=True)

    # Sumario final
    print(f"\n{'='*62}")
    print(f"  SUMARIO FINAL — {compare_metric.upper()}")
    print(f"{'='*62}")
    for i, r in enumerate(all_results):
        val = r['metrics'][compare_metric]
        print(f"  {i+1}. {r['model_name']:<12}  {val:.4f}  ({val*100:.2f}%)")
    print()

    # Grafico de comparacao
    if show_comparison:
        save_path = None
        if save_comparison:
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, f'comparison_{compare_metric}.png')
        plot_model_comparison(
            all_results,
            metric    = compare_metric,
            save_path = save_path
        )

    return all_results


def train_best(
    best_config,
    data,
    epochs       = 50,
    early_stopping_patience = 5,
    results_dir  = 'results/optimization',
):
    """
    Trains the final model with the best configuration found by optimization,
    this time with the full number of epochs.

    Parameters:
        best_config : dict returned by makeObjective() as best_config.
        data        : dict returned by get_dataset().
        epochs      : Full training epochs. Default: 50.
        early_stopping_patience : EarlyStopping patience. Default: 5.
        results_dir : Results folder.

    Returns:
        results dict compatible with run_experiment() format
        (can be used with print_metrics_table, plot_confusion_matrix, etc.)

    Example:
        best, history = makeObjective('PSO', data)
        results = train_best(best, data, epochs=50)
    """
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from code.utils.evaluation import evaluate_model, print_metrics_table

    print(f"\n{'='*62}")
    print(f"  FINAL TRAINING with optimized configuration")
    print(f"{'='*62}")
    for k, v in best_config.items():
        print(f"  {k:<16} {v}")
    print()

    model = buildDynamicCNN(best_config, data['input_shape'], data['num_classes'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=early_stopping_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-6, verbose=1),
    ]

    t_start = time.time()
    history = model.fit(
        data['x_train'], data['y_train'],
        validation_data=(data['x_val'], data['y_val']),
        epochs=epochs,
        batch_size=best_config['batch_size'],
        callbacks=callbacks,
        verbose=1,
    )
    t_total = time.time() - t_start

    metrics = evaluate_model(
        model, data['x_test'], data['y_test'],
        data['y_test_raw'], data['class_names']
    )

    results = {
        'model_name':         'OPTIMIZED_CNN',
        'dataset':            'cifar10',
        'timestamp':          datetime.now().isoformat(),
        'config':             best_config,
        'train_time_seconds': round(t_total, 2),
        'history':            {k: [float(v) for v in vals]
                               for k, vals in history.history.items()},
        'metrics':            metrics,
        'class_names':        data['class_names'],
        'model':              model,
    }

    print_metrics_table(metrics, 'OPTIMIZED_CNN')
    return results