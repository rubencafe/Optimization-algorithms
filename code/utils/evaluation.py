"""
evaluation.py
-------------
Avaliacao de modelos: metricas, graficos e persistencia de resultados.

Funcoes principais:
    evaluate_model()          — calcula todas as metricas (accuracy, precision, recall, F1,
                                TP/FP/FN/TN por classe, matriz de confusao)
    plot_training_history()   — curvas de accuracy e loss (treino vs validacao)
    plot_confusion_matrix()   — heatmap da matriz de confusao
    plot_per_class_metrics()  — grafico de barras precision/recall/F1 por classe
    print_metrics_table()     — tabela formatada no terminal
    plot_model_comparison()   — comparacao de multiplos modelos num grafico de barras
    save_results()            — guarda resultados em JSON
    load_results()            — carrega resultados de JSON

Uso:
    from evaluation import evaluate_model, plot_confusion_matrix, plot_training_history
    metrics = evaluate_model(model, x_test, y_test_cat, y_test_raw, class_names)
    plot_training_history(results['history'])
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Avaliacao principal
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, x_test, y_test_cat, y_test_raw, class_names):
    """
    Avalia um modelo treinado e calcula todas as metricas de classificacao.

    Parametros:
        model        : Modelo Keras treinado.
        x_test       : Imagens de teste (array normalizado).
        y_test_cat   : Labels de teste em one-hot.
        y_test_raw   : Labels de teste em inteiros (para sklearn).
        class_names  : Lista de nomes das classes.

    Retorna:
        dict com:
            accuracy, loss
            precision_macro, recall_macro, f1_macro
            per_class  — {classe: {precision, recall, f1-score, support}}
            tp_fp_fn_tn — {classe: {TP, FP, FN, TN}}
            confusion_matrix — lista 2D
            y_pred     — predicoes inteiras
    """
    # Avaliacao Keras (loss + accuracy)
    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

    # Predicoes
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    # Metricas macro (sklearn)
    precision_macro = precision_score(y_test_raw, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(   y_test_raw, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(       y_test_raw, y_pred, average='macro', zero_division=0)

    # Metricas por classe
    report    = classification_report(
        y_test_raw, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    per_class = {cls: report[cls] for cls in class_names}

    # Matriz de confusao
    cm = confusion_matrix(y_test_raw, y_pred)

    # TP, FP, FN, TN por classe (derivados da matriz de confusao)
    tp_fp_fn_tn = {}
    for i, cls in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        tp_fp_fn_tn[cls] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

    return {
        'accuracy':         float(accuracy),
        'loss':             float(loss),
        'precision_macro':  float(precision_macro),
        'recall_macro':     float(recall_macro),
        'f1_macro':         float(f1_macro),
        'per_class':        per_class,
        'tp_fp_fn_tn':      tp_fp_fn_tn,
        'confusion_matrix': cm.tolist(),
        'y_pred':           y_pred.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Impressao formatada
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics, model_name='Modelo'):
    """
    Imprime uma tabela formatada com todas as metricas no terminal.

    Parametros:
        metrics    : dict retornado por evaluate_model().
        model_name : Nome a mostrar no cabecalho.
    """
    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  RESULTADOS — {model_name}")
    print(sep)
    print(f"  {'Accuracy':<28}  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  {'Loss':<28}  {metrics['loss']:.4f}")
    print(f"  {'Precision (macro avg)':<28}  {metrics['precision_macro']:.4f}")
    print(f"  {'Recall    (macro avg)':<28}  {metrics['recall_macro']:.4f}")
    print(f"  {'F1-Score  (macro avg)':<28}  {metrics['f1_macro']:.4f}")
    print(sep)
    print(f"\n  {'Classe':<13} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>7}  "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}")
    print(f"  {'-'*65}")
    for cls in metrics['tp_fp_fn_tn']:
        t   = metrics['tp_fp_fn_tn'][cls]
        pc  = metrics['per_class'][cls]
        print(
            f"  {cls:<13} {t['TP']:>6} {t['FP']:>6} {t['FN']:>6} {t['TN']:>7}  "
            f"{pc['precision']:>6.3f} {pc['recall']:>6.3f} {pc['f1-score']:>6.3f} "
            f"{int(pc['support']):>6}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Graficos
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, title='Historico de Treino', save_path=None):
    """
    Plota as curvas de accuracy e loss (treino vs validacao).

    Parametros:
        history   : dict com keys 'accuracy','val_accuracy','loss','val_loss'
                    (como em results['history']).
        title     : Titulo do grafico.
        save_path : Caminho para guardar a figura (opcional).
    """
    epochs = range(1, len(history['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Accuracy ---
    ax1.plot(epochs, history['accuracy'],     'b-o', label='Treino',    markersize=4)
    ax1.plot(epochs, history['val_accuracy'], 'r-o', label='Validacao', markersize=4)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Loss ---
    ax2.plot(epochs, history['loss'],     'b-o', label='Treino',    markersize=4)
    ax2.plot(epochs, history['val_loss'], 'r-o', label='Validacao', markersize=4)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Grafico guardado: {save_path}")
    plt.show()


def plot_confusion_matrix(cm, class_names,
                           title='Matriz de Confusao',
                           normalize=False,
                           save_path=None):
    """
    Plota a matriz de confusao como heatmap.

    Parametros:
        cm          : Matriz de confusao (lista 2D ou numpy array).
        class_names : Nomes das classes.
        normalize   : Se True, normaliza por linha (mostra recall por classe).
        save_path   : Caminho para guardar (opcional).
    """
    cm = np.array(cm)

    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt     = '.2f'
        title  += ' (Normalizada)'
    else:
        cm_plot = cm
        fmt     = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor='lightgray',
        cbar_kws={'shrink': 0.8}
    )
    ax.set_ylabel('Classe Real', fontsize=12)
    ax.set_xlabel('Classe Prevista', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0,  fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Grafico guardado: {save_path}")
    plt.show()


def plot_per_class_metrics(per_class, class_names,
                            title='Metricas por Classe',
                            save_path=None):
    """
    Grafico de barras com Precision, Recall e F1-Score por classe.

    Parametros:
        per_class   : dict retornado em metrics['per_class'].
        class_names : Lista de nomes das classes.
        title       : Titulo do grafico.
        save_path   : Caminho para guardar (opcional).
    """
    precision = [per_class[c]['precision'] for c in class_names]
    recall    = [per_class[c]['recall']    for c in class_names]
    f1        = [per_class[c]['f1-score']  for c in class_names]

    x     = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_p = ax.bar(x - width, precision, width, label='Precision', color='steelblue',  alpha=0.85)
    bars_r = ax.bar(x,         recall,    width, label='Recall',    color='darkorange', alpha=0.85)
    bars_f = ax.bar(x + width, f1,        width, label='F1-Score',  color='seagreen',   alpha=0.85)

    # Anotacoes de valor em cima de cada barra
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7.5
            )

    ax.set_xlabel('Classe', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
    ax.set_ylim([0, 1.12])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Grafico guardado: {save_path}")
    plt.show()


def plot_model_comparison(results_list, metric='accuracy',
                           title=None, save_path=None):
    """
    Grafico de barras comparando multiplos experimentos num dado metrica.

    Parametros:
        results_list : Lista de dicts retornados por run_experiment().
        metric       : Metrica a comparar:
                       'accuracy' | 'f1_macro' | 'precision_macro' | 'recall_macro'
        title        : Titulo (opcional; gerado automaticamente se None).
        save_path    : Caminho para guardar (opcional).
    """
    names  = [r['model_name'] for r in results_list]
    values = [r['metrics'][metric] for r in results_list]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

    if title is None:
        title = f'Comparacao de Modelos — {metric.replace("_", " ").title()}'

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 5))
    bars = ax.bar(names, values, color=colors, alpha=0.88,
                  edgecolor='black', linewidth=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10
        )

    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.12])
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Grafico guardado: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia em json
# ─────────────────────────────────────────────────────────────────────────────

def save_results_json(results, results_dir='results'):
    """
    Guarda os resultados de um experimento num ficheiro JSON.

    Parametros:
        results     : dict retornado por run_experiment().
        results_dir : Directoria onde guardar. Default: 'results/'

    Retorna:
        str: caminho do ficheiro criado.
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = os.path.join(results_dir, f"{results['model_name']}_{timestamp}.json")

    # Remove objectos nao serializaveis (modelo Keras)
    serializable = {k: v for k, v in results.items() if k not in ('model',)}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"  Resultados guardados: {filename}")
    return filename


def load_results_json(filepath):
    """
    Carrega resultados de experimento de um ficheiro JSON.

    Parametros:
        filepath : Caminho para o ficheiro JSON.

    Retorna:
        dict com os resultados do experimento.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
# ─────────────────────────────────────────────────────────────────────────────
# Persistencia em CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(results, filepath='results/metrics.csv'):
    """
    Guarda as metricas globais de um experimento num CSV.
    Se o ficheiro ja existir, acrescenta uma nova linha (modo append).
    Desta forma todos os experimentos ficam acumulados no mesmo ficheiro,
    facilitando comparacoes e analise posterior.

    Colunas geradas:
        timestamp, model_name, dataset,
        epochs_trained, batch_size, dropout_rate, learning_rate,
        train_time_s,
        accuracy, loss, precision_macro, recall_macro, f1_macro,
        precision_<classe>, recall_<classe>, f1_<classe>  (uma coluna por classe)

    Parametros:
        results  : dict retornado por run_experiment().
        filepath : Caminho do CSV. Default: 'results/metrics.csv'

    Retorna:
        pd.DataFrame com todos os experimentos acumulados.

    Exemplo:
        save_metrics_csv(results)
        save_metrics_csv(results, filepath='results/all_experiments.csv')
    """
    m   = results['metrics']
    cfg = results['config']

    # Linha base com metricas globais
    row = {
        'timestamp':       results['timestamp'],
        'model_name':      results['model_name'],
        'dataset':         results['dataset'],
        'epochs_trained':  cfg['epochs_trained'],
        'batch_size':      cfg['batch_size'],
        'dropout_rate':    cfg['dropout_rate'],
        'learning_rate':   cfg['learning_rate'],
        'train_time_s':    results['train_time_seconds'],
        'accuracy':        m['accuracy'],
        'loss':            m['loss'],
        'precision_macro': m['precision_macro'],
        'recall_macro':    m['recall_macro'],
        'f1_macro':        m['f1_macro'],
    }

    # Metricas por classe como colunas adicionais
    for cls, vals in m['per_class'].items():
        safe = cls.replace(' ', '_')
        row[f'precision_{safe}'] = vals['precision']
        row[f'recall_{safe}']    = vals['recall']
        row[f'f1_{safe}']        = vals['f1-score']

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    df_new = pd.DataFrame([row])

    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(filepath, index=False)
    n = len(df)
    print(f"  Metricas guardadas: {filepath}  ({n} experimento{'s' if n != 1 else ''} no total)")
    return df


def save_history_csv(results, filepath=None):
    """
    Guarda o historico de treino epoch-a-epoch num CSV.
    Cada linha corresponde a uma epoch e inclui o nome do modelo
    e o timestamp para identificar a que experimento pertence.

    Colunas geradas:
        model_name, timestamp, epoch,
        accuracy, val_accuracy, loss, val_loss

    Parametros:
        results  : dict retornado por run_experiment().
        filepath : Caminho do CSV. Se None, gera automaticamente em
                   results/history_<model>_<timestamp>.csv

    Retorna:
        pd.DataFrame com o historico.

    Exemplo:
        save_history_csv(results)
        save_history_csv(results, filepath='results/history_CNN_M.csv')
    """
    if filepath is None:
        ts       = results['timestamp'].replace(':', '-').replace('.', '-')
        name     = results['model_name']
        filepath = f"results/history_{name}_{ts}.csv"

    history = results['history']
    n_epochs = len(history['accuracy'])

    rows = []
    for i in range(n_epochs):
        row = {
            'model_name':   results['model_name'],
            'timestamp':    results['timestamp'],
            'epoch':        i + 1,
            'accuracy':     history['accuracy'][i],
            'val_accuracy': history['val_accuracy'][i],
            'loss':         history['loss'][i],
            'val_loss':     history['val_loss'][i],
        }
        # Incluir lr se estiver disponivel (ReduceLROnPlateau guarda-o)
        if 'lr' in history:
            row['lr'] = history['lr'][i]
        rows.append(row)

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Historico guardado : {filepath}  ({n_epochs} epochs)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Carregar CSVs e re-gerar graficos
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics_csv(filepath='results/metrics.csv'):
    """
    Carrega o CSV de metricas acumuladas.

    Retorna:
        pd.DataFrame com todos os experimentos.

    Exemplo:
        df = load_metrics_csv()
        print(df[['model_name', 'accuracy', 'f1_macro']])
    """
    df = pd.read_csv(filepath)
    print(f"  {len(df)} experimento(s) carregado(s) de '{filepath}'")
    return df


def load_history_csv(filepath):
    """
    Carrega o CSV de historico de treino de um experimento.

    Retorna:
        pd.DataFrame com colunas epoch, accuracy, val_accuracy, loss, val_loss.

    Exemplo:
        df = load_history_csv('results/history_CNN_M_2026-03-18.csv')
    """
    df = pd.read_csv(filepath)
    print(f"  Historico carregado: '{filepath}'  ({len(df)} epochs)")
    return df


def plot_from_csv(model_name, metrics_csv='results/metrics.csv',
                  history_csv=None, history_dir='results',
                  class_names=None):
    """
    Carrega os CSVs guardados e re-gera os mesmos graficos do run_experiment().

    Graficos gerados:
        1. Curvas de treino (accuracy + loss) — requer history_csv
        2. Metricas por classe (precision/recall/F1) — requer metrics_csv
        3. Comparacao geral de todos os modelos no CSV — requer metrics_csv

    Nota: a matriz de confusao nao esta no CSV — usa load_results() + o JSON
          para a obter (ver exemplo abaixo).

    Parametros:
        model_name   : Nome do modelo a visualizar (e.g. 'CNN_M').
        metrics_csv  : Caminho para o CSV de metricas. Default: 'results/metrics.csv'
        history_csv  : Caminho para o CSV de historico deste modelo.
                       Se None, tenta encontrar automaticamente em history_dir.
        history_dir  : Pasta onde procurar ficheiros history_*.csv. Default: 'results'
        class_names  : Lista de nomes de classes. Se None, tenta inferir do CSV.

    Exemplo:
        plot_from_csv('CNN_M')
        plot_from_csv('CNN_L_BN', history_csv='results/history_CNN_L_BN_2026-03-18.csv')
    """
    import glob

    # ── 1. Carregar metrics CSV ───────────────────────────────────────────────
    df = pd.read_csv(metrics_csv)
    row = df[df['model_name'] == model_name]

    if row.empty:
        available = df['model_name'].tolist()
        raise ValueError(
            f"Modelo '{model_name}' nao encontrado em '{metrics_csv}'.\n"
            f"Disponiveis: {available}"
        )
    row = row.iloc[-1]   # ultima execucao deste modelo

    # ── 2. Inferir class_names do CSV se nao fornecidas ───────────────────────
    if class_names is None:
        # As colunas de precisao seguem o padrao precision_<classe>
        class_names = [
            c.replace('precision_', '')
            for c in df.columns
            if c.startswith('precision_') and c != 'precision_macro'
        ]

    # ── 3. Reconstruir per_class dict ─────────────────────────────────────────
    per_class = {}
    for cls in class_names:
        safe = cls.replace(' ', '_')
        per_class[cls] = {
            'precision': row.get(f'precision_{safe}', 0.0),
            'recall':    row.get(f'recall_{safe}',    0.0),
            'f1-score':  row.get(f'f1_{safe}',        0.0),
            'support':   0,   # nao guardado no CSV
        }

    # ── 4. Reconstruir metricas globais ───────────────────────────────────────
    metrics = {
        'accuracy':        row['accuracy'],
        'loss':            row['loss'],
        'precision_macro': row['precision_macro'],
        'recall_macro':    row['recall_macro'],
        'f1_macro':        row['f1_macro'],
        'per_class':       per_class,
    }

    # ── 5. Imprimir tabela de metricas ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {model_name}  |  {row['dataset'].upper()}")
    print(f"  Epochs: {int(row['epochs_trained'])}  |  "
          f"Batch: {int(row['batch_size'])}  |  "
          f"LR: {row['learning_rate']}  |  "
          f"Dropout: {row['dropout_rate']}")
    print(f"{'='*55}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision macro : {metrics['precision_macro']:.4f}")
    print(f"  Recall macro    : {metrics['recall_macro']:.4f}")
    print(f"  F1 macro        : {metrics['f1_macro']:.4f}")
    print(f"{'='*55}\n")

    # ── 6. Grafico: metricas por classe ───────────────────────────────────────
    if class_names:
        plot_per_class_metrics(
            per_class, class_names,
            title=f'Metricas por Classe — {model_name}'
        )

    # ── 7. Carregar history CSV e plotar curvas de treino ─────────────────────
    if history_csv is None:
        # Tenta encontrar automaticamente: history_<MODEL>_*.csv
        pattern = os.path.join(history_dir, f'history_{model_name}_*.csv')
        matches = sorted(glob.glob(pattern))
        if matches:
            history_csv = matches[-1]   # o mais recente
            print(f"  History CSV encontrado: {history_csv}")

    if history_csv and os.path.exists(history_csv):
        df_hist = pd.read_csv(history_csv)
        history_dict = {
            'accuracy':     df_hist['accuracy'].tolist(),
            'val_accuracy': df_hist['val_accuracy'].tolist(),
            'loss':         df_hist['loss'].tolist(),
            'val_loss':     df_hist['val_loss'].tolist(),
        }
        plot_training_history(
            history_dict,
            title=f'Historico de Treino — {model_name}'
        )
    else:
        print(f"  (Sem history CSV para '{model_name}' — grafico de treino omitido)")

    return metrics


def plot_all_from_csv(metrics_csv='results/metrics.csv',
                      metrics=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro')):
    """
    Carrega o CSV de metricas e gera graficos de comparacao para todas as metricas.

    Parametros:
        metrics_csv : Caminho para o CSV. Default: 'results/metrics.csv'
        metrics     : Tuplo de metricas a comparar.

    Exemplo:
        plot_all_from_csv()
        plot_all_from_csv(metrics=('accuracy', 'f1_macro'))
    """
    df = pd.read_csv(metrics_csv)
    print(f"\n  {len(df)} experimento(s) em '{metrics_csv}':")
    print(df[['model_name', 'accuracy', 'f1_macro', 'epochs_trained', 'dropout_rate']].to_string(index=False))
    print()

    # Usar a ultima execucao de cada modelo (caso haja repetidos)
    df_last = df.groupby('model_name', sort=False).last().reset_index()

    # Reconstruir formato esperado por plot_model_comparison
    results_list = []
    for _, row in df_last.iterrows():
        results_list.append({
            'model_name': row['model_name'],
            'metrics': {
                'accuracy':        row['accuracy'],
                'f1_macro':        row['f1_macro'],
                'precision_macro': row['precision_macro'],
                'recall_macro':    row['recall_macro'],
            }
        })

    for metric in metrics:
        plot_model_comparison(
            results_list,
            metric=metric,
            title=f'Comparacao de Modelos — {metric.replace("_", " ").title()}'
        )
        
# ─────────────────────────────────────────────────────────────────────────────
# Carregar todos os dados de um modelo
# ─────────────────────────────────────────────────────────────────────────────
def load_model_data(model_name, results_dir='results'):
    """
    Carrega todos os dados guardados de um modelo a partir dos ficheiros
    CSV e JSON existentes na pasta de resultados.

    Ficheiros lidos (quando existem):
        metrics.csv                      — metricas globais e por classe
        history_<MODEL>_*.csv            — historico de treino por epoch
        <MODEL>_*.json                   — dados completos (inclui matriz de confusao)

    Parametros:
        model_name  (str) : Nome do modelo, e.g. 'CNN_M'.
        results_dir (str) : Pasta onde estao os ficheiros. Default: 'results'

    Retorna:
        dict com:
            'model_name'   — nome do modelo
            'config'       — configuracao do experimento (do JSON ou CSV)
            'metrics'      — dict com accuracy, loss, precision_macro,
                             recall_macro, f1_macro, per_class, confusion_matrix
                             (confusion_matrix so disponivel se JSON existir)
            'history'      — dict com accuracy, val_accuracy, loss, val_loss por epoch
                             (None se nao houver history CSV)
            'df_metrics'   — linha do pd.DataFrame do metrics.csv (ou None)
            'df_history'   — pd.DataFrame do history CSV (ou None)
            'json_path'    — caminho do JSON usado (ou None)
            'history_path' — caminho do history CSV usado (ou None)

    Exemplo:
        data = load_model_data('CNN_M')

        print(data['metrics']['accuracy'])
        print(data['metrics']['confusion_matrix'])  # requer JSON
        plot_training_history(data['history'])
        plot_confusion_matrix(data['metrics']['confusion_matrix'], data['class_names'])
    """
    import glob

    result = {
        'model_name':   model_name,
        'config':       {},
        'metrics':      {},
        'history':      None,
        'class_names':  None,
        'df_metrics':   None,
        'df_history':   None,
        'json_path':    None,
        'history_path': None,
    }

    # ── 1. Tentar carregar JSON (fonte mais completa) ──────────────────────────
    json_pattern = os.path.join(results_dir, f'{model_name}_*.json')
    json_files   = sorted(glob.glob(json_pattern))

    if json_files:
        json_path = json_files[-1]   # o mais recente
        with open(json_path, 'r', encoding='utf-8') as f:
            full = json.load(f)

        result['json_path']   = json_path
        result['config']      = full.get('config', {})
        result['metrics']     = full.get('metrics', {})
        result['class_names'] = full.get('class_names')

        # history do JSON (fallback se nao houver CSV)
        if 'history' in full:
            result['history'] = full['history']

        print(f"  JSON carregado      : {json_path}")
    else:
        print(f"  (Sem JSON para '{model_name}' em '{results_dir}')")

    # ── 2. Carregar metrics.csv ────────────────────────────────────────────────
    metrics_csv = os.path.join(results_dir, 'metrics.csv')
    if os.path.exists(metrics_csv):
        df_all = pd.read_csv(metrics_csv)
        rows   = df_all[df_all['model_name'] == model_name]

        if not rows.empty:
            row = rows.iloc[-1]   # ultima execucao
            result['df_metrics'] = row

            # Preencher config se nao veio do JSON
            if not result['config']:
                result['config'] = {
                    'epochs_trained':  int(row.get('epochs_trained', 0)),
                    'batch_size':      int(row.get('batch_size', 0)),
                    'dropout_rate':    float(row.get('dropout_rate', 0.0)),
                    'learning_rate':   float(row.get('learning_rate', 0.0)),
                }

            # Preencher metricas globais se nao vieram do JSON
            if not result['metrics']:
                # Inferir class_names das colunas precision_<classe>
                class_names = [
                    c.replace('precision_', '')
                    for c in df_all.columns
                    if c.startswith('precision_') and c != 'precision_macro'
                ]
                result['class_names'] = result['class_names'] or class_names

                per_class = {}
                for cls in class_names:
                    safe = cls.replace(' ', '_')
                    per_class[cls] = {
                        'precision': float(row.get(f'precision_{safe}', 0.0)),
                        'recall':    float(row.get(f'recall_{safe}',    0.0)),
                        'f1-score':  float(row.get(f'f1_{safe}',        0.0)),
                        'support':   0,
                    }

                result['metrics'] = {
                    'accuracy':        float(row['accuracy']),
                    'loss':            float(row['loss']),
                    'precision_macro': float(row['precision_macro']),
                    'recall_macro':    float(row['recall_macro']),
                    'f1_macro':        float(row['f1_macro']),
                    'per_class':       per_class,
                }

            print(f"  metrics.csv lido    : {metrics_csv}")
        else:
            print(f"  (Modelo '{model_name}' nao encontrado em metrics.csv)")
    else:
        print(f"  (metrics.csv nao encontrado em '{results_dir}')")

    # ── 3. Carregar history CSV (sobrepoe history do JSON se existir) ──────────
    hist_pattern = os.path.join(results_dir, f'history_{model_name}_*.csv')
    hist_files   = sorted(glob.glob(hist_pattern))

    if hist_files:
        hist_path = hist_files[-1]   # o mais recente
        df_hist   = pd.read_csv(hist_path)

        result['df_history']   = df_hist
        result['history_path'] = hist_path
        result['history'] = {
            'accuracy':     df_hist['accuracy'].tolist(),
            'val_accuracy': df_hist['val_accuracy'].tolist(),
            'loss':         df_hist['loss'].tolist(),
            'val_loss':     df_hist['val_loss'].tolist(),
        }
        if 'lr' in df_hist.columns:
            result['history']['lr'] = df_hist['lr'].tolist()

        print(f"  History CSV carregado: {hist_path}")
    else:
        print(f"  (Sem history CSV para '{model_name}' em '{results_dir}')")

    # ── 4. Resumo ──────────────────────────────────────────────────────────────
    m = result['metrics']
    if m:
        print(f"\n  {'─'*45}")
        print(f"  Modelo      : {model_name}")
        print(f"  Accuracy    : {m.get('accuracy', 'n/a'):.4f}")
        print(f"  F1 macro    : {m.get('f1_macro', 'n/a'):.4f}")
        print(f"  Precision   : {m.get('precision_macro', 'n/a'):.4f}")
        print(f"  Recall      : {m.get('recall_macro', 'n/a'):.4f}")
        has_cm = bool(m.get('confusion_matrix'))
        has_h  = result['history'] is not None
        print(f"  Conf. matrix: {'sim' if has_cm else 'nao (sem JSON)'}")
        print(f"  History     : {'sim (' + str(len(result['history']['accuracy'])) + ' epochs)' if has_h else 'nao'}")
        print(f"  {'─'*45}\n")

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Compare multiple algorithms
# ─────────────────────────────────────────────────────────────────────────────

def compare_algorithms(
    algorithms,
    data,
    eval_epochs = 5,
    n_generations      = 15,
    pop_size    = 20,
    results_dir = 'results/optimization',
):
    """
    Runs multiple algorithms in sequence and compares results.

    Parameters:
        algorithms  (list) : List of names, e.g. ['PSO', 'GA', 'DE'].
        data        (dict) : Dict returned by get_dataset().
        eval_epochs (int)  : Epochs per evaluation. Default: 5.
        n_generations      (int)  : Iterations per algorithm. Default: 15.
        pop_size    (int)  : Population per algorithm. Default: 20.
        results_dir (str)  : Results folder. Default: 'results/optimization'.

    Returns:
        list of dicts, sorted by val_accuracy descending.

    Example:
        results = compare_algorithms(['PSO', 'GA', 'DE'], data)
    """
    all_results = []
    for alg in algorithms:
        best_config, run_results = runOptimization(
            alg, data,
            eval_epochs=eval_epochs,
            n_generations=n_generations,
            pop_size=pop_size,
            results_dir=results_dir,
        )
        all_results.append(run_results)

    # Sort by val_accuracy
    all_results.sort(key=lambda r: r['best']['val_accuracy'], reverse=True)

    # Final summary
    print(f"\n{'='*62}")
    print(f"  ALGORITHM COMPARISON")
    print(f"{'='*62}")
    print(f"  {'Algorithm':<10}  {'Best val_acc':>12}  {'Time (min)':>10}  {'Evaluations':>11}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['algorithm']:<10}  "
              f"{r['best']['val_accuracy']:>12.4f}  "
              f"{r['total_time_s']/60:>10.1f}  "
              f"{r['n_evaluations']:>11}")
    print()

    # Comparison plot
    plot_algorithm_comparison(all_results, results_dir)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Train final model with best configuration
# ─────────────────────────────────────────────────────────────────────────────

def train_best(
    best_config,
    data,
    epochs       = 50,
    early_stopping_patience = 5,
    model_name   = 'OPTIMIZED_CNN',
    results_dir  = 'results/optimization',
):
    import time
    import numpy as np
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from code.utils.models import buildTemplateCNN

    print(f"\n{'='*62}")
    print(f"  FINAL TRAINING with optimized configuration")
    print(f"{'='*62}")
    for k, v in best_config.items():
        print(f"  {k:<16} {v}")
    print()

    model = buildTemplateCNN(best_config, data['input_shape'], data['num_classes'])
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
        validation_data=(data['x_test'], data['y_test']),
        epochs=epochs,
        batch_size=best_config['batch_size'],
        callbacks=callbacks,
        verbose=1,
    )
    t_total = time.time() - t_start

    # Derive integer labels for sklearn metrics (y_test is one-hot).
    y_test_raw = np.argmax(data['y_test'], axis=1)
    metrics = evaluate_model(
        model, data['x_test'], data['y_test'],
        y_test_raw, data['class_names']
    )

    results = {
        'model_name':         model_name,
        'dataset':            'cifar10',
        'timestamp':          datetime.now().isoformat(),
        'config': {
            **best_config,
            'epochs_requested': epochs,
            'epochs_trained':   len(history.history['accuracy']),
        },
        'train_time_seconds': round(t_total, 2),
        'history':            {k: [float(v) for v in vals]
                               for k, vals in history.history.items()},
        'metrics':            metrics,
        'class_names':        data['class_names'],
        'model':              model,
    }

    print_metrics_table(metrics, model_name)
    print(f"  Train time       : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print(f"  Epochs trained   : {results['config']['epochs_trained']} / {epochs}")
    return results


def train_best_from_json(
    json_file,
    data,
    epochs                  = 50,
    early_stopping_patience = 5,
    results_dir             = 'results/optimization',
):
    if os.path.isfile(json_file):
        json_path = json_file
    else:
        candidate = os.path.join(results_dir, json_file)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(
                f"JSON file not found.\n"
                f"  Tried as path     : {json_file}\n"
                f"  Tried in dir      : {candidate}"
            )
        json_path = candidate

    print(f"\n  File loaded        : {json_path}")

    with open(json_path, 'r') as f:
        run_results = json.load(f)

    best_config  = run_results['best']['config']
    best_val_acc = run_results['best']['val_accuracy']
    alg_name     = run_results.get('algorithm', 'UNKNOWN').upper()

    print(f"  Algorithm          : {alg_name}")
    print(f"  Best val_acc (opt) : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")

    return train_best(
        best_config,
        data,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        model_name=f'OPTIMIZED_{alg_name}',
        results_dir=results_dir,
    )


def train_best_from_config(
    best_config,
    data,
    epochs                  = 50,
    early_stopping_patience = 5,
    algorithm               = None,
    results_dir             = 'results/optimization',
):
    """
    Same as ``train_best_from_json`` but receives the config dict directly
    instead of a JSON path. Useful when you already have the configuration
    in memory (e.g. straight out of ``runOptimization``) or when you want
    to retrain an arbitrary hand-picked config.

    The expected ``best_config`` structure is the same as
    ``run_results['best']['config']`` produced by
    ``utils.optimizer.runOptimization`` — i.e. a dict with the keys
    ``learning_rate``, ``dropout_rate``, ``activation``, ``use_bn``,
    ``batch_size``, ``base_filters``, ``n_blocks``, ``dense_units``,
    ``optimizer``, ``pooling``, ``kernel_size``. The ``use_bn`` key is
    accepted for backward compatibility but ignored — batch normalization
    is now always applied (see FIXED_BN in optimizer.py).

    Parameters:
        best_config             (dict) : Config dict (see structure above).
        data                    (dict) : Dict returned by getDataset().
        epochs                  (int)  : Full training epochs. Default: 50.
        early_stopping_patience (int)  : EarlyStopping patience. Default: 5.
        algorithm               (str)  : Optional tag used in the saved
                                         model name (e.g. 'PSO', 'DE').
                                         If None, 'MANUAL' is used.
        results_dir             (str)  : Results folder.

    Returns:
        results dict compatible with run_experiment() format.

    Example:
        data = getDataset('cifar10')
        best = {
            'learning_rate': 1e-3, 'dropout_rate': 0.5, 'activation': 'relu',
            'batch_size': 32,  'base_filters': 32, 'n_blocks': 2,
            'dense_units': 512,
            'optimizer': 'adamw', 'pooling': 'max', 'kernel_size': 3,
        }
        results = train_best_from_config(best, data, epochs=80, algorithm='PSO')
    """
    tag = (algorithm or 'MANUAL').upper()

    print(f"\n  Source             : in-memory config")
    print(f"  Algorithm tag      : {tag}")

    return train_best(
        best_config,
        data,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        model_name=f'OPTIMIZED_{tag}',
        results_dir=results_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optimization plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_optimization_history(run_results, save_path=None):
    """
    Plots the evolution of best val_accuracy over evaluations.

    Parameters:
        run_results : dict returned by makeObjective().
        save_path   : Path to save the figure (optional).
    """
    log    = pd.DataFrame(run_results['eval_log'])
    alg    = run_results['algorithm']
    best_so_far = log['val_accuracy'].cummax()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Optimization History — {alg}', fontsize=13, fontweight='bold')

    # Val accuracy per evaluation + running best
    ax1.scatter(log['eval'], log['val_accuracy'],
                alpha=0.4, s=20, color='steelblue', label='Each evaluation')
    ax1.plot(log['eval'], best_so_far,
             color='red', linewidth=2, label='Running best')
    ax1.set_xlabel('Evaluation')
    ax1.set_ylabel('Val Accuracy')
    ax1.set_title('Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution of explored accuracies
    ax2.hist(log['val_accuracy'], bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    ax2.axvline(log['val_accuracy'].max(), color='red', linestyle='--',
                label=f"Best: {log['val_accuracy'].max():.4f}")
    ax2.set_xlabel('Val Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Evaluations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_algorithm_comparison(all_results, results_dir='results/optimization'):
    """
    Bar chart comparing algorithms, and overlaid convergence curves.

    Parameters:
        all_results : list of dicts returned by compare_algorithms().
        results_dir : Folder to save the figure (optional).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optimization Algorithm Comparison', fontsize=13, fontweight='bold')

    names    = [r['algorithm']              for r in all_results]
    best_acc = [r['best']['val_accuracy']   for r in all_results]
    colors   = plt.cm.tab10(np.linspace(0, 0.7, len(names)))

    # Bars — best result per algorithm
    bars = ax1.bar(names, best_acc, color=colors, alpha=0.88,
                   edgecolor='black', linewidth=0.6)
    for bar, val in zip(bars, best_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Best Val Accuracy')
    ax1.set_title('Best Result per Algorithm')
    ax1.set_ylim([0, 1.08])
    ax1.grid(True, axis='y', alpha=0.3)

    # Overlaid convergence curves
    for r, color in zip(all_results, colors):
        log         = pd.DataFrame(r['eval_log'])
        best_so_far = log['val_accuracy'].cummax()
        ax2.plot(log['eval'], best_so_far, color=color, linewidth=1.5, label=r['algorithm'])
    ax2.set_xlabel('Evaluation')
    ax2.set_ylabel('Val Accuracy')
    ax2.set_title('Convergence Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(results_dir, f'comparison_{ts}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Comparison plot   : {save_path}")


def plot_optimization_history(run_results, save_path=None):
    """
    Plots the evolution of best val_accuracy over evaluations.

    Parameters:
        run_results : dict returned by makeObjective().
        save_path   : Path to save the figure (optional).
    """
    log    = pd.DataFrame(run_results['eval_log'])
    alg    = run_results['algorithm']
    best_so_far = log['val_accuracy'].cummax()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Optimization History — {alg}', fontsize=13, fontweight='bold')

    # Val accuracy per evaluation + running best
    ax1.scatter(log['eval'], log['val_accuracy'],
                alpha=0.4, s=20, color='steelblue', label='Each evaluation')
    ax1.plot(log['eval'], best_so_far,
             color='red', linewidth=2, label='Running best')
    ax1.set_xlabel('Evaluation')
    ax1.set_ylabel('Val Accuracy')
    ax1.set_title('Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution of explored accuracies
    ax2.hist(log['val_accuracy'], bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    ax2.axvline(log['val_accuracy'].max(), color='red', linestyle='--',
                label=f"Best: {log['val_accuracy'].max():.4f}")
    ax2.set_xlabel('Val Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Evaluations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_algorithm_comparison(all_results, results_dir='results/optimization'):
    """
    Bar chart comparing algorithms, and overlaid convergence curves.

    Parameters:
        all_results : list of dicts returned by compare_algorithms().
        results_dir : Folder to save the figure (optional).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optimization Algorithm Comparison', fontsize=13, fontweight='bold')

    names    = [r['algorithm']              for r in all_results]
    best_acc = [r['best']['val_accuracy']   for r in all_results]
    colors   = plt.cm.tab10(np.linspace(0, 0.7, len(names)))

    # Bars — best result per algorithm
    bars = ax1.bar(names, best_acc, color=colors, alpha=0.88,
                   edgecolor='black', linewidth=0.6)
    for bar, val in zip(bars, best_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Best Val Accuracy')
    ax1.set_title('Best Result per Algorithm')
    ax1.set_ylim([0, 1.08])
    ax1.grid(True, axis='y', alpha=0.3)

    # Overlaid convergence curves
    for r, color in zip(all_results, colors):
        log         = pd.DataFrame(r['eval_log'])
        best_so_far = log['val_accuracy'].cummax()
        ax2.plot(log['eval'], best_so_far, color=color, linewidth=1.5, label=r['algorithm'])
    ax2.set_xlabel('Evaluation')
    ax2.set_ylabel('Val Accuracy')
    ax2.set_title('Convergence Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(results_dir, f'comparison_{ts}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Comparison plot   : {save_path}")