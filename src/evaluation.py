from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Calcula as métricas obrigatórias do projeto a partir das previsões de um modelo."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)

    return metrics


def evaluate_model(model, X, y, model_name="modelo"):
    """Avalia um modelo já treinado, imprimindo as métricas, a matriz de confusão
    e o relatório de classificação.

    Retorna um dicionário com as métricas (Acurácia, Precisão, Recall, F1-Score
    e AUC-ROC) para permitir a comparação padronizada entre os modelos da equipe.
    """
    y_pred = model.predict(X)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)

    metrics = compute_metrics(y, y_pred, y_proba)

    print(f"--- Métricas: {model_name} ---")
    for nome, valor in metrics.items():
        print(f"{nome:10s}: {valor:.4f}")

    print("\nMatriz de confusão:")
    print(confusion_matrix(y, y_pred))

    print("\nRelatório de classificação:")
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    metrics["modelo"] = model_name
    return metrics


def save_results(metrics, output_path="../results/resultados_modelos.csv"):
    """Adiciona (ou atualiza) a linha de métricas de um modelo em um CSV
    compartilhado, usado na Fase 3 para comparar o desempenho dos modelos.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nova_linha = pd.DataFrame([metrics])

    if output_path.exists():
        resultados = pd.read_csv(output_path)
        resultados = resultados[resultados["modelo"] != metrics["modelo"]]
        resultados = pd.concat([resultados, nova_linha], ignore_index=True)
    else:
        resultados = nova_linha

    colunas = ["modelo", "accuracy", "precision", "recall", "f1", "auc_roc"]
    resultados = resultados[colunas]
    resultados.to_csv(output_path, index=False)

    return resultados
