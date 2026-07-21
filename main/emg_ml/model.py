"""
Carga del dataset de features y train/test del modelo baseline (Random Forest) con validacion 
Leave-One-Subject-Out (LOSO): en cada fold se entrena con todos los sujetos menos uno, y se 
prueba con el sujeto excluido

Evita que el modelo "memorice" la fisiologia de una persona en vez de aprender el gesto en si, 
y escala automaticamente segun agregues mas sujetos
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut


NON_FEATURE_COLS = {"Subject", "Label", "Emotion", "Block", "Confidence", "Window_Start_s"}


def load_dataset(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols


def run_loso(df, feature_cols, label_col="Label", group_col="Subject",
             only_high_confidence=False, n_estimators=300, random_state=42):
    # Corre LOSO completo e imprime el reporte por fold + el agregado.
    # only_high_confidence=True entrena SOLO con muestras Confidence=="Alta"
    X = df[feature_cols].values
    y = df[label_col].values
    groups = df[group_col].values

    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    importances = np.zeros(len(feature_cols))
    n_folds = 0

    subjects = sorted(df[group_col].unique())
    if len(subjects) < 2:
        print(f"⚠ Solo hay {len(subjects)} sujeto(s) ({subjects}). "
              f"LOSO necesita al menos 2 para tener sentido — "
              f"agrega mas sujetos antes de confiar en estas metricas.")
        return

    for train_idx, test_idx in logo.split(X, y, groups):
        test_subject = groups[test_idx][0]

        train_mask = np.ones(len(train_idx), dtype=bool)
        if only_high_confidence:
            train_mask = df.iloc[train_idx]["Confidence"].values == "Alta"

        X_train = X[train_idx][train_mask]
        y_train = y[train_idx][train_mask]
        X_test = X[test_idx]
        y_test = y[test_idx]

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                      class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"\n{'='*60}")
        print(f"Fold: test = {test_subject}  (train: {[s for s in subjects if s != test_subject]})")
        print(f"  train windows: {len(X_train)}  |  test windows: {len(X_test)}")
        print(classification_report(y_test, y_pred, zero_division=0))

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        importances += clf.feature_importances_
        n_folds += 1

    print(f"\n{'='*60}")
    print("REPORTE AGREGADO (todos los folds LOSO juntos)")
    print(f"{'='*60}")
    print(classification_report(all_true, all_pred, zero_division=0))

    labels_sorted = sorted(set(all_true) | set(all_pred))
    cm = confusion_matrix(all_true, all_pred, labels=labels_sorted)
    print("Matriz de confusion (filas=real, columnas=prediccion):")
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_string())

    importances /= n_folds
    imp_df = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 15 features mas importantes (promedio entre folds):")
    print(imp_df.head(15).to_string())

    return imp_df