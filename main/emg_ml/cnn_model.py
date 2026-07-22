"""
CNN 1D para clasificacion de ventanas EMG crudas (tiempo x canal),
evaluada con LOSO

La entrada viene de raw_dataset.py, que arma las ventanas a partir de las
columnas EMG*_Filtered para que el CNN aprenda de la forma de onda real
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras import layers, models


def build_cnn(input_shape, n_classes, random_state=42):
    tf.random.set_seed(random_state)
    model = models.Sequential([
        layers.Input(shape=input_shape),                    # (n_samples_fixed, n_channels)
        layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _scale_per_channel(X_train, X_test):
    # Normaliza cada canal por separado; media/std calculadas SOLO con train
    n_channels = X_train.shape[2]
    X_train_s = np.zeros_like(X_train, dtype=np.float32)
    X_test_s = np.zeros_like(X_test, dtype=np.float32)
    for c in range(n_channels):
        mean = X_train[:, :, c].mean()
        std = X_train[:, :, c].std() + 1e-8
        X_train_s[:, :, c] = (X_train[:, :, c] - mean) / std
        X_test_s[:, :, c] = (X_test[:, :, c] - mean) / std
    return X_train_s, X_test_s


def run_loso_cnn(raw_data, only_high_confidence=False, epochs=40, batch_size=32,
                  random_state=42, return_results=False):
    # Entrena un CNN 1D nuevo en cada fold LOSO (misma logica que run_loso, pero 
    # adaptada a input 3D y a Keras en vez de sklearn)

    X, y_raw = raw_data["X"], raw_data["y"]
    groups, confidence = raw_data["groups"], raw_data["confidence"]

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)
    class_names = label_enc.classes_

    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    fold_results = []

    subjects = sorted(set(groups))
    if len(subjects) < 2:
        print(f"⚠ Solo hay {len(subjects)} sujeto(s). LOSO necesita al menos 2.")
        return None

    for train_idx, test_idx in logo.split(X, y, groups):
        test_subject = groups[test_idx][0]

        train_mask = np.ones(len(train_idx), dtype=bool)
        if only_high_confidence:
            train_mask = confidence[train_idx] == "Alta"

        X_train, y_train = X[train_idx][train_mask], y[train_idx][train_mask]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train_s, X_test_s = _scale_per_channel(X_train, X_test)

        model = build_cnn(input_shape=X_train.shape[1:], n_classes=len(class_names),
                           random_state=random_state)
        model.fit(X_train_s, y_train, epochs=epochs, batch_size=batch_size,
                  verbose=0, validation_split=0.1)

        y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
        fold_acc = accuracy_score(y_test, y_pred)

        print(f"Fold: test = {test_subject}")
        print(f"  train windows: {len(X_train)}  |  test windows: {len(X_test)}")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        fold_results.append({"test_subject": test_subject, "n_train": len(X_train),
                              "n_test": len(X_test), "accuracy": fold_acc})

    print("REPORTE AGREGADO (CNN, todos los folds LOSO juntos)")
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

    labels_sorted = sorted(set(all_true) | set(all_pred))
    label_names_sorted = [class_names[i] for i in labels_sorted]
    cm = confusion_matrix(all_true, all_pred, labels=labels_sorted)
    print("Matriz de confusion (filas=real, columnas=prediccion):")
    print(pd.DataFrame(cm, index=label_names_sorted, columns=label_names_sorted).to_string())

    if not return_results:
        return None

    report_dict = classification_report(all_true, all_pred, target_names=class_names,
                                         zero_division=0, output_dict=True)
    return {
        "cm": cm,
        "labels": label_names_sorted,
        "report_dict": report_dict,
        "fold_results": fold_results,
        "all_true": all_true,
        "all_pred": all_pred,
    }