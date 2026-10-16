"""CNN 1D sobre la forma de onda cruda (Keras). TensorFlow es opcional: se importa al usarla."""
import numpy as np
import pandas as pd

from .common import prediction_frame, undersample_indices


def _keras():
    try:
        import keras
    except ImportError as e:
        raise ImportError("La CNN necesita Keras/TensorFlow: pip install -r requirements-cnn.txt") from e
    return keras


def build_cnn(input_shape: tuple, n_classes: int, cfg: dict):
    keras = _keras()
    L = keras.layers
    x = inp = keras.Input(shape=input_shape)
    for f in cfg.get("filters", [32, 64]):
        x = L.Conv1D(f, 7, padding="same")(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)
        x = L.MaxPooling1D(2)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(cfg.get("dropout", 0.3))(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg.get("lr", 1e-3)),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _split_validation(df: pd.DataFrame, tr: np.ndarray, val_frac: float, seed: int):
    """Separa validación por SEGMENTOS completos (nunca ventanas sueltas, que se solapan)."""
    sub = df.iloc[tr]
    seg = (sub["subject"].astype(str) + "|" + sub["session"].astype(str) + "|"
           + sub["segment_id"].astype(str)).to_numpy()
    segs = pd.DataFrame({"seg": seg, "label": sub["label_id"].to_numpy()}).drop_duplicates("seg")
    rng = np.random.default_rng(seed)
    val = set()
    for _, g in segs.groupby("label"):
        if len(g) > 1:
            val.update(rng.choice(g["seg"].to_numpy(), max(1, int(round(val_frac * len(g)))), replace=False))
    is_val = np.isin(seg, list(val))
    return tr[~is_val], tr[is_val]


def cross_validate_cnn(df: pd.DataFrame, windows: np.ndarray, folds, cfg: dict, seed: int) -> pd.DataFrame:
    """Misma salida que train.cross_validate (una fila por ventana de test)."""
    keras = _keras()
    ccfg = cfg.get("cnn", {})
    ids = sorted(df["label_id"].unique())
    y = df["label_id"].map({c: i for i, c in enumerate(ids)}).to_numpy()
    parts = []
    for f in folds:
        tr = f.train
        if cfg.get("balance_train", True):
            tr = tr[undersample_indices(y[tr], seed)]
        fit, val = _split_validation(df, tr, ccfg.get("val_frac", 0.2), seed)
        if ccfg.get("normalize", "arcsinh") == "arcsinh":
            # arcsinh(x / s): lineal cerca de 0 y logarítmico para amplitudes grandes, de modo que las
            # clases de poca amplitud (reposo) no queden aplastadas por los gestos fuertes. `s` es el
            # nivel de ruido típico (mediana de |x|) del entrenamiento del fold.
            s = np.median(np.abs(windows[fit]), axis=(0, 1)) + 1e-12
            norm = lambda a: np.arcsinh(a / s).astype("float32")
        else:                                                   # estandarización clásica por canal
            mu, sd = windows[fit].mean(axis=(0, 1)), windows[fit].std(axis=(0, 1)) + 1e-12
            norm = lambda a: ((a - mu) / sd).astype("float32")

        keras.utils.set_random_seed(seed)
        model = build_cnn(windows.shape[1:], len(ids), ccfg)
        monitor = "val_loss" if len(val) else "loss"
        model.fit(norm(windows[fit]), y[fit],
                  validation_data=(norm(windows[val]), y[val]) if len(val) else None,
                  epochs=ccfg.get("epochs", 40), batch_size=ccfg.get("batch_size", 64), verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor=monitor, patience=ccfg.get("patience", 8),
                                                           restore_best_weights=True)])
        pred = model.predict(norm(windows[f.test]), verbose=0).argmax(axis=1)
        parts.append(prediction_frame(df, f.test, np.asarray(ids)[pred], f.name))
    return pd.concat(parts, ignore_index=True)