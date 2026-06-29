import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.fft import rfft, rfftfreq

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@dataclass
class Config:
    data_root:        str = "../Test1"
    output_path:      str = "../Test1"
    emg_channels:    list = field(default_factory=lambda: ["EMG1_Envelope_RMS", "EMG2_Envelope_RMS", "EMG3_Envelope_RMS", "EMG4_Envelope_RMS"])
    fs:             float = 1000.0
    cnn_fixed_length: int = 600
    wamp_threshold: float = 1e-6
    aug_jitter_std: float = 0.01
    aug_time_warp:   bool = True
    aug_n_copies:     int = 3
    cnn_epochs:       int = 80
    cnn_batch_size:   int = 8
    cnn_lr:         float = 1e-3
    seed:             int = 42

CFG = Config()
np.random.seed(CFG.seed)
tf.random.set_seed(CFG.seed)

EMOTIONS = ["Sonrisa", "Disgusto", "Sorprendido", "Triste"]
COLORS_EMO = ["#27ae60", "#e74c3c", "#2980b9", "#8e44ad"]
LE = LabelEncoder().fit(EMOTIONS)
N_CLASSES = len(EMOTIONS)

# Procesamiento de Señales e Interpolación 
def _resample_channels(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Remuestrea una señal multicanal a una longitud fija mediante interpolación."""
    if len(sig) == target_len: return sig
    x_orig, x_new = np.linspace(0, 1, len(sig)), np.linspace(0, 1, target_len)
    return np.stack([np.interp(x_new, x_orig, sig[:, ch]) for ch in range(sig.shape[1])], axis=1).astype(np.float32)

def _normalize_channels(sig: np.ndarray) -> np.ndarray:
    return ((sig - sig.mean(axis=0, keepdims=True)) / (sig.std(axis=0, keepdims=True) + 1e-9)).astype(np.float32)

def load_epochs(data_root: str) -> list[dict]:
    root, epochs = Path(data_root), []
    for emo in EMOTIONS:
        emo_dir = root / emo
        if not emo_dir.exists(): continue
        
        for csv_path in sorted(emo_dir.glob("bloque_*.csv")):
            df = pd.read_csv(csv_path)
            available = [c for c in CFG.emg_channels if c in df.columns]
            if not available: continue

            dt = np.diff(df["Epoch_Time" if "Epoch_Time" in df.columns else "Time_Seconds"].values)
            fs = float(1.0 / np.median(dt[dt > 0])) if len(dt) > 0 else CFG.fs
            block = int(df["Block"].iloc[0]) if "Block" in df.columns else -1

            epochs.append({
                "signal": df[available].values.astype(np.float32), "emotion": emo,
                "block": block, "fs": fs, "n_ch": len(available), "path": str(csv_path)
            })
    print(f"  Cargadas {len(epochs)} épocas de {len(EMOTIONS)} emociones")
    return epochs

# ── Extracción de Características (Pipeline A) ────────────────────────────────

def extract_features(epoch: dict) -> np.ndarray:
    """Extrae de forma compacta 13 features por canal (8 temporales + 5 espectrales)."""
    sig, fs = epoch["signal"].astype(np.float64), epoch["fs"]
    feats = []

    for ch_idx in range(sig.shape[1]):
        x = sig[:, ch_idx]
        dx = np.diff(x)
        
        # Extracción Temporal Eficiente
        feats.extend([
            np.sqrt(np.mean(x**2)),                          # RMS
            np.mean(np.abs(x)),                              # MAV
            np.var(x),                                       # Variance
            np.sum(x**2),                                    # Energy
            np.sum(np.abs(dx)),                              # Waveform Length
            np.sum(np.abs(dx) > CFG.wamp_threshold),          # WAMP
            np.sum(np.diff(np.sign(x - x.mean())) != 0),     # Zero Crossings
            np.sum((dx[:-1] * dx[1:]) < 0)                   # Slope Sign Changes
        ])

        # Extracción Frecuencial (FFT)
        yf = np.abs(rfft(x)) ** 2
        xf = rfftfreq(len(x), d=1.0/fs)
        total_p = yf.sum()

        mdf = xf[np.searchsorted(np.cumsum(yf), total_p / 2)] if total_p > 0 else 0.0
        mnf = float(np.sum(xf * yf) / total_p) if total_p > 0 else 0.0
        p_lo = float(yf[xf <= 50].sum())
        p_hi = float(yf[(xf > 50) & (xf <= 150)].sum())
        
        feats.extend([mdf, mnf, p_lo, p_hi, p_hi / (p_lo + 1e-12)])

    return np.array(feats, dtype=np.float32)

def build_feature_dataset(epochs: list[dict]):
    X = np.stack([extract_features(e) for e in epochs])
    y = LE.transform([e["emotion"] for e in epochs])
    blocks = np.array([e["block"] for e in epochs])
    return X, y, blocks

# ── Data Augmentation y Preparación CNN (Pipeline B) ──────────────────────────

def augment_epoch(sig: np.ndarray) -> np.ndarray:
    aug = sig.copy()
    # 1. Jitter (Ruido gaussiano)
    for ch in range(aug.shape[1]):
        aug[:, ch] += np.random.normal(0, CFG.aug_jitter_std * aug[:, ch].std(), len(aug))
    
    # 2. Time Warp (Estiramiento/Compresión temporal) corregido
    if CFG.aug_time_warp:
        factor = np.random.uniform(0.90, 1.10)
        n_orig = len(aug)
        n_warp = int(n_orig * factor)
        
        x_orig = np.linspace(0, 1, n_orig)
        x_warp = np.linspace(0, 1, n_warp)
        
        # x=x_warp (puntos nuevos), xp=x_orig (eje original), fp=aug[:, ch] (valores originales)
        aug = np.stack([
            np.interp(x_warp, x_orig, aug[:, ch]) 
            for ch in range(aug.shape[1])
        ], axis=1)
        
    return aug.astype(np.float32)

def build_cnn_dataset(epochs: list[dict], L: int):
    X_list = [_normalize_channels(_resample_channels(e["signal"], L)) for e in epochs]
    y = LE.transform([e["emotion"] for e in epochs])
    blocks = np.array([e["block"] for e in epochs])
    return np.stack(X_list), y, blocks

def augment_train_set(X_tr: np.ndarray, y_tr: np.ndarray, n_copies: int):
    L = X_tr.shape[1]
    Xaug, yaug = [X_tr], [y_tr]
    for _ in range(n_copies):
        Xb = np.stack([_normalize_channels(_resample_channels(augment_epoch(X_tr[i]), L)) for i in range(len(X_tr))])
        Xaug.append(Xb)
        yaug.append(y_tr)
    return np.concatenate(Xaug), np.concatenate(yaug)

# ── Modelos y Validación Cross-Validation (LOBO) ─────────────────────────────

def build_cnn(input_shape, n_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=7, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.2),
        
        layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.2),
        
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.GlobalAveragePooling1D(), layers.Dropout(0.3),
        
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(CFG.cnn_lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def lobo_classical(X, y, blocks, clf):
    unique_blocks = np.unique(blocks)
    y_true, y_pred, y_proba = [], [], []
    for test_blk in unique_blocks:
        tr, te = blocks != test_blk, blocks == test_blk
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]).fit(X[tr], y[tr])
        y_true.extend(y[te].tolist())
        y_pred.extend(pipe.predict(X[te]).tolist())
        y_proba.extend(pipe.predict_proba(X[te]).tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_proba)

def lobo_cnn(X, y, blocks, L, n_ch):
    unique_blocks = np.unique(blocks)
    y_true, y_pred, y_proba = [], [], []
    for test_blk in unique_blocks:
        tr, te = blocks != test_blk, blocks == test_blk
        X_tr_aug, y_tr_aug = augment_train_set(X[tr], y[tr], CFG.aug_n_copies)
        
        model = build_cnn((L, n_ch), N_CLASSES)
        model.fit(X_tr_aug, y_tr_aug, epochs=CFG.cnn_epochs, batch_size=CFG.cnn_batch_size, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="loss")])
        
        probs = model.predict(X[te], verbose=0)
        y_true.extend(y[te].tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_proba.extend(probs.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_proba)

def compute_metrics(name, y_true, y_pred, y_proba):
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    try:    auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except: auc = float("nan")
    
    report = classification_report(y_true, y_pred, target_names=LE.classes_, output_dict=True, zero_division=0)
    return {
        "model": name, "acc": round(accuracy_score(y_true, y_pred), 4), "auc": round(auc, 4),
        **{f"f1_{emo}": round(report[emo]["f1-score"], 3) for emo in LE.classes_},
        "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba
    }

# ── Visualizaciones y Reportes de Métricas ────────────────────────────────────

def plot_report(results: list[dict], output_path: str):
    n_models = len(results)
    fig = plt.figure(figsize=(6 * n_models, 16))
    gs = gridspec.GridSpec(3, n_models, figure=fig, hspace=0.4, wspace=0.3)

    for col, res in enumerate(results):
        # 1. Matrices de Confusión
        ax = fig.add_subplot(gs[0, col])
        cm = confusion_matrix(res["y_true"], res["y_pred"], normalize="true")
        sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap="Blues", xticklabels=LE.classes_, yticklabels=LE.classes_, cbar=False, linewidths=0.5)
        ax.set_title(f"{res['model']}\nAcc={res['acc']:.1%} AUC={res['auc']:.3f}", weight="bold", fontsize=10)
        
        # 2. Curvas ROC
        ax_roc = fig.add_subplot(gs[1, col])
        y_bin = label_binarize(res["y_true"], classes=list(range(N_CLASSES)))
        for idx, (emo, color) in enumerate(zip(LE.classes_, COLORS_EMO)):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, idx], res["y_proba"][:, idx])
                ax_roc.plot(fpr, tpr, color=color, lw=1.5, label=f"{emo[:3]} ({roc_auc_score(y_bin[:, idx], res['y_proba'][:, idx]):.2f})")
            except: pass
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax_roc.set_title(f"ROC — {res['model']}", weight="bold", fontsize=9)
        ax_roc.legend(fontsize=7, loc="lower right")
        ax_roc.grid(True, alpha=0.2)

    # 3. Tabla resumen comparativa
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    col_labels = ["Modelo", "Accuracy", "AUC"] + [f"F1 {e}" for e in LE.classes_]
    table_data = [[r["model"], f"{r['acc']:.1%}", f"{r['auc']:.3f}"] + [f"{r.get(f'f1_{e}', 0):.3f}" for e in LE.classes_] for r in results]
    
    tbl = ax_table.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50"); tbl[0, j].set_text_props(color="white", weight="bold")
    tbl[int(np.argmax([r["acc"] for r in results])) + 1, 0].set_facecolor("#d5f5e3") # Resaltar mejor

    fig.savefig(Path(output_path) / "emg_classifier_report.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_feature_importance(X, y, feature_names, output_path):
    rf = RandomForestClassifier(n_estimators=150, random_state=CFG.seed).fit(StandardScaler().fit_transform(X), y)
    idx = np.argsort(rf.feature_importances_)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(20), rf.feature_importances_[idx], color=[COLORS_EMO[i % 4] for i in range(20)], alpha=0.8)
    ax.set_xticks(range(20))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right", fontsize=8)
    ax.set_title("Top 20 Features — Random Forest", weight="bold", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(output_path) / "emg_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

# ── Exportar y Predicción Futura (Inferencia) ────────────────────────────────

def save_best_model(results, X, y, X_cnn, y_cnn, L, n_ch, output_path):
    root = Path(output_path) / "best_model"
    root.mkdir(exist_ok=True)

    # Entrenar y almacenar el mejor clásico con toda la muestra
    best_cls = max([r for r in results if r["model"] != "CNN 1D"], key=lambda r: r["acc"])
    clfs = {"SVM": SVC(kernel="rbf", C=10, probability=True, random_state=CFG.seed),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=CFG.seed),
            "KNN": KNeighborsClassifier(n_neighbors=3)}
    
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clfs[best_cls["model"]])]).fit(X, y)
    with open(root / f"{best_cls['model'].replace(' ','_')}_best.pkl", "wb") as f:
        pickle.dump({"pipeline": pipe, "label_encoder": LE, "config": CFG}, f)

    # Entrenar y almacenar CNN con toda la muestra integrada
    X_aug, y_aug = augment_train_set(X_cnn, y_cnn, CFG.aug_n_copies)
    cnn = build_cnn((L, n_ch), N_CLASSES)
    cnn.fit(X_aug, y_aug, epochs=CFG.cnn_epochs, batch_size=CFG.cnn_batch_size, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    cnn.save(root / "cnn1d_best.keras")

    pd.Series({"emg_channels": CFG.emg_channels, "cnn_fixed_length": L, "fs": CFG.fs, "emotions": list(LE.classes_)}).to_json(root / "inference_meta.json", indent=2)

def predict_from_csv(csv_path: str, model_dir: str = None) -> dict:
    if model_dir is None: model_dir = str(Path(CFG.output_path) / "best_model")
    df = pd.read_csv(csv_path)
    sig = df[[c for c in CFG.emg_channels if c in df.columns]].values.astype(np.float32)
    
    dt = np.diff(df["Epoch_Time" if "Epoch_Time" in df.columns else "Time_Seconds"].values)
    fs = float(1.0 / np.median(dt[dt > 0])) if len(dt) > 0 else CFG.fs
    epoch = {"signal": sig, "fs": fs}

    # Inferencia Clásica
    pkl_path = next(Path(model_dir).glob("*.pkl"))
    with open(pkl_path, "rb") as f: saved = pickle.load(f)
    cls_prob = saved["pipeline"].predict_proba(extract_features(epoch).reshape(1, -1))[0]

    # Inferencia CNN
    cnn = keras.models.load_model(Path(model_dir) / "cnn1d_best.keras")
    cnn_prob = cnn.predict(_normalize_channels(_resample_channels(sig, CFG.cnn_fixed_length))[np.newaxis], verbose=0)[0]
    ens_prob = (cls_prob + cnn_prob) / 2.0

    return {
        "classical_pred": LE.inverse_transform([np.argmax(cls_prob)])[0],
        "cnn_pred": LE.inverse_transform([np.argmax(cnn_prob)])[0],
        "ensemble_pred": LE.inverse_transform([np.argmax(ens_prob)])[0]
    }

# ── Pipeline Ejecutor Principal ───────────────────────────────────────────────

def run(cfg: Config = None):
    cfg = cfg or CFG
    print("=" * 64 + "\n  EMG Emotion Classifier — Pipeline Integrado\n" + "=" * 64)

    epochs = load_epochs(cfg.data_root)
    if len(epochs) < N_CLASSES: raise ValueError("Dataset insuficiente.")

    L, n_ch = cfg.cnn_fixed_length, epochs[0]["signal"].shape[1]
    X_feat, y, blocks = build_feature_dataset(epochs)
    X_cnn, y_cnn, _ = build_cnn_dataset(epochs, L)

    feat_suffix = ["RMS","MAV","Var","Energy","WL","WAMP","ZC","SSC","MDF","MNF","P_lo","P_hi","Ratio"]
    feat_names = [f"{ch.replace('_Envelope_RMS','')}_{s}" for ch in cfg.emg_channels for s in feat_suffix]

    # Ejecución Cross-Validation de Modelos
    clfs = {"SVM": SVC(kernel="rbf", C=10, probability=True, random_state=cfg.seed),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=cfg.seed),
            "KNN": KNeighborsClassifier(n_neighbors=3)}

    results = []
    for name, clf in clfs.items():
        yt, yp, yprob = lobo_classical(X_feat, y, blocks, clf)
        results.append(compute_metrics(name, yt, yp, yprob))
        print(f"  [{name:13}] Acc={results[-1]['acc']:.1%}  AUC={results[-1]['auc']:.3f}")

    yt, yp, yprob = lobo_cnn(X_cnn, y_cnn, blocks, L, n_ch)
    results.append(compute_metrics("CNN 1D", yt, yp, yprob))
    print(f"  ['CNN 1D':13] Acc={results[-1]['acc']:.1%}  AUC={results[-1]['auc']:.3f}")

    # Reportes y Persistencia
    plot_report(results, cfg.output_path)
    plot_feature_importance(X_feat, y, feat_names, cfg.output_path)
    
    df_m = pd.DataFrame([{k: v for k, v in r.items() if k not in ("y_true","y_pred","y_proba")} for r in results])
    df_m.to_csv(Path(cfg.output_path) / "emg_classifier_metrics.csv", index=False)
    
    save_best_model(results, X_feat, y, X_cnn, y_cnn, L, n_ch, cfg.output_path)
    print(f"\nPipeline Finalizado. Mejor modelo: {max(results, key=lambda r: r['acc'])['model']}\n" + "=" * 64)
    return results

if __name__ == "__main__":
    run()