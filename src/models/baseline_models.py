"""Fábrica de modelos tabulares. Todos van en un Pipeline con StandardScaler
(se ajusta solo con el entrenamiento de cada fold)."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import load_yaml

MODEL_NAMES = ["dummy", "rf", "svm", "knn", "lda", "mlp"]


def load_models_cfg() -> dict:
    return load_yaml("experiment/models.yaml")


def build_model(name: str, models_cfg: dict | None = None, seed: int | None = None) -> Pipeline:
    cfg = models_cfg or load_models_cfg()
    seed = cfg.get("seed", 42) if seed is None else seed
    p = dict((cfg.get("models") or {}).get(name) or {})
    if name == "dummy":
        est = DummyClassifier(strategy=p.get("strategy", "stratified"), random_state=seed)
    elif name == "rf":
        est = RandomForestClassifier(random_state=seed, **p)
    elif name == "svm":
        est = SVC(**p)
    elif name == "knn":
        est = KNeighborsClassifier(**p)
    elif name == "lda":
        est = LinearDiscriminantAnalysis(**p)
    elif name == "mlp":
        p["hidden_layer_sizes"] = tuple(p.get("hidden_layer_sizes", (64, 32)))
        est = MLPClassifier(random_state=seed, **p)
    else:
        raise ValueError(f"Modelo desconocido '{name}'. Disponibles: {MODEL_NAMES + ['cnn']}")
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])