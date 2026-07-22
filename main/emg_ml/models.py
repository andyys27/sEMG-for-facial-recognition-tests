"""
Fabrica de modelos para el pipeline LOSO tabular. 
Cada modelo se envuelve en un Pipeline con StandardScaler: Random Forest 
no lo necesita, pero SVM, KNN, LDA y MLP si
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_NAMES = {
    "rf": "RandomForest",
    "svm": "SVM",
    "knn": "KNN",
    "lda": "LDA",
    "mlp": "MLP",
}


def build_model(name, random_state=42):
    name = name.lower()

    if name in ("rf", "random_forest"):
        clf = RandomForestClassifier(n_estimators=300, random_state=random_state,
                                      class_weight="balanced")
    elif name == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                   random_state=random_state)
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    elif name == "lda":
        clf = LinearDiscriminantAnalysis()
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             alpha=1e-3, max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {list(MODEL_NAMES)}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])