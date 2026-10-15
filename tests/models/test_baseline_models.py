import pytest
from src.dataset import feature_columns
from src.models import MODEL_NAMES, build_model
from tests.toy import make_toy_df, small_cfg


def test_all_models_fit_and_predict():
    df = make_toy_df()
    X, y = df[feature_columns(df)].to_numpy(), df["label_id"].to_numpy()
    for name in MODEL_NAMES:
        m = build_model(name, small_cfg())
        m.fit(X, y)
        assert m.predict(X).shape == y.shape


def test_unknown_model():
    with pytest.raises(ValueError, match="desconocido"):
        build_model("xyz", small_cfg())