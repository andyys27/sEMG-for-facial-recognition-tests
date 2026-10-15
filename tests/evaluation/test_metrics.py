import numpy as np
import pandas as pd
import pytest
from src.evaluation import classification_metrics, segment_vote, summarize, class_info


def test_class_info_order():
    ids, names = class_info()
    assert names == ["Reposo", "Sonrisa", "Triste", "Disgusto", "Sorprendido"] and ids == [0, 1, 2, 3, 4]


def test_classification_metrics_known_values():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 1], ["a", "b"])
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_segment_vote_majority():
    pred = pd.DataFrame({"subject": "s", "session": "x", "segment_id": [0, 0, 0, 1, 1],
                         "y_true": [1, 1, 1, 0, 0], "y_pred": [1, 1, 0, 0, 0]})
    v = segment_vote(pred)
    assert v["y_pred"].tolist() == [1, 0] and v["y_true"].tolist() == [1, 0]


def test_summarize_structure():
    pred = pd.DataFrame({"fold": ["a"] * 4 + ["b"] * 4, "subject": "s", "session": "x",
                         "segment_id": [0, 0, 1, 1, 2, 2, 3, 3], "y_true": [0, 1] * 4,
                         "y_pred": [0, 1, 0, 1, 0, 0, 0, 1]})
    s = summarize(pred, [0, 1], ["a", "b"])
    assert s["n_folds"] == 2 and set(s["per_fold"]) == {"a", "b"}
    assert s["fold_stats"]["accuracy"]["mean"] == pytest.approx((1.0 + 0.75) / 2)
    assert s["segment_vote"]["n"] == 4