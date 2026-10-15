import numpy as np
import pytest
from src.features import extract_features
from src.dataset import build_dataset, get_splits, feature_columns, export_dataset, load_dataset

CH = ["EMG1", "EMG2", "EMG3", "EMG4"]
FS = 1000


def test_features_known_values():
    t = np.arange(200) / FS
    W = np.tile(np.sin(2 * np.pi * 100 * t)[None, :, None] * 1e-4, (3, 1, 4))   # 100 Hz, 100 uV
    W[:, :, 2:] *= 0.1                                                             # grupo G34 10x menor
    df = extract_features(W, FS, CH)
    assert len(df) == 3
    assert np.allclose(df["EMG1_RMS"], 1e-4 / np.sqrt(2), rtol=0.02)
    assert np.allclose(df["EMG1_MAV"], 1e-4 * 2 / np.pi, rtol=0.05)   # discreto: ~3 % bajo el valor continuo
    assert 38 <= df["EMG1_ZC"].iloc[0] <= 40                                       # 20 ciclos
    assert abs(df["EMG1_MNF"].iloc[0] - 100) < 6
    assert df["ratio_G12_G34"].iloc[0] == pytest.approx(10, rel=0.05)
    assert df["diff_G12_G34"].iloc[0] > 0
    assert not df.isna().any().any()


def test_features_frequency_ordering():
    t = np.arange(200) / FS
    lo = np.tile(np.sin(2 * np.pi * 40 * t)[None, :, None], (1, 1, 4))
    hi = np.tile(np.sin(2 * np.pi * 200 * t)[None, :, None], (1, 1, 4))
    assert extract_features(lo, FS, CH)["EMG1_MNF"][0] < extract_features(hi, FS, CH)["EMG1_MNF"][0]


def _fake_npz(path, subject, seed):
    rng = np.random.default_rng(seed)
    labels = ["Reposo", "Sonrisa", "Disgusto", "Sorprendido", "Triste"]
    seg, trial, rep, lab, lid = [], [], [], [], []
    sid = 0
    for r in range(3):                                          # 3 reps
        for i, name in enumerate(labels[1:]):
            for nm, lbl in ((name, name), ("Reposo", "Reposo")):
                seg += [sid] * 4; trial += [r * 4 + i] * 4; rep += [r] * 4
                lab += [lbl] * 4; lid += [labels.index(lbl)] * 4; sid += 1
    seg += [sid] * 4; trial += [-1] * 4; rep += [-1] * 4; lab += ["Reposo"] * 4; lid += [0] * 4   # reposo inicial
    n = len(seg)
    np.savez_compressed(path, windows=rng.normal(0, 1e-5, (n, 200, 4)).astype("float32"),
                        segment_id=seg, label=lab, label_id=lid, trial=trial, rep=rep,
                        channels=CH, fs_hz=1000.0, subject=subject, session="session_01")


@pytest.fixture
def seg_dir(tmp_path):
    _fake_npz(tmp_path / "s1_session_01_windows.npz", "s1", 0)
    _fake_npz(tmp_path / "s2_session_01_windows.npz", "s2", 1)
    return tmp_path


def test_build_dataset_schema(seg_dir):
    df, raw = build_dataset(seg_dir)
    assert set(df.subject) == {"s1", "s2"}
    assert len(feature_columns(df)) == 4 * 17 + 2              # 17 feats/canal + ratio/diff
    assert raw["windows"].shape[0] == len(df)
    assert not df[feature_columns(df)].isna().any().any()


def test_loso_splits(seg_dir):
    df, _ = build_dataset(seg_dir)
    folds = list(get_splits(df))                                # auto -> loso (2 sujetos)
    assert [f.name for f in folds] == ["subject=s1", "subject=s2"]
    for f in folds:
        assert set(df.iloc[f.test].subject).isdisjoint(set(df.iloc[f.train].subject))


def test_leave_one_rep_out_no_leakage(seg_dir):
    df, _ = build_dataset(seg_dir)
    df = df[df.subject == "s1"].reset_index(drop=True)          # un solo sujeto
    folds = list(get_splits(df))
    assert [f.name for f in folds] == ["rep=0", "rep=1", "rep=2"]   # rep=-1 nunca es test
    for f in folds:
        tr, te = df.iloc[f.train], df.iloc[f.test]
        assert set(zip(tr.subject, tr.segment_id)).isdisjoint(set(zip(te.subject, te.segment_id)))
        assert (tr.rep == -1).any()                             # reposo inicial siempre en train
        assert te.label.nunique() == 5


def test_export(seg_dir, tmp_path):
    df, raw = build_dataset(seg_dir)
    p_csv, p_raw = export_dataset(df, raw, tmp_path / "out")
    assert p_csv.exists() and p_raw.exists()


def test_export_load_roundtrip(seg_dir, tmp_path):
    df, raw = build_dataset(seg_dir)
    export_dataset(df, raw, tmp_path / "out")
    df2, raw2 = load_dataset(tmp_path / "out")                  # no debe requerir pickle
    assert len(df2) == len(df) and raw2["windows"].shape[0] == len(df2)
    assert raw2["label"].tolist() == df["label"].tolist()
    assert load_dataset(tmp_path / "out", with_raw=False)[1] is None