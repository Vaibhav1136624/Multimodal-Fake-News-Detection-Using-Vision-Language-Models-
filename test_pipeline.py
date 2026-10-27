import numpy as np
import pytest
from fake_news import fusion, split_groups, read_manifest


def test_interaction_alignment():
    t = np.array([[1., 0.], [0., 1.]])
    v = np.array([[1., 0.], [1., 0.]])
    x = fusion(t, v)
    assert x.shape == (2, 9)
    np.testing.assert_allclose(x[:, -1], [1., 0.])
    np.testing.assert_allclose(x[:, 4:6], [[0, 0], [1, 1]])


def test_groups_never_cross_splits():
    groups = np.repeat(np.arange(20), 10)
    y = np.tile([0, 1], 100)
    tr, va, te = split_groups(y, groups)
    assert not set(groups[tr]) & set(groups[va])
    assert not set(groups[tr]) & set(groups[te])
    assert not set(groups[va]) & set(groups[te])
    assert len(set(tr) | set(va) | set(te)) == len(y)


def test_bad_embeddings_rejected():
    with pytest.raises(ValueError): fusion([[np.nan]], [[1.]])
    with pytest.raises(ValueError): fusion([[1., 2.]], [[1.]])


def test_unlabeled_manifest_and_missing_image(tmp_path):
    import pandas as pd
    image = tmp_path / 'photo.png'
    image.write_bytes(b'placeholder')
    path = tmp_path / 'manifest.csv'
    pd.DataFrame([{'id': 'a', 'text': 'claim', 'image': 'photo.png', 'group': 'event'}]).to_csv(path, index=False)
    df = read_manifest(path)
    assert df.label.tolist() == [-1]
    assert df.image.tolist() == [str(image)]
    image.unlink()
    with pytest.raises(ValueError, match='Image not found'): read_manifest(path)


def test_merf_unseen_groups_and_serialization(tmp_path):
    import pandas as pd
    import joblib
    from merf.merf import MERF
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 4))
    y = np.tile([0., 1.], 12)
    z = np.ones((24, 1))
    groups = pd.Series(np.repeat(['a', 'b', 'c', 'd'], 6))
    model = MERF(RandomForestRegressor(n_estimators=10, random_state=7), max_iterations=2)
    model.fit(x, z, groups, y)
    unseen = pd.Series(['new'] * 24)
    scores = model.predict(x, z, unseen)
    np.testing.assert_allclose(scores, model.trained_fe_model.predict(x))
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(x, z, groups), model.predict(x, z, groups))
