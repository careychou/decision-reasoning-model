import math

import pytest

from decision_model import heads

SCALAR = heads.normalize_output({"type": "scalar", "labels": list("01234"),
                                 "values": [-1.0, -0.5, 0.0, 0.5, 1.0], "tolerance": 0.25})


def test_default_is_binary():
    o = heads.normalize_output(None)
    assert o["type"] == "binary" and o["labels"] == ["No", "Yes"]


def test_scalar_needs_ascending_values():
    with pytest.raises(ValueError):
        heads.normalize_output({"type": "scalar", "labels": ["0", "1"], "values": [1, 0]})


def test_two_hot_reproduces_target():
    for y in (-1.0, -0.7, 0.0, 0.3, 1.0):
        t = heads.soft_target({"label": 0, "target": y}, SCALAR)
        assert math.isclose(sum(t), 1.0)
        assert math.isclose(heads.expectation(t, SCALAR["values"]), y)


def test_two_hot_clips_out_of_range():
    t = heads.soft_target({"label": 4, "target": 3.0}, SCALAR)
    assert heads.expectation(t, SCALAR["values"]) == 1.0


def test_one_hot_for_classes():
    o = heads.normalize_output({"type": "multiclass", "labels": list("ABC")})
    assert heads.soft_target({"label": 2}, o) == [0.0, 0.0, 1.0]


def test_read_multiclass_and_scalar():
    o = heads.normalize_output({"type": "ordinal", "labels": list("ABCD"), "names": list("wxyz")})
    r = heads.read([0.1, 0.6, 0.2, 0.1], o)
    assert r["label"] == "B" and r["name"] == "x" and r["confidence"] == 0.6
    assert abs(r["expected_band"] - 2.3) < 1e-9
    s = heads.read([0, 0, 1, 0, 0], SCALAR)
    assert s["estimate"] == 0.0 and s["spread"] == 0.0


def test_fit_temperature_softens_overconfidence():
    logits = [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]
    targets = [[1, 0], [0, 1], [1, 0], [0, 1]]
    assert heads.fit_temperature(logits, targets) > 1.0


def test_auto_threshold_hits_precision():
    conf = [i / 100 for i in range(100)]
    right = [c >= 0.5 for c in conf]
    t = heads.auto_threshold(conf, right, 0.98, min_support=10)
    cov = heads.coverage(conf, right, t)
    assert cov["precision"] >= 0.98 and t <= 0.51


def test_emd2_penalizes_far_misses_more():
    t = [1.0, 0.0, 0.0, 0.0]
    near = heads.emd2([0.0, 1.0, 0.0, 0.0], t)
    far = heads.emd2([0.0, 0.0, 0.0, 1.0], t)
    assert heads.emd2(t, t) == 0.0 and near == 1.0 and far == 3.0


def test_emd_loss_needs_ordered_output():
    import pytest
    with pytest.raises(ValueError):
        heads.normalize_output({"type": "multiclass", "labels": ["A", "B"], "loss": "emd"})
    assert heads.normalize_output({"type": "ordinal", "labels": ["A", "B"], "loss": "ce+emd"})["loss"] == "ce+emd"
