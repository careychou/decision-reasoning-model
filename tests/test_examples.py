import importlib.util
import json
import random
from pathlib import Path

import pytest

from decision_model import heads

ROOT = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = ["cosmetics-preference", "conference-attend", "travel-budget"]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / name / "gen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", EXAMPLES)
def test_task_json_is_valid(name):
    o = heads.normalize_output(json.loads((ROOT / name / "task.json").read_text())["output"])
    # labels must be single tokens: Yes/No, or one letter / digit each
    assert o["type"] == "binary" or all(len(lab) == 1 for lab in o["labels"])


@pytest.mark.parametrize("name", EXAMPLES)
def test_render_is_deterministic_and_labels_in_range(name):
    g = load(name)
    n = len(json.loads((ROOT / name / "task.json").read_text())["output"]["labels"])
    for i in range(200):
        z = g.sample(random.Random(i), str(i))
        assert g.render(random.Random(3), z, False) == g.render(random.Random(3), z, False)
        assert 0 <= g.label(z)["label"] < n


@pytest.mark.parametrize("name", EXAMPLES)
def test_no_single_label_dominates(name):
    g = load(name)
    rng = random.Random(0)
    ys = [g.label(g.sample(rng, str(i)))["label"] for i in range(3000)]
    assert max(ys.count(k) for k in set(ys)) / len(ys) < 0.7


def test_examples_from_the_essay():
    c = load("cosmetics-preference")
    serum = {"skin": "dry", "finish_pref": "dewy", "finish": "dewy", "hydrating": True, "fragrance": False,
             "active": "none", "usual_spend": 32, "price": 30, "similar_rating": 5, "returned_similar": False}
    matte = serum | {"finish": "matte", "hydrating": False, "similar_rating": 2}
    assert c.label(serum)["label"] == 4 and c.label(matte)["label"] <= 1
    t = load("travel-budget")
    austin = {"city": "Austin", "nights": 3, "lead_days": 21, "peak_week": True, "seat": "economy",
              "walk_to_venue": True, "rideshare": True, "overrun_pct": 0}
    assert 1700 <= t.cost(austin) <= 2000
