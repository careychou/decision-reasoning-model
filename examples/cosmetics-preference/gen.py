"""Synthetic data: how much would she like this product? Ordered 1-5 preference band.

One row = one shopper profile + one candidate product + the signals behind the pick.
The band comes from a hidden rule over her skin, finish preference, sensitivities, spend and
what she rated, kept or returned before. Score every candidate and sort by the expected band.

    python examples/cosmetics-preference/gen.py --out data/cosmetics-preference
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from decision_model.synth import assemble, bullets_or_prose, main, pick  # noqa: E402

SKIN = ["dry", "oily", "combination", "sensitive"]
FINISH = ["dewy", "natural", "matte"]
PRODUCTS = {
    "serum": ["hydrating serum", "vitamin C serum", "niacinamide serum"],
    "moisturizer": ["tinted moisturizer", "gel moisturizer", "rich cream"],
    "foundation": ["matte foundation", "skin tint", "satin foundation"],
    "night cream": ["retinol night cream", "peptide night cream"],
    "sunscreen": ["SPF 50 fluid", "mineral SPF 30"],
    "cleanser": ["foaming cleanser", "cream cleanser"],
}
ACTIVES = ["none", "retinol", "vitamin C", "acids"]


def sample(rng: random.Random, uid: str) -> dict:
    cat = rng.choice(list(PRODUCTS))
    usual = rng.choice([18, 25, 32, 40, 55])
    return {
        "uid": uid, "category": cat, "product": rng.choice(PRODUCTS[cat]),
        "skin": rng.choice(SKIN), "finish_pref": rng.choice(FINISH),
        "finish": rng.choice(FINISH), "hydrating": rng.random() < 0.5,
        "fragrance": rng.random() < 0.35, "active": rng.choice(ACTIVES),
        "usual_spend": usual, "price": round(usual * rng.uniform(0.6, 2.2)),
        "similar_rating": rng.choice([None, None, 1, 2, 3, 4, 5]),
        "returned_similar": rng.random() < 0.15, "reviews": rng.choice([12, 85, 340, 2100]),
    }


def preference(z: dict) -> float:
    s = 3.0
    if z["finish"] == z["finish_pref"]:
        s += 1.0
    elif {z["finish"], z["finish_pref"]} == {"dewy", "matte"}:
        s -= 1.0
    if z["hydrating"] and z["skin"] in ("dry", "combination"):
        s += 0.8
    if z["skin"] == "dry" and z["finish"] == "matte":
        s -= 0.6
    if z["skin"] == "oily" and z["finish"] == "dewy":
        s -= 0.6
    if z["fragrance"] and z["skin"] == "sensitive":
        s -= 1.5
    if z["active"] in ("retinol", "acids") and z["skin"] == "sensitive":
        s -= 1.0
    if z["price"] > 1.6 * z["usual_spend"]:
        s -= 1.0
    elif z["price"] <= z["usual_spend"]:
        s += 0.3
    if z["similar_rating"] is not None:
        s += (z["similar_rating"] - 3) * 0.6
    if z["returned_similar"]:
        s -= 1.0
    return s


def label(z: dict) -> dict:
    return {"label": min(max(round(preference(z)), 1), 5) - 1}


def render(rng: random.Random, z: dict, ood: bool) -> str:
    profile = pick(rng, [
        f"Shopper has {z['skin']} skin and prefers a {z['finish_pref']} finish. She usually spends about ${z['usual_spend']} per item.",
        f"Her profile: {z['skin']} skin, likes {z['finish_pref']} looks, typical item around ${z['usual_spend']}.",
        f"Skin type {z['skin']}; finish preference {z['finish_pref']}; usual price point ${z['usual_spend']}.",
        f"skin={z['skin']} finish_pref={z['finish_pref']} usual_spend_usd={z['usual_spend']}",
    ], ood)
    traits = [f"{z['finish']} finish"]
    if z["hydrating"]:
        traits.append("hydrating")
    traits.append("fragranced" if z["fragrance"] else "fragrance-free")
    if z["active"] != "none":
        traits.append(f"with {z['active']}")
    product = pick(rng, [
        f"Candidate: {z['product']} ({z['category']}), ${z['price']}, " + ", ".join(traits) + ".",
        f"We could show her the {z['product']} at ${z['price']}. It is " + ", ".join(traits) + ".",
        f"Product under review is a {z['category']}: {z['product']}, priced ${z['price']}; " + "; ".join(traits) + ".",
        f"item={z['product']} | cat={z['category']} | price=${z['price']} | traits=" + "/".join(traits),
    ], ood)
    facts = [pick(rng, [f"It has {z['reviews']} reviews.", f"Review count: {z['reviews']}.",
                        f"{z['reviews']} people have reviewed it.", f"n_reviews={z['reviews']}"], ood)]
    r = z["similar_rating"]
    if r is not None:
        facts.append(pick(rng, [f"She rated a similar product {r} out of 5.",
                                f"Her rating on the closest past purchase was {r}/5.",
                                f"Past rating for a look-alike item: {r} stars.",
                                f"similar_item_rating={r}"], ood))
    if z["returned_similar"]:
        facts.append(pick(rng, ["She returned a similar item last quarter.",
                                "A close match came back as a return.",
                                "Return history includes a near-identical product.",
                                "returned_similar=true"], ood))
    elif rng.random() < 0.5:
        facts.append(pick(rng, ["No returns in this category.", "She kept everything she bought here.",
                                "Zero returns on file for this type.", "returned_similar=false"], ood))
    return assemble(rng, [("Shopper", profile), ("Product", product),
                          ("Signals", bullets_or_prose(rng, facts, ood))])


if __name__ == "__main__":
    main(sample, label, render, kind=lambda z: z["category"],
         task_json=Path(__file__).with_name("task.json"), n_classes=5)
