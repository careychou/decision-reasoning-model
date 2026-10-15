"""Synthetic data: what should this trip cost? A dollar amount with a spread.

One row = one trip: where and when, how far ahead it's booked, her travel habits and what her
past trips came in at. The target comes from a hidden cost model (flight + hotel + meals + local
travel, scaled by her own habits). The answer is read as digit bins from $1,000 to $3,000; the
estimate is the probability-weighted value and the spread says how sure the model is.

    python examples/travel-budget/gen.py --out data/travel-budget
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from decision_model.synth import assemble, bullets_or_prose, main, pick  # noqa: E402

# city -> (typical round-trip economy fare, hotel per night, meals per day)
CITIES = {"Austin": (420, 210, 70), "Denver": (360, 190, 65), "Chicago": (380, 240, 75),
          "Seattle": (480, 250, 80), "Boston": (460, 280, 85), "New Orleans": (400, 220, 70)}
VALUES = [1000 + 250 * k for k in range(9)]


def sample(rng: random.Random, uid: str) -> dict:
    return {
        "uid": uid, "city": rng.choice(list(CITIES)), "nights": rng.choice([1, 2, 3, 3, 4, 5]),
        "lead_days": rng.choice([5, 10, 21, 45, 75]), "peak_week": rng.random() < 0.35,
        "seat": rng.choice(["economy", "economy", "extra legroom", "business"]),
        "walk_to_venue": rng.random() < 0.5, "rideshare": rng.random() < 0.5,
        "overrun_pct": rng.choice([-10, 0, 0, 5, 10, 20]),
    }


def cost(z: dict) -> float:
    fare, hotel, meals = CITIES[z["city"]]
    fare *= {"economy": 1.0, "extra legroom": 1.25, "business": 2.1}[z["seat"]]
    if z["lead_days"] < 14:
        fare *= 1.4
    elif z["lead_days"] > 40:
        fare *= 0.9
    hotel *= (1.3 if z["peak_week"] else 1.0) * (1.15 if z["walk_to_venue"] else 1.0)
    days = z["nights"] + 1
    local = days * (45 if z["rideshare"] else 15)
    total = fare + hotel * z["nights"] + meals * days + local
    return round(total * (1 + z["overrun_pct"] / 100), -1)


def label(z: dict) -> dict:
    y = cost(z)
    k = min(range(len(VALUES)), key=lambda i: abs(VALUES[i] - y))
    return {"label": k, "target": y}


def render(rng: random.Random, z: dict, ood: bool) -> str:
    n, lead = z["nights"], z["lead_days"]
    ctx = pick(rng, [
        f"Trip to {z['city']}, {n} night{'s' * (n > 1)}, booking {lead} days out." + (" It's the week of a big conference in town." if z["peak_week"] else ""),
        f"She's heading to {z['city']} for {n} night{'s' * (n > 1)}; flights booked {lead} days ahead." + (" Hotels are busy that week." if z["peak_week"] else ""),
        f"Destination {z['city']}. Length {n} night(s). Lead time {lead} days." + (" Peak demand week." if z["peak_week"] else ""),
        f"city={z['city']} nights={n} lead_days={lead} peak={str(z['peak_week']).lower()}",
    ], ood)
    proposal = pick(rng, [
        "Set a travel budget for this trip so she can book and get flagged if it runs over.",
        "The assistant needs one number to set aside for this trip.",
        "Set aside a budget for flights, hotel, meals and getting around.",
        "action=set_trip_budget",
    ], ood)
    facts = [
        pick(rng, [f"She flies {z['seat']}.", f"Seat preference: {z['seat']}.",
                   f"Usually books {z['seat']}.", f"seat={z['seat'].replace(' ', '_')}"], ood),
        pick(rng, ["She likes a hotel she can walk to the venue from." if z["walk_to_venue"] else "Any hotel nearby is fine.",
                   "Wants to stay walking distance from the venue." if z["walk_to_venue"] else "Not picky about hotel location.",
                   "Hotel must be close to the venue." if z["walk_to_venue"] else "Fine with a hotel a short ride away.",
                   f"walk_to_venue={str(z['walk_to_venue']).lower()}"], ood),
        pick(rng, ["She takes rideshares around town." if z["rideshare"] else "She uses transit around town.",
                   "Gets around by rideshare." if z["rideshare"] else "Gets around by train and bus.",
                   "Local travel: mostly cars." if z["rideshare"] else "Local travel: mostly public transit.",
                   f"rideshare={str(z['rideshare']).lower()}"], ood),
    ]
    o = z["overrun_pct"]
    facts.append(pick(rng, [
        f"Her last trips came in about {abs(o)}% {'over' if o > 0 else 'under'} plan." if o else "Her last trips landed right on plan.",
        f"Past trips ran {o:+d}% against budget." if o else "Past trips matched budget.",
        f"History: actuals were {abs(o)}% {'above' if o > 0 else 'below'} the plan." if o else "History: actuals matched the plan.",
        f"past_overrun_pct={o}",
    ], ood))
    return assemble(rng, [("Context", ctx), ("Proposal", proposal),
                          ("Signals", bullets_or_prose(rng, facts, ood))])


if __name__ == "__main__":
    main(sample, label, render, kind=lambda z: z["city"],
         task_json=Path(__file__).with_name("task.json"), n_classes=len(VALUES))
