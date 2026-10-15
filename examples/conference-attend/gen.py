"""Synthetic data: should I attend this conference? Yes / No.

One row = one invite: the event, your calendar, the cost against what's left in the budget,
the travel, and the case for going. The answer comes from a hidden rule over those facts.
Read P(Yes) with two lines: above the top line go, below the bottom line skip, between them ask.

    python examples/conference-attend/gen.py --out data/conference-attend
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from decision_model.synth import assemble, bullets_or_prose, main, pick  # noqa: E402

EVENTS = ["Applied ML Summit", "Data Engineering Days", "Product Analytics Forum",
          "Cloud Native Conf", "Design Systems Week", "AI Agents Expo"]
CITIES = {"Austin": 3, "Denver": 4, "Chicago": 4, "Seattle": 6, "Boston": 6, "Lisbon": 13}
CONFLICTS = ["none", "a recurring 1:1", "your team offsite", "a customer deadline"]
CONFLICT_COST = {"none": 0.0, "a recurring 1:1": 0.4, "your team offsite": 1.6, "a customer deadline": 3.0}


def sample(rng: random.Random, uid: str) -> dict:
    return {
        "uid": uid, "event": rng.choice(EVENTS), "city": rng.choice(list(CITIES)),
        "relevant_talks": rng.choice([0, 0, 1, 2, 3, 4]), "speaking": rng.random() < 0.15,
        "conflict": rng.choice(CONFLICTS), "fee": rng.choice([0, 450, 900, 1450, 2100]),
        "budget_left": rng.choice([800, 2000, 3500, 5000, 8000]),
        "colleague_notes": rng.random() < 0.35, "recorded": rng.random() < 0.4,
        "trips_this_year": rng.choice([0, 1, 2, 3, 4]), "manager_ok": rng.random() < 0.7,
    }


def attend(z: dict) -> bool:
    s = 1.1 * min(z["relevant_talks"], 3) + (2.5 if z["speaking"] else 0.0)
    s -= CONFLICT_COST[z["conflict"]]
    travel = 600 if CITIES[z["city"]] < 8 else 1300
    if z["fee"] + travel > z["budget_left"]:
        s -= 2.0
    if CITIES[z["city"]] >= 8:
        s -= 0.7
    if z["recorded"]:
        s -= 0.8
    if z["colleague_notes"]:
        s -= 0.6
    s -= 0.25 * z["trips_this_year"]
    if not z["manager_ok"]:
        s -= 1.0
    return s > 0.0


def label(z: dict) -> dict:
    return {"label": int(attend(z))}


def render(rng: random.Random, z: dict, ood: bool) -> str:
    h = CITIES[z["city"]]
    ctx = pick(rng, [
        f'"{z["event"]}" in {z["city"]}, about {h} hours door to door. On your calendar that week: {z["conflict"]}.',
        f'You were invited to {z["event"]} ({z["city"]}). Travel is roughly {h} hours each way; the same week has {z["conflict"]}.',
        f'Event: {z["event"]}. Location: {z["city"]} ({h}h travel). Calendar conflict: {z["conflict"]}.',
        f'event="{z["event"]}" city={z["city"]} travel_h={h} conflict="{z["conflict"]}"',
    ], ood)
    fee = "free to attend" if z["fee"] == 0 else f"${z['fee']:,} registration"
    proposal = pick(rng, [
        f"Register and fly out; {fee} plus travel. Training budget left this year: ${z['budget_left']:,}.",
        f"The ask is to go: {fee}, travel extra, against ${z['budget_left']:,} left in your budget.",
        f"Go, at {fee} plus flights and hotel. Remaining budget ${z['budget_left']:,}.",
        f"action=attend cost={fee} budget_left_usd={z['budget_left']}",
    ], ood)
    t = z["relevant_talks"]
    facts = [pick(rng, [f"{t} talks touch your current project." if t else "No talks on your current project.",
                        f"Agenda overlap with your work: {t} sessions.",
                        f"You flagged {t} sessions as relevant.",
                        f"relevant_talks={t}"], ood)]
    if z["speaking"]:
        facts.append(pick(rng, ["You have a speaking slot.", "They accepted your talk.",
                                "You are on the program as a speaker.", "speaker=true"], ood))
    if z["colleague_notes"]:
        facts.append(pick(rng, ["A colleague is going and offered notes.", "Someone on your team will attend and share notes.",
                                "A teammate can cover it and report back.", "colleague_attending=true"], ood))
    if z["recorded"]:
        facts.append(pick(rng, ["Sessions are recorded.", "Talks go online a week later.",
                                "Recordings will be posted.", "recorded=true"], ood))
    facts.append(pick(rng, [f"You've taken {z['trips_this_year']} work trips this year.",
                            f"Trips so far this year: {z['trips_this_year']}.",
                            f"{z['trips_this_year']} trips already on the books this year.",
                            f"trips_ytd={z['trips_this_year']}"], ood))
    facts.append(pick(rng, ["Your manager is supportive." if z["manager_ok"] else "Your manager has doubts.",
                            "Manager: in favor." if z["manager_ok"] else "Manager: not convinced.",
                            "Your lead said go if you think it's worth it." if z["manager_ok"] else "Your lead would rather you stay.",
                            f"manager_ok={str(z['manager_ok']).lower()}"], ood))
    return assemble(rng, [("Context", ctx), ("Proposal", proposal),
                          ("Reasoning", bullets_or_prose(rng, facts, ood))])


if __name__ == "__main__":
    main(sample, label, render, kind=lambda z: "speaker" if z["speaking"] else "attendee",
         task_json=Path(__file__).with_name("task.json"), n_classes=2)
