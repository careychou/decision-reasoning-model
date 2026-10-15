# Travel budget: one number with a spread

**Shape:** scalar. Nine digit bins from $1,000 to $3,000 in $250 steps; the estimate is the
probability-weighted value and the spread is how wide that distribution is.

She is planning a trip. The assistant knows the city, the nights, how far ahead she's booking,
whether it's a busy week, how she likes to fly and get around, and how her last trips came in
against plan. The DRM reads it as text and returns a dollar amount.

```text
Context:  Trip to Austin, 3 nights, booking 21 days out. It's the week of a big conference in town.
Proposal: Set aside a budget for flights, hotel, meals and getting around.
Signals:  She flies economy. Wants to stay walking distance from the venue.
          Gets around by rideshare. Her last trips landed right on plan.
```

```json
{"estimate": 1850, "spread": 200, "route": "auto"}
```

Set aside $1,850 and flag anything past $2,050. A per-diem rule of thumb would say $1,200 here,
because it can't see the busy week or the walk-to-venue hotel.

## How a number comes out of next-token logits

The question lists the bins (`0` = $1,000 … `8` = $3,000 or more). Each digit is one token, so
one forward pass gives nine logits. Softmax them, then:

- estimate = Σ pₖ · valueₖ
- spread = √Σ pₖ · (valueₖ − estimate)²

Training uses a **two-hot** target: a true cost of $1,820 puts 72% of the weight on the $1,750
bin and 28% on $2,000, so the expected value of the target is exactly $1,820. A narrow spread is
the "sure enough" signal; `evaluate.py` tunes the cut on validation so the auto lane lands within
`tolerance` ($200) at the target precision.

## The hidden rule

`gen.py` prices the trip as flight (fare by city × seat class; +40% if booked under two weeks out,
−10% if over 40 days) + hotel × nights (+30% on a busy week, +15% for walk-to-venue) + meals per
day + local travel per day, then scales by her past over- or under-run.

```bash
python examples/travel-budget/gen.py --out data/travel-budget
```
