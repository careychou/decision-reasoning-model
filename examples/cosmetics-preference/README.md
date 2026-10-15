# Cosmetics preference: order of preference

**Shape:** ordinal, five bands (`1` would not buy … `5` top pick), trained with cross-entropy plus EMD.

A shopper is browsing skincare. We know her skin type, the finish she likes, what she usually
spends, and what she rated, kept or sent back before. For each candidate product the DRM reads
that profile, the product and the signals as plain text and returns a distribution over five
ordered bands. The expected band is the score; sort by it and you have her order of preference.

```text
Shopper: Dry skin, prefers a dewy finish, usually spends about $32.
Product: Candidate: hydrating serum (serum), $30, dewy finish, hydrating, fragrance-free.
Signals: She rated a similar product 5 out of 5. No returns in this category.
```

```json
{"label": "5", "name": "top pick", "confidence": 0.71, "expected_band": 4.6, "route": "auto"}
```

Run the same profile against a tinted moisturizer, a matte foundation and a retinol night cream,
and the list sorts itself: serum 4.6, tinted moisturizer 4.1, matte foundation 2.3, retinol 1.8.

## Why ordinal and why EMD

Bands have an order. Calling a "top pick" a "likely" is a near miss; calling it "would not buy" is
not. Plain cross-entropy treats both misses the same. The squared earth mover's distance compares
the cumulative distributions, so it charges more the further the mass sits from the true band.
`task.json` sets `"loss": "ce+emd"` to use both.

## The hidden rule

`gen.py` starts every product at 3 and moves it: finish match +1 (dewy vs matte −1), hydrating on
dry or combination skin +0.8, fragrance on sensitive skin −1.5, retinol or acids on sensitive skin
−1, price over 1.6× her usual −1, her rating on a similar item ±0.6 per star, a past return −1.
The text never states the rule; the model has to find it from the wording.

```bash
python examples/cosmetics-preference/gen.py --out data/cosmetics-preference
```

The numbers in the example above are illustrative, from the essay. Train and score it yourself to
see what your model returns.
