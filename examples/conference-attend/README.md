# Conference: attend, Yes or No

**Shape:** binary (`No` skip, `Yes` attend), read as a calibrated P(Yes) with two lines.

An assistant gets a conference invite. It knows your calendar, what's left in your training
budget, how far the trip is, which talks touch your work, and whether a colleague is already
going. The DRM reads all of that as one block of text and returns P(Yes).

```text
context:   "Applied ML Summit", Austin, Mar 12–14; your team offsite is on the 13th
proposal:  register and fly out, $1,450 plus travel
reasoning: two talks on your current project; a colleague is going and offered notes
question:  Would you attend? Yes or No.
```

```json
{"p_yes": 0.68, "decision": "Yes", "route": "to_person"}
```

Ask it again and you get 0.68 again. It is one forward pass over the same text, with no sampling.

## Two lines, three lanes

`evaluate.py` picks two thresholds on the validation set so each automatic lane hits a target
precision (98% by default):

| P(Yes) | lane |
|---|---|
| above the top line | `auto_yes`: register |
| below the bottom line | `auto_no`: decline |
| in between | `to_person`: ask you |

0.68 sits between the lines, so the assistant asks instead of deciding. That middle lane is the
point. The model only acts on its own where it has earned it.

## The hidden rule

Relevant talks (+1.1 each, up to three) and a speaking slot (+2.5) push toward Yes. A conflict
(a 1:1 −0.4, the team offsite −1.6, a customer deadline −3), going over budget (−2), long travel
(−0.7), recorded sessions (−0.8), a colleague taking notes (−0.6), trips already taken (−0.25
each) and a doubtful manager (−1) push toward No. Yes when the total is above zero.

```bash
python examples/conference-attend/gen.py --out data/conference-attend
```
