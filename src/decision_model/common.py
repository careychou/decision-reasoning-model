"""Shared prompt + single-pass label scorer (next-token logits over a fixed set of label tokens).

Binary tasks read the {No, Yes} logits; multiclass / ordinal / scalar tasks read K label tokens
(letters or digits) declared in task.json "output" (see decision_model.heads)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from decision_model.heads import normalize_output

SYSTEM_PROMPT = (
    "You are a decision model. Given a proposal with its context and reasoning, decide "
    "whether the person it is for would say yes to it. Answer with exactly one word: Yes or No."
)
QUESTION = "Would the person say yes to this? Answer Yes or No."
OUTPUT = normalize_output(None)


def set_task(path: str | Path | None) -> None:
    """Swap the system prompt + question for another decision task.

    `path` is a task.json with {"system", "question", optional "output"} (see examples/*/task.json).
    No "output" block means binary Yes/No. None keeps the generic Yes/No default above.
    """
    global SYSTEM_PROMPT, QUESTION, OUTPUT
    if path:
        t = json.loads(Path(path).read_text())
        SYSTEM_PROMPT, QUESTION = t["system"], t["question"]
        OUTPUT = normalize_output(t.get("output"))
    return OUTPUT


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{text}\n\n{QUESTION}"},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:  # Qwen3 hybrid-thinking templates: force non-thinking mode
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def label_token_ids(tokenizer, labels: list[str] | None = None) -> list[int]:
    """Token id of each label as the assistant's first token (labels must be single tokens)."""
    ids = []
    for word in labels or OUTPUT["labels"]:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"'{word}' is not a single token for this tokenizer: {toks}")
        ids.append(toks[0])
    return ids


def answer_token_ids(tokenizer) -> tuple[int, int]:
    """Token ids for 'No' and 'Yes' (binary tasks)."""
    no, yes = label_token_ids(tokenizer, ["No", "Yes"])
    return no, yes


def setup_tokenizer(tokenizer):
    tokenizer.padding_side = "left"  # last position == next-token position for every row
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def label_logits(model, tokenizer, prompts: list[str], ans_ids, max_len: int, device):
    """Return [B, K] logits for the K label tokens at the next-token position."""
    # Truncate from the left: the question and answer slot sit at the end of the prompt.
    tokenizer.truncation_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len).to(device)
    # logits_to_keep=1: project only the last position onto the vocab (avoids a B x L x 152k tensor).
    out = model(**enc, logits_to_keep=1)
    last = out.logits[:, -1, :]
    return last[:, list(ans_ids)].float()


two_way_logits = label_logits  # binary: ans_ids = (No, Yes)


@torch.no_grad()
def score_logits(model, tokenizer, texts: list[str], max_len=768, batch_size=16, device="cuda"):
    """Deterministic [K] label logits per text for the current task (no sampling, one pass)."""
    model.eval()
    ids = label_token_ids(tokenizer)
    res = []
    for i in range(0, len(texts), batch_size):
        prompts = [build_prompt(tokenizer, t) for t in texts[i:i + batch_size]]
        res.extend(label_logits(model, tokenizer, prompts, ids, max_len, device).cpu().tolist())
    return res


@torch.no_grad()
def score(model, tokenizer, texts: list[str], max_len=768, batch_size=16, device="cuda"):
    """Deterministic P(Yes) for each text (no sampling, single forward pass)."""
    model.eval()
    ans = answer_token_ids(tokenizer)
    log_odds = []
    for i in range(0, len(texts), batch_size):
        prompts = [build_prompt(tokenizer, t) for t in texts[i:i + batch_size]]
        lg = two_way_logits(model, tokenizer, prompts, ans, max_len, device)
        log_odds.extend((lg[:, 1] - lg[:, 0]).cpu().tolist())
    return log_odds  # logit of P(Yes) restricted to {No, Yes}
