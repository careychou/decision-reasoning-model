"""LoRA fine-tune an instruct LLM to score a decision via next-token logits over K label tokens.

Binary (No/Yes), multiclass, ordinal and scalar (expected value over digit bins) share one loss:
cross-entropy between softmax(label logits) and a soft target (one-hot, or two-hot for scalar)."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from decision_model.common import (build_prompt, label_logits, label_token_ids, load_jsonl,
                                   pick_device, score_logits, set_task, setup_tokenizer)
from decision_model.heads import is_right, soft_target, softmax


def ordered_loss(logits: torch.Tensor, target: torch.Tensor, out: dict) -> torch.Tensor:
    """Soft-target CE, squared EMD over CDFs, or both (output.loss)."""
    logp = F.log_softmax(logits, -1)
    ce = -(target * logp).sum(-1).mean()
    if out.get("loss", "ce") == "ce":
        return ce
    emd = ((logp.exp().cumsum(-1) - target.cumsum(-1)) ** 2).sum(-1).mean()
    return emd if out["loss"] == "emd" else ce + out.get("emd_weight", 1.0) * emd


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/qwen2.5-1.5b")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default=None, help="task.json (system prompt + question); default = generic Yes/No")
    a = ap.parse_args()
    out_spec = set_task(a.task)

    set_seed(a.seed)
    device = pick_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = setup_tokenizer(AutoTokenizer.from_pretrained(a.model))
    ans = label_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(device)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    train = load_jsonl(Path(a.data) / "train.jsonl")
    val = load_jsonl(Path(a.data) / "val.jsonl")
    steps_per_epoch = math.ceil(len(train) / (a.batch_size * a.grad_accum))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    log = []
    step, micro, t0 = 0, 0, time.time()
    rng = random.Random(a.seed)
    model.train()
    while step < total_steps:
        order = list(range(len(train)))
        rng.shuffle(order)
        for i in range(0, len(order), a.batch_size):
            batch = [train[j] for j in order[i:i + a.batch_size]]
            prompts = [build_prompt(tok, r["text"]) for r in batch]
            target = torch.tensor([soft_target(r, out_spec) for r in batch], device=device)
            logits = label_logits(model, tok, prompts, ans, a.max_len, device)
            loss = ordered_loss(logits, target, out_spec) / a.grad_accum
            loss.backward()
            micro += 1
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step == total_steps:
                rec = {"step": step, "loss": round(loss.item() * a.grad_accum, 4),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.append(rec)
            if step >= total_steps:
                break

    lg = score_logits(model, tok, [r["text"] for r in val], a.max_len, 32, device)
    val_acc = sum(is_right(softmax(z), r, out_spec) for z, r in zip(lg, val)) / len(val)
    print(json.dumps({"val_acc": round(val_acc, 4)}))

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if a.task:  # scorer reloads the same prompt from the adapter dir
        (out / "adapter" / "task.json").write_text(Path(a.task).read_text())
    (out / "train_log.json").write_text(json.dumps(
        {"args": vars(a), "log": log, "val_acc": val_acc,
         "train_seconds": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
