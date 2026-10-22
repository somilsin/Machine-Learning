"""Data and reward helpers for my classic RLHF project.

Human preferences: Anthropic/hh-rlhf (helpful-base). Each row has a chosen
and a rejected transcript that share every turn except the last assistant
reply. Verifiable rewards: openai/gsm8k, where I can check the answer.
"""
import random
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import DEVICE  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"


def parse_hh(text):
    parts = re.split(r"\n\n(Human|Assistant): ", "\n\n" + text.strip())
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        msgs.append({"role": "user" if role == "Human" else "assistant", "content": content.strip()})
    return msgs


def hh_pairs(split="train", n=4000, seed=7, max_chars=1500):
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split).shuffle(seed=seed)
    out = []
    for row in ds:
        c, r = parse_hh(row["chosen"]), parse_hh(row["rejected"])
        if (len(c) < 2 or len(c) != len(r) or c[-1]["role"] != "assistant" or c[:-1] != r[:-1]
                or c[-1]["content"] == r[-1]["content"] or len(row["chosen"]) > max_chars):
            continue
        if any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(c)):
            continue
        out.append({"prompt": c[:-1], "chosen": c[-1]["content"], "rejected": r[-1]["content"]})
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- reward model scoring

@torch.no_grad()
def rm_scores(rm, tok, prompts, answers, bs=4, max_len=640):
    """Scalar reward for each (conversation, answer) pair."""
    scores = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template(p + [{"role": "assistant", "content": a}], tokenize=False)
                 for p, a in zip(prompts[i:i + bs], answers[i:i + bs])]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(DEVICE)
        scores += rm(**enc).logits.squeeze(-1).float().tolist()
    return scores


# ---------------------------------------------------------------- GSM8K verifier

GSM8K_INSTRUCTION = "\nSolve it step by step. End with the final answer on its own line as: #### <number>"


def gsm8k_items(split="train", n=None, seed=7):
    ds = list(load_dataset("openai/gsm8k", "main", split=split))
    random.Random(seed).shuffle(ds)
    ds = ds[:n] if n else ds
    return [{"q": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for r in ds]


NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract_answer(text):
    r"""Final number: after '####' if present, else inside \boxed{}, else the last number in the text."""
    # note to self: Qwen2.5-Instruct writes \boxed{} and ignores my #### instruction. A strict
    # extractor gave every sample 0 reward and every GRPO group went flat (zero gradient).
    for marker in ("####", r"\boxed{"):
        if marker in text:
            m = re.search(NUMBER, text.split(marker)[-1])
            if m:
                return m.group().replace(",", "")
    nums = re.findall(NUMBER, text)
    return nums[-1].replace(",", "") if nums else None


def is_correct(text, gold):
    pred = extract_answer(text)
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def gsm8k_reward(text, gold):
    """1.0 for the right number plus 0.1 for using the requested '####' format."""
    return float(is_correct(text, gold)) + (0.1 if "####" in text else 0.0)
