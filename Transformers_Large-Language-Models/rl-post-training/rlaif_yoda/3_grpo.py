"""Stage 2b · online RLAIF with GRPO: the judge is the reward model, live.

Every step: 4 prompts x G samples -> judge scores -> group advantages ->
one clipped policy-gradient update with a KL penalty to the SFT model.
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import Logger, add_lora, grpo_step, load_model, prompt_ids, save_merged, seed_all  # noqa: E402
from common import RUNS, combined_reward, make_judge, yoda_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=str(RUNS / "yoda_sft"))
ap.add_argument("--judge", default="local", choices=["local", "openrouter"])
ap.add_argument("--judge_model", default=None)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--prompts_per_step", type=int, default=4)
ap.add_argument("--G", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--w_style", type=float, default=0.7)
ap.add_argument("--max_new_tokens", type=int, default=96)
ap.add_argument("--micro_bs", type=int, default=4)
ap.add_argument("--out", default=str(RUNS / "yoda_grpo"))
args = ap.parse_args()

seed_all()
train, _ = yoda_split()
model, tok = load_model(args.policy, for_training=True)
model = add_lora(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
judge = make_judge(args.judge, args.judge_model)
log = Logger(str(RUNS / "logs" / "grpo.jsonl"))
style_log = []


def reward_fn(items, texts):
    r, s, a = combined_reward(judge, [it["q"] for it in items], texts, args.w_style, 1 - args.w_style)
    style_log.append((sum(s) / len(s), sum(a) / len(a)))
    return r


for step in range(1, args.steps + 1):
    batch = random.sample(train, args.prompts_per_step)
    items = [{"ids": prompt_ids(tok, ex["instruction"]), "q": ex["instruction"]} for ex in batch]
    stats = grpo_step(model, tok, opt, items, reward_fn, G=args.G, max_new_tokens=args.max_new_tokens,
                      temperature=1.0, beta=args.beta, micro_bs=args.micro_bs)
    style, answer = style_log[-1]
    log.log(step, style=style, answer=answer, **stats)

save_merged(model, tok, args.out)
