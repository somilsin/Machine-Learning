"""Stage 3 · RL. The same GRPO loop with two reward sources.

--reward rm     classic RLHF: my reward model trained on human preferences
--reward gsm8k  verifiable reward: a rule checks the final number (R1-Zero style)
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import DEVICE, DTYPE, Logger, add_lora, grpo_step, load_model, prompt_ids, save_merged, seed_all  # noqa: E402
from common import GSM8K_INSTRUCTION, RUNS, gsm8k_items, gsm8k_reward, hh_pairs, rm_scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reward", default="rm", choices=["rm", "gsm8k"])
ap.add_argument("--policy", default=None, help="default: runs/hh_sft for rm, Qwen2.5-0.5B-Instruct for gsm8k")
ap.add_argument("--rm", default=str(RUNS / "hh_rm"))
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--prompts_per_step", type=int, default=4)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--len_penalty", type=float, default=0.0, help="reward -= len_penalty * tokens/100 (rm only)")
ap.add_argument("--max_new_tokens", type=int, default=None)
ap.add_argument("--micro_bs", type=int, default=4)
ap.add_argument("--out", default=None)
args = ap.parse_args()

seed_all()
if args.reward == "rm":
    policy = args.policy or str(RUNS / "hh_sft")
    max_new = args.max_new_tokens or 128
    pool = [{"msgs": ex["prompt"]} for ex in hh_pairs("train", 3000, seed=11)]
    rm_tok = AutoTokenizer.from_pretrained(args.rm)
    rm = AutoModelForSequenceClassification.from_pretrained(args.rm, dtype=DTYPE).to(DEVICE).eval()

    def reward_fn(items, texts):
        r = rm_scores(rm, rm_tok, [it["msgs"] for it in items], texts)
        return [ri - args.len_penalty * len(rm_tok.encode(t)) / 100 for ri, t in zip(r, texts)]
else:
    policy = args.policy or "Qwen/Qwen2.5-0.5B-Instruct"
    max_new = args.max_new_tokens or 300
    pool = [{"msgs": [{"role": "user", "content": it["q"] + GSM8K_INSTRUCTION}], "gold": it["gold"]}
            for it in gsm8k_items("train")]

    def reward_fn(items, texts):
        return [gsm8k_reward(t, it["gold"]) for it, t in zip(items, texts)]

out_dir = args.out or str(RUNS / f"grpo_{args.reward}")
model, tok = load_model(policy, for_training=True)
model = add_lora(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
log = Logger(str(RUNS / "logs" / f"grpo_{args.reward}.jsonl"))

for step in range(1, args.steps + 1):
    items = [dict(it, ids=prompt_ids(tok, it["msgs"])) for it in random.sample(pool, args.prompts_per_step)]
    stats = grpo_step(model, tok, opt, items, reward_fn, G=args.G, max_new_tokens=max_new,
                      temperature=1.0, beta=args.beta, micro_bs=args.micro_bs)
    log.log(step, **stats)
    if step % 25 == 0:
        print("  best sample:", stats["sample"].replace("\n", " ")[:200])

save_merged(model, tok, out_dir)
