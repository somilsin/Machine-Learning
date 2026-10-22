"""Stage 0 · supervised fine-tuning. My LoRA run from 2_LLM_Finetuning.ipynb, saved this time.

Same data and objective as that notebook: loss only on the answer tokens.
What I changed: AdamW instead of Lion, batches of 4 and I merge and save the
adapter so the RL stages can start from it.
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import Logger, add_lora, answer_ids, load_model, pad_batch, prompt_ids, sample, save_merged, seed_all, token_logprobs  # noqa: E402
from common import RUNS, yoda_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="LiquidAI/LFM2-350M")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--out", default=str(RUNS / "yoda_sft"))
args = ap.parse_args()

seed_all()
train, test = yoda_split()
model, tok = load_model(args.model, for_training=True)
model = add_lora(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
log = Logger(str(RUNS / "logs" / "sft.jsonl"))

# note to self: I keep only answers that fit whole. A cut answer loses its end-of-turn
# token and teaches the model never to stop. RL rollouts then all run to the token cap.
examples = []
for ex in train:
    p, c = prompt_ids(tok, ex["instruction"]), answer_ids(tok, ex["instruction"], ex["response_style"])
    if len(p) + len(c) <= args.max_len:
        examples.append((p, c))
print(f"{len(examples)} of {len(train)} training examples fit in {args.max_len} tokens")

for step in range(1, args.steps + 1):
    batch = random.sample(examples, args.bs)
    P, C = [b[0] for b in batch], [b[1] for b in batch]
    ids, attn, mask = pad_batch(P, C, tok.pad_token_id)
    logp = token_logprobs(model, ids, attn)
    loss = -(logp * mask).sum() / mask.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    if step % 10 == 0:
        log.log(step, loss=loss.item())
    if step % 100 == 0 or step == args.steps:
        q = test[0]["instruction"]
        out = sample(model, tok, prompt_ids(tok, q), 1, max_new_tokens=60, temperature=0.7)[0]
        print("  Q:", q, "\n  A:", tok.decode(out, skip_special_tokens=True))

save_merged(model, tok, args.out)
