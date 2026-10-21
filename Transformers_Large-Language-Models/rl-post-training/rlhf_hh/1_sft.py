"""Stage 1 · SFT: teach a base model the chat format on human-preferred replies.

Loss only on the final assistant turn of each hh-rlhf "chosen" conversation.
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import Logger, add_lora, answer_ids, load_model, pad_batch, prompt_ids, sample, save_merged, seed_all, token_logprobs  # noqa: E402
from common import RUNS, hh_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max_prompt", type=int, default=384)
ap.add_argument("--max_answer", type=int, default=192)
ap.add_argument("--out", default=str(RUNS / "hh_sft"))
args = ap.parse_args()

seed_all()
data = hh_pairs("train", args.n)
model, tok = load_model(args.model, for_training=True)
model = add_lora(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
log = Logger(str(RUNS / "logs" / "sft.jsonl"))

# Keep only answers that fit whole: a cut answer loses its end-of-turn token
# and teaches the model never to stop.
examples = []
for ex in data:
    p, c = prompt_ids(tok, ex["prompt"]), answer_ids(tok, ex["prompt"], ex["chosen"])
    if len(p) <= args.max_prompt and len(c) <= args.max_answer:
        examples.append((p, c))
print(f"{len(examples)} of {len(data)} conversations fit")

for step in range(1, args.steps + 1):
    batch = random.sample(examples, args.bs)
    P, C = [b[0] for b in batch], [b[1] for b in batch]
    ids, attn, mask = pad_batch(P, C, tok.pad_token_id)
    loss = -(token_logprobs(model, ids, attn) * mask).sum() / mask.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    if step % 20 == 0:
        log.log(step, loss=loss.item())
    if step % 200 == 0 or step == args.steps:
        conv = data[0]["prompt"]
        out = sample(model, tok, prompt_ids(tok, conv), 1, max_new_tokens=80, temperature=0.7)[0]
        print("  USER:", conv[-1]["content"][:120], "\n  MODEL:", tok.decode(out, skip_special_tokens=True))

save_merged(model, tok, args.out)
