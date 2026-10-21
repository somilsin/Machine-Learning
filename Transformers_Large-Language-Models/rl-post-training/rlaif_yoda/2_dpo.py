"""Stage 2a · offline RLAIF with DPO on the judge-labelled pairs.

Policy = SFT model + a fresh LoRA. Reference = the same model with the LoRA
switched off: no second copy of the weights is needed.
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import Logger, add_lora, answer_ids, dpo_loss, load_model, pad_batch, prompt_ids, save_merged, seed_all, token_logprobs  # noqa: E402
from common import RUNS, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=str(RUNS / "yoda_sft"))
ap.add_argument("--prefs", default=str(RUNS / "prefs.jsonl"))
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--out", default=str(RUNS / "yoda_dpo"))
args = ap.parse_args()

seed_all()
pairs = read_jsonl(args.prefs)
model, tok = load_model(args.policy, for_training=True)
model = add_lora(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
log = Logger(str(RUNS / "logs" / "dpo.jsonl"))


def summed_logps(P, C):
    ids, attn, mask = pad_batch(P, C, tok.pad_token_id)
    pol = (token_logprobs(model, ids, attn) * mask).sum(1)
    with torch.no_grad(), model.disable_adapter():
        ref = (token_logprobs(model, ids, attn) * mask).sum(1)
    return pol, ref


step = 0
for epoch in range(args.epochs):
    random.shuffle(pairs)
    for s in range(0, len(pairs) - args.bs + 1, args.bs):
        b = pairs[s:s + args.bs]
        P = [prompt_ids(tok, r["prompt"]) for r in b]
        W = [answer_ids(tok, r["prompt"], r["chosen"])[:args.max_len] for r in b]
        L = [answer_ids(tok, r["prompt"], r["rejected"])[:args.max_len] for r in b]
        pol, ref = summed_logps(P + P, W + L)             # chosen and rejected in one pass
        n = len(b)
        loss, acc, margin = dpo_loss(pol[:n], pol[n:], ref[:n], ref[n:], args.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        step += 1
        if step % 10 == 0:
            log.log(step, epoch=epoch, loss=loss.item(), reward_acc=acc, margin=margin,
                    chosen_logp=pol[:n].mean().item(), rejected_logp=pol[n:].mean().item())

save_merged(model, tok, args.out)
