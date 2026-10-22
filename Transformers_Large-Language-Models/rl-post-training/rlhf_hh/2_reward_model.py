"""Stage 2 · my reward model from human preferences (Bradley-Terry).

I start from the SFT model with a fresh scalar head. Loss:
    -log sigmoid(r(chosen) - r(rejected))
I report held-out pairwise accuracy. I expect well under 100%: human
preferences are noisy and hh-rlhf pairs often differ only slightly.
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import DEVICE, TRAIN_DTYPE, Logger, add_lora, save_merged, seed_all  # noqa: E402
from common import RUNS, hh_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(RUNS / "hh_sft"))
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--n_test", type=int, default=400)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--out", default=str(RUNS / "hh_rm"))
args = ap.parse_args()

seed_all()
train, test = hh_pairs("train", args.n), hh_pairs("test", args.n_test)
tok = AutoTokenizer.from_pretrained(args.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1, dtype=TRAIN_DTYPE).to(DEVICE)
model.config.pad_token_id = tok.pad_token_id
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = add_lora(model, task_type="SEQ_CLS")          # note to self: the new score head is trained in full
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
log = Logger(str(RUNS / "logs" / "rm.jsonl"))


def rewards(batch, key):
    texts = [tok.apply_chat_template(ex["prompt"] + [{"role": "assistant", "content": ex[key]}], tokenize=False)
             for ex in batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len,
              add_special_tokens=False).to(DEVICE)
    return model(**enc).logits.squeeze(-1).float()


@torch.no_grad()
def accuracy(pairs):
    model.eval()
    hits = 0
    for i in range(0, len(pairs), args.bs):
        b = pairs[i:i + args.bs]
        hits += (rewards(b, "chosen") > rewards(b, "rejected")).sum().item()
    model.train()
    return hits / len(pairs)


print(f"held-out accuracy before training: {accuracy(test):.3f}")
step = 0
for epoch in range(args.epochs):
    random.shuffle(train)
    for i in range(0, len(train) - args.bs + 1, args.bs):
        b = train[i:i + args.bs]
        rc, rr = rewards(b, "chosen"), rewards(b, "rejected")
        loss = -F.logsigmoid(rc - rr).mean()           # Bradley-Terry: chosen should outscore rejected
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        step += 1
        if step % 25 == 0:
            log.log(step, loss=loss.item(), batch_acc=(rc > rr).float().mean().item(),
                    margin=(rc - rr).mean().item())
print(f"held-out accuracy after training: {accuracy(test):.3f}")
save_merged(model, tok, args.out)
