"""Shared building blocks for both post-training projects.

Everything an interviewer may ask you to write by hand lives here:
per-token log-probs, group advantages, the GRPO loss with the k3 KL estimator
and the DPO loss. Both projects import from this file.
"""
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 needs an Ampere or newer GPU (compute capability 8+). On older cards such as
# Colab's T4 the trained model stays in fp32 (fp16 training without a loss scaler
# can underflow) and frozen models (judge, reward model, reference) run in fp16.
_BF16 = DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8
if DEVICE == "cuda":
    TRAIN_DTYPE = torch.bfloat16 if _BF16 else torch.float32
    DTYPE = torch.bfloat16 if _BF16 else torch.float16
else:
    TRAIN_DTYPE = DTYPE = torch.float32
if os.environ.get("RLKIT_TRAIN_DTYPE"):
    TRAIN_DTYPE = getattr(torch, os.environ["RLKIT_TRAIN_DTYPE"])


def seed_all(seed=7):
    random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- models

def load_model(path, for_training=False):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=TRAIN_DTYPE if for_training else DTYPE).to(DEVICE)
    if for_training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model, tok


def add_lora(model, r=16, alpha=32, task_type="CAUSAL_LM"):
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                     target_modules="all-linear", task_type=task_type)
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable of {total:,} ({100 * trainable / total:.2f}%)")
    return model


def save_merged(peft_model, tok, out_dir):
    """Fold the LoRA update into the base weights so the next stage loads a plain model."""
    merged = peft_model.merge_and_unload()
    merged.config.use_cache = True
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("saved", out_dir)


def eos_ids(tok):
    ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id:
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


# ---------------------------------------------------------------- chat formatting

def prompt_ids(tok, messages):
    """Token ids for a conversation that ends where the assistant should start writing."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return list(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))


def answer_ids(tok, messages, answer):
    """Token ids of one assistant turn (including its end-of-turn token) that follow prompt_ids()."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    p = prompt_ids(tok, messages)
    full = list(tok.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True))
    if full[:len(p)] != p:
        raise ValueError("chat template is not prefix-stable for this model")
    return full[len(p):]


# ---------------------------------------------------------------- sampling

@torch.no_grad()
def sample(model, tok, p_ids, n, max_new_tokens=128, temperature=1.0):
    """n sampled completions for one prompt, each cut after its first end-of-turn token."""
    was_training = model.training
    model.eval()
    x = torch.tensor([p_ids], device=DEVICE)
    out = model.generate(
        input_ids=x, attention_mask=torch.ones_like(x),
        do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
        top_p=1.0, top_k=0, max_new_tokens=max_new_tokens, num_return_sequences=n,
        eos_token_id=eos_ids(tok), pad_token_id=tok.pad_token_id, use_cache=True)
    stops = set(eos_ids(tok))
    comps = []
    for row in out[:, x.shape[1]:].tolist():
        cut = []
        for t in row:
            if t == tok.pad_token_id and t not in stops:
                break
            cut.append(t)
            if t in stops:
                break
        comps.append(cut)
    if was_training:
        model.train()
    return comps


def is_truncated(tok, completion):
    return not completion or completion[-1] not in set(eos_ids(tok))


# ---------------------------------------------------------------- log-probs

def pad_batch(prompts, completions, pad_id):
    """Right-padded prompt+completion batch.

    loss_mask[i, t] = 1 where position t predicts a completion token,
    so it lines up with token_logprobs() output (which drops the last position).
    """
    seqs = [p + c for p, c in zip(prompts, completions)]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    mask = torch.zeros((len(seqs), T - 1))
    for i, (p, s) in enumerate(zip(prompts, seqs)):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        mask[i, len(p) - 1:len(s) - 1] = 1
    return ids.to(DEVICE), attn.to(DEVICE), mask.to(DEVICE)


def token_logprobs(model, ids, attn):
    """log pi(token_t+1 | tokens_<=t) for every position: shape [B, T-1]."""
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    labels = ids[:, 1:]
    picked = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return picked - logits.logsumexp(-1)


# ---------------------------------------------------------------- losses

def group_advantages(rewards, G, scale_std=True):
    """GRPO baseline: compare each sample with the other samples for the same prompt."""
    r = rewards.view(-1, G)
    adv = r - r.mean(1, keepdim=True)
    if scale_std:
        adv = adv / (r.std(1, keepdim=True) + 1e-4)
    return adv.view(-1)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, eps=0.2, beta=0.04):
    """Clipped policy-gradient loss plus k3 KL to the reference, averaged per sequence."""
    ratio = torch.exp(logp - old_logp)
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1                       # k3 estimator: non-negative, low variance
    per_tok = (pg + beta * kl) * mask
    lengths = mask.sum(1).clamp(min=1)
    loss = (per_tok.sum(1) / lengths).mean()
    mean_kl = ((kl * mask).sum(1) / lengths).mean()
    return loss, mean_kl.detach()


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """Inputs are summed response log-probs, shape [B]."""
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -F.logsigmoid(logits).mean()
    return loss, (logits > 0).float().mean().item(), (logits / beta).mean().item()


# ---------------------------------------------------------------- one GRPO step

def grpo_step(model, tok, opt, items, reward_fn, G=4, max_new_tokens=128, temperature=1.0,
              beta=0.04, eps=0.2, micro_bs=4, scale_std=True, max_grad_norm=1.0):
    """Rollout -> reward -> group advantage -> one clipped policy-gradient update.

    items: list of dicts, each with "ids" (prompt token ids) plus whatever reward_fn needs.
    reward_fn(items_per_sample, completion_texts) -> list of floats.
    The model must be a PEFT model: disabling the adapter gives the frozen reference policy.
    """
    P, C, meta = [], [], []
    for it in items:
        for c in sample(model, tok, it["ids"], G, max_new_tokens, temperature):
            P.append(it["ids"])
            C.append(c)
            meta.append(it)
    texts = [tok.decode(c, skip_special_tokens=True).strip() for c in C]
    rewards = torch.tensor(reward_fn(meta, texts), dtype=torch.float32, device=DEVICE)
    adv = group_advantages(rewards, G, scale_std)
    flat_groups = (rewards.view(-1, G).std(1) < 1e-6).float().mean().item()

    model.train()
    opt.zero_grad(set_to_none=True)
    N = len(C)
    loss_sum, kl_sum = 0.0, 0.0
    for s in range(0, N, micro_bs):
        sl = slice(s, s + micro_bs)
        ids, attn, mask = pad_batch(P[sl], C[sl], tok.pad_token_id)
        logp = token_logprobs(model, ids, attn)
        with torch.no_grad(), model.disable_adapter():
            ref = token_logprobs(model, ids, attn)
        # One update per rollout batch: the sampling policy equals the current one:
        # old_logp = logp.detach() makes the ratio 1 in value while keeping its gradient.
        loss, kl = grpo_loss(logp, logp.detach(), ref, adv[sl].unsqueeze(1), mask, eps, beta)
        w = ids.shape[0] / N
        (loss * w).backward()
        loss_sum += loss.item() * w
        kl_sum += kl.item() * w
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm).item()
    opt.step()

    return {
        "reward": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "kl": kl_sum,
        "loss": loss_sum,
        "grad_norm": grad_norm,
        "len": sum(len(c) for c in C) / N,
        "truncated": sum(is_truncated(tok, c) for c in C) / N,
        "flat_groups": flat_groups,
        "sample": texts[int(rewards.argmax())][:200],
    }


# ---------------------------------------------------------------- logging

class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")   # a rerun replaces the old log
        self.t0 = time.time()

    def log(self, step, **kv):
        kv = {"step": step, "min": round((time.time() - self.t0) / 60, 2), **kv}
        self.f.write(json.dumps(kv) + "\n")
        self.f.flush()
        short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kv.items() if k != "sample"}
        print(short)
