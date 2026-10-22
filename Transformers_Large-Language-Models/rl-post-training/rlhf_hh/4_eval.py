"""Stage 4 · did RL actually help?

--mode rm     SFT vs RLHF policy on held-out hh-rlhf prompts: reward-model score,
              win rate and length. A higher RM score with much longer answers
              is the classic sign of reward hacking. I read the samples.
--mode gsm8k  greedy accuracy on held-out GSM8K questions for each model.
"""
import argparse
import gc
import statistics as st
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import DEVICE, DTYPE, load_model, prompt_ids, sample, seed_all  # noqa: E402
from common import GSM8K_INSTRUCTION, RUNS, gsm8k_items, hh_pairs, is_correct, rm_scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="rm", choices=["rm", "gsm8k"])
ap.add_argument("--models", nargs="+", default=None)
ap.add_argument("--rm", default=str(RUNS / "hh_rm"))
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()
seed_all()


def generate_all(path, convs, max_new, temperature):
    model, tok = load_model(path)
    outs = [tok.decode(sample(model, tok, prompt_ids(tok, c), 1, max_new, temperature)[0],
                       skip_special_tokens=True).strip() for c in convs]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs


if args.mode == "rm":
    models = args.models or [str(RUNS / "hh_sft"), str(RUNS / "grpo_rm")]
    convs = [ex["prompt"] for ex in hh_pairs("test", args.n, seed=3)]
    outs = {m: generate_all(m, convs, 128, 0.7) for m in models}
    rm_tok = AutoTokenizer.from_pretrained(args.rm)
    rm = AutoModelForSequenceClassification.from_pretrained(args.rm, dtype=DTYPE).to(DEVICE).eval()
    scores = {m: rm_scores(rm, rm_tok, convs, o) for m, o in outs.items()}
    base = models[0]
    for m in models:
        win = st.mean(float(a > b) for a, b in zip(scores[m], scores[base]))
        words = st.mean(len(t.split()) for t in outs[m])
        print(f"{Path(m).name:24} RM {st.mean(scores[m]):+.3f}   win vs {Path(base).name}: {win:.0%}   words {words:.0f}")
    print("\nexample prompt:", convs[0][-1]["content"][:150])
    for m in models:
        print(f"[{Path(m).name}]", outs[m][0][:300].replace("\n", " "))
else:
    models = args.models or ["Qwen/Qwen2.5-0.5B-Instruct", str(RUNS / "grpo_gsm8k")]
    items = gsm8k_items("test", args.n)
    convs = [[{"role": "user", "content": it["q"] + GSM8K_INSTRUCTION}] for it in items]
    for m in models:
        outs = generate_all(m, convs, 300, 0)
        acc = st.mean(float(is_correct(o, it["gold"])) for o, it in zip(outs, items))
        fmt = st.mean(float("####" in o) for o in outs)
        print(f"{Path(m).name:28} accuracy {acc:.1%}   used the #### format {fmt:.0%}")
