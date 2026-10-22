"""Stage 3 · I score every stage on the same held-out prompts with the same judge.

Controls as in my earlier notebook: Dolly's plain answers are the negative
control and the course's Yoda rewrites are the positive control. A model's
score only means something relative to those two rows.
"""
import argparse
import gc
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import load_model, prompt_ids, sample, seed_all  # noqa: E402
from common import RUNS, make_judge, yoda_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["LiquidAI/LFM2-350M", str(RUNS / "yoda_sft"),
                                                str(RUNS / "yoda_dpo"), str(RUNS / "yoda_grpo")])
ap.add_argument("--judge", default="local", choices=["local", "openrouter"])
ap.add_argument("--judge_model", default=None)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max_new_tokens", type=int, default=96)
args = ap.parse_args()

seed_all()
_, test = yoda_split()
test = test[:args.n]
questions = [ex["instruction"] for ex in test]
judge = make_judge(args.judge, args.judge_model)


def score(name, texts):
    s, a = judge.style(texts), judge.answers(questions, texts)
    return {"name": name, "style": st.mean(s), "style_sd": st.pstdev(s), "answer": st.mean(a),
            "answer_sd": st.pstdev(a), "words": st.mean(len(t.split()) for t in texts), "example": texts[0][:160]}


rows = [score("control: plain Dolly answers", [ex["response"] for ex in test]),
        score("control: Yoda rewrites", [ex["response_style"] for ex in test])]
for path in args.models:
    if not Path(path).exists() and "/" not in path:
        print("skip (not found):", path)
        continue
    try:
        model, tok = load_model(path)
    except OSError:
        print("skip (not found):", path)
        continue
    texts = [tok.decode(sample(model, tok, prompt_ids(tok, q), 1, args.max_new_tokens, 0.7)[0],
                        skip_special_tokens=True).strip() for q in questions]
    rows.append(score(Path(path).name, texts))
    del model
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'model':32} {'style':>13} {'answer':>13} {'words':>6}")
for r in rows:
    print(f"{r['name']:32} {r['style']:.2f} ± {r['style_sd']:.2f}   {r['answer']:.2f} ± {r['answer_sd']:.2f}   {r['words']:5.0f}")
gap = rows[1]["style"] - rows[0]["style"]
print(f"\njudge check: Yoda rewrites minus plain answers = {gap:+.2f} on style", end=" ")
print("(usable)" if gap >= 0.15 else "(TOO SMALL: use a stronger --judge_model before trusting any number above)")
(RUNS / "eval.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nsaved", RUNS / "eval.json")
