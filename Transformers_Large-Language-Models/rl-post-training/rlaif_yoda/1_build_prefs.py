"""Stage 1 · RLAIF preference data: the judge labels pairs instead of a human.

For each training instruction, sample K answers from the SFT model, score
each with the two-rubric judge and keep (best, worst) as a preference pair
when the judge separates them clearly. This is the Constitutional AI / RLAIF
move: AI feedback replaces human preference labels.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import load_model, prompt_ids, sample, seed_all  # noqa: E402
from common import RUNS, combined_reward, make_judge, write_jsonl, yoda_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=str(RUNS / "yoda_sft"))
ap.add_argument("--judge", default="local", choices=["local", "openrouter"])
ap.add_argument("--judge_model", default=None)
ap.add_argument("--prompts", type=int, default=300)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--min_gap", type=float, default=0.1)
ap.add_argument("--max_new_tokens", type=int, default=96)
ap.add_argument("--out", default=str(RUNS / "prefs.jsonl"))
args = ap.parse_args()

seed_all()
train, _ = yoda_split()
random.shuffle(train)
model, tok = load_model(args.policy)
judge = make_judge(args.judge, args.judge_model)

rows, gaps = [], []
for n, ex in enumerate(train[:args.prompts], 1):
    q = ex["instruction"]
    comps = sample(model, tok, prompt_ids(tok, q), args.k, args.max_new_tokens, temperature=0.9)
    texts = [tok.decode(c, skip_special_tokens=True).strip() for c in comps]
    reward, style, answer = combined_reward(judge, [q] * args.k, texts)
    best = max(range(args.k), key=lambda i: reward[i])
    worst = min(range(args.k), key=lambda i: reward[i])
    gap = reward[best] - reward[worst]
    gaps.append(gap)
    if gap >= args.min_gap:
        rows.append({"prompt": q, "chosen": texts[best], "rejected": texts[worst],
                     "chosen_score": reward[best], "rejected_score": reward[worst],
                     "chosen_style": style[best], "chosen_answer": answer[best]})
    if n % 25 == 0:
        print(f"{n}/{args.prompts} prompts · {len(rows)} pairs kept · mean gap {sum(gaps) / len(gaps):.3f}")

write_jsonl(Path(args.out), rows)
print(f"wrote {len(rows)} pairs to {args.out}")
