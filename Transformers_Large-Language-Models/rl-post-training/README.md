# RL post-training: two small projects

My two runnable post-training pipelines share one file of core code ([rlkit.py](rlkit.py)):

| Project | What I study | Base model | Reward |
|---|---|---|---|
| [rlaif_yoda/](rlaif_yoda/) | My LLM-as-a-judge from `2_LLM_Finetuning.ipynb` turned into a reward. Offline RLAIF (judge-labelled pairs → DPO) and online RLAIF (judge in the loop → GRPO) | LiquidAI/LFM2-350M | Two-rubric LLM judge |
| [rlhf_hh/](rlhf_hh/) | Classic RLHF: SFT → reward model from human preferences → RL. The same RL loop also runs on a verifiable reward | Qwen/Qwen2.5-0.5B | Bradley-Terry reward model or a GSM8K answer checker |

I kept the training loops in plain PyTorch rather than TRL so I can explain every line: token log-probs, group advantages, the clipped loss, the k3 KL estimator and the DPO loss all live in `rlkit.py`.

---

## How I run it

**Colab (free T4 GPU):** Runtime → Change runtime type → T4 GPU. Then:

```bash
git clone https://github.com/somilsin/Machine-Learning.git
cd Machine-Learning/Transformers_Large-Language-Models/rl-post-training
pip install -r requirements.txt
pip install mitdeeplearning --no-deps --no-build-isolation
```

The notebook [3_RL_Post_Training.ipynb](../3_RL_Post_Training.ipynb) runs every step below in Colab with my notes and plots.

**Local:** Python 3.11 with a CUDA build of PyTorch, then the same commands.

Every script takes `--help`. Outputs go to `runs/` inside each project (ignored by git). Training logs are JSON lines in `runs/logs/`.

---

## Project 1 · From LLM-as-a-judge to RLAIF

### How my earlier notebook maps onto RLAIF

| In `2_LLM_Finetuning.ipynb` | In RLAIF |
|---|---|
| The judge's system prompt with a 0-10 Yoda score | The **constitution**: a written rubric that replaces human labels |
| `LLMJudgeEvaluator.score()` returning 0-1 | The **reward model** r(x, y) |
| Plain Dolly answers vs Yoda rewrites as controls | **Reward model validation**: the judge must separate known-bad from known-good before I train on it |
| My note that the judge can reward superficial phrasing | **Reward hacking**: handled here with a second rubric that scores whether the question was answered |
| Scoring generated samples after training | Scoring samples **during** training and updating the policy on them |

I made two changes so the judge works as a reward:

1. **Expected-digit scoring.** The judge reads the probability of each digit 0-9 at the first answer position and returns the expected value. One forward pass, smooth rewards and never a parse failure. My earlier notebook asked for JSON and parsed it, which can fail and only gives 11 possible values.
2. **Two rubrics.** `reward = 0.7 × style + 0.3 × answers-the-question`. A style-only reward is easy to hack: speak like Yoda and ignore the question.

### Steps

I run these from `rlaif_yoda/`.

**Step 0 · SFT (my earlier LoRA run, saved this time)**
```bash
python 0_sft.py --steps 300
```
Same data and split (seed 7) as my earlier notebook. Loss only on answer tokens. I keep only answers that fit in 256 tokens so every target ends with the end-of-turn token. The merged model goes to `runs/yoda_sft`. I expect `loss` to fall from about 3 toward 2 and the sample answer to turn Yoda-like by step 100.

**Step 1 · Check the judge before trusting it**
```bash
python 4_eval.py --models none --n 40
```
This scores only the two controls. The last line prints the judge check: I need a gap of at least 0.15 between Yoda rewrites and plain answers on style. With a smaller gap I switch to a stronger judge (`--judge_model Qwen/Qwen2.5-3B-Instruct` on a T4 or `--judge openrouter` with `OPENROUTER_API_KEY` set; that one costs money per call).

**Step 2a · Offline RLAIF: judge-labelled pairs → DPO**
```bash
python 1_build_prefs.py --prompts 300 --k 4
python 2_dpo.py --epochs 2
```
For each prompt the SFT model writes 4 answers. The judge scores them and the best and worst become a preference pair when they differ by at least 0.1. DPO then trains on those pairs with the SFT model as the reference (the same weights with the LoRA switched off). I watch `reward_acc` rise above 0.5 and `margin` grow. I also check `chosen_logp`: when it falls along with `rejected_logp` that is the known DPO failure mode.

**Step 2b · Online RLAIF: judge in the loop → GRPO**
```bash
python 3_grpo.py --steps 150
```
Each step takes 4 prompts × 4 samples. The judge scores every sample, advantages are computed within each group of 4 and one clipped policy-gradient update is applied with a KL penalty to the SFT model.

**Step 3 · Compare every stage on the same held-out prompts**
```bash
python 4_eval.py --n 40
```
This prints style, answer and length for the base, SFT, DPO and GRPO models next to both controls. I report means with their spread.

### Local smoke test (RTX 3050 Ti, 4 GB)

- The pipeline ran end to end: SFT for 100 steps took about 2 minutes, 50 judge-labelled prompts took about 4 minutes and GRPO took about 22 seconds a step.
- **The 0.5B judge failed the judge check.** It scored plain answers 0.58 and Yoda rewrites 0.62 on style: almost no separation. Qwen2.5-1.5B-Instruct scored them 0.26 and 0.46, which passes. The default is now 1.5B. The lesson: validate the reward model against controls before training on it.
- The 1.5B judge does not fit next to the policy on a 4 GB card during GRPO. On that card the 0.5B judge (`--judge_model Qwen/Qwen2.5-0.5B-Instruct`) is only good for plumbing tests.
- `kl` sat near 0.0006 from the first step even though the LoRA starts as a no-op. That is numerical noise between the bf16 reference pass and the mixed-precision LoRA pass: with fp32 training (as on a T4) it starts at exactly 0. It is the same kind of issue as the log-prob mismatch between vLLM and a trainer in production RL.

---

## Project 2 · Classic RLHF on human preferences (plus a verifiable reward)

I run these from `rlhf_hh/`. The defaults suit a T4. On a 4 GB card I use the small settings listed in the smoke test notes.

**Step 1 · SFT on human-preferred replies**
```bash
python 1_sft.py --steps 600
```
Qwen2.5-0.5B (base) learns the chat format from the `chosen` replies in Anthropic/hh-rlhf (helpful-base). The loss is only on the final assistant turn. Saved to `runs/hh_sft`.

**Step 2 · Reward model (Bradley-Terry)**
```bash
python 2_reward_model.py --n 4000
```
The model starts from the SFT weights with a fresh scalar head and trains on `-log σ(r_chosen − r_rejected)`. It prints held-out pairwise accuracy before and after training. Human preference data is noisy: I expect well under 100%.

**Step 3a · RLHF: GRPO against the reward model**
```bash
python 3_grpo.py --reward rm --steps 200
```
I want `reward` to rise and `kl` to grow slowly. If `len` climbs while reward rises the policy is probably exploiting a length bias in the reward model. Then I rerun with `--len_penalty 0.05` and compare.

**Step 3b · Same loop, verifiable reward (GSM8K)**
```bash
python 3_grpo.py --reward gsm8k --steps 200
```
Policy: Qwen2.5-0.5B-Instruct. Reward: 1.0 for the right final number plus 0.1 for using the requested `#### <number>` format. `flat_groups` is the share of prompts where all 8 samples scored the same and gave no gradient. DAPO's dynamic sampling exists to fix exactly this.

**Step 4 · Evaluate**
```bash
python 4_eval.py --mode rm --n 100        # RM score, win rate vs SFT, length
python 4_eval.py --mode gsm8k --n 200     # greedy accuracy and format use, before vs after
```

### Local smoke test

- All four stages ran on the 4 GB card with tiny settings: `1_sft.py --n 400 --steps 120`, `2_reward_model.py --n 120 --n_test 60`, `3_grpo.py --steps 4 --G 4 --prompts_per_step 2 --micro_bs 2`.
- After the fix below the GSM8K reward varied within groups and gradients flowed. Accuracy over 20 questions after 4 steps (40% before, 30% after) is noise at that size. Real comparisons need 200+ questions after 200 steps.
- **Bug 1:** SFT originally cut long answers mid-way, which removed the end-of-turn token. The RL policy then never stopped (every rollout hit the token cap and one sample trailed off into “5 10 15 20…”). The fix keeps only answers that fit whole.
- **Bug 2:** the GSM8K reward was 0 for every sample. Qwen2.5-Instruct writes `\boxed{}` and ignored the `####` instruction. Every group was flat and the gradient was exactly zero. Answer extraction now accepts `####`, `\boxed{}` or the last number. The format bonus stays tied to `####`.

Both are reward-design lessons I want to keep in mind for every RL run.

---

## How I read the logs

| Field | Healthy | Worry when |
|---|---|---|
| `reward` | rises then flattens | rises while held-out eval falls (reward hacking) |
| `kl` | grows slowly | jumps suddenly (lower `--lr` or raise `--beta`) |
| `len` | stable or slowly rising | climbs fast with reward (length hacking) |
| `truncated` | low | high (raise `--max_new_tokens` or fix SFT stopping) |
| `flat_groups` | below 0.5 | near 1.0 (task too easy or too hard; no gradient) |
| `grad_norm` | steady | spikes before a collapse |
| `loss` | near 0 in value | expected: advantages sum to 0 within a group and the value says little; the gradient is what matters |

---

## Notes on my earlier notebook run

From the saved outputs of `2_LLM_Finetuning.ipynb`:

- It was a smaller local run: 20 LoRA steps per style (leprechaun then Yoda) with a 256-token context. LoRA r = 8 trained 245,760 of 354,729,728 parameters (0.07%).
- The judge stage was wired but not executed: no OpenRouter key was set and the notebook prints “No external judge scores were produced”.
- The only post-training number there is the Yoda test loss of 3.97. After 20 steps the samples are not yet in Yoda style.

The local judge in this folder is free to run. My next step is a full-length rerun with real judge scores.
