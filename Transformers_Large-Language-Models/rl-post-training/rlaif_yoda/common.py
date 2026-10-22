"""Data and judge for my Yoda RLAIF project.

The data is the same as in 2_LLM_Finetuning.ipynb: Dolly instructions paired
with the MIT 6.S191 Yoda-style rewrites, split 80/20 with seed 7. The test
prompts match that notebook.

The judge is my LLM-as-a-judge from that notebook turned into a reward model.
I use two rubrics instead of one. A style score alone is easy to hack (say
something Yoda-like and ignore the question). The second rubric checks the answer.
"""
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlkit import DEVICE, DTYPE  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"

# I wrote this example myself (not a film quote) so the rubric carries no copied text.
YODA_EXAMPLE = ("Patience you must have, young one. Hard the path is. Walk it you will. "
                "Ask the right question first, you must. Then the answer, find it you shall.")

STYLE_RUBRIC = f"""You are an impartial judge that evaluates if text was written by Yoda.

An example piece of text from Yoda is:
{YODA_EXAMPLE}

Score how closely the new text follows Yoda's style: inverted word order, short sentences, his voice.
9 means exactly Yoda's style. 5 means mixed. 0 means not Yoda at all.
Reply with a single digit from 0 to 9 and nothing else."""

ANSWER_RUBRIC = """You are an impartial judge. You see a question and a response.
Score how well the response answers the question. Ignore the writing style completely.
9 means it fully and correctly answers the question. 5 means partly. 0 means it ignores the question or is nonsense.
Reply with a single digit from 0 to 9 and nothing else."""


def yoda_split(seed=7):
    spec = importlib.util.find_spec("mitdeeplearning")
    if spec is None:
        raise SystemExit("pip install mitdeeplearning --no-deps   (it ships the Yoda style file)")
    style_file = Path(spec.submodule_search_locations[0]) / "data" / "text_styles" / "yoda.txt"
    responses = [line.strip().replace("\\n", "\n")
                 for line in style_file.read_text(encoding="utf-8").splitlines()]
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    styled = dolly.select(range(len(responses))).map(
        lambda row, i: {"response_style": responses[i]}, with_indices=True)
    split = styled.train_test_split(test_size=0.2, seed=seed)
    return list(split["train"]), list(split["test"])


class LocalJudge:
    """Scores 0-1 by reading the judge's probability for each digit 0-9.

    note to self:

    the expected digit instead of parsing generated text gives a smooth
    reward with one forward pass and never fails to parse.
    """

    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", batch_size=8, max_chars=700):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).to(DEVICE).eval()
        self.digits = [self.tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
        self.bs = batch_size
        self.max_chars = max_chars

    @torch.no_grad()
    def _score(self, system, users):
        scores = []
        for i in range(0, len(users), self.bs):
            chats = [self.tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": u}],
                add_generation_prompt=True, tokenize=False) for u in users[i:i + self.bs]]
            enc = self.tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
            last = self.model(**enc).logits[:, -1].float()
            p = last[:, self.digits].softmax(-1)
            scores += (p @ torch.arange(10.0, device=DEVICE) / 9).tolist()
        return scores

    def style(self, texts):
        return self._score(STYLE_RUBRIC, [f"Evaluate this text: {t[:self.max_chars]}" for t in texts])

    def answers(self, questions, texts):
        return self._score(ANSWER_RUBRIC, [f"Question: {q[:self.max_chars]}\n\nResponse: {t[:self.max_chars]}"
                                           for q, t in zip(questions, texts)])


class OpenRouterJudge:
    """Same rubrics through OpenRouter (my earlier notebook used google/gemini-2.5-flash). Costs money per call."""

    def __init__(self, model_id="google/gemini-2.5-flash", max_chars=700):
        import requests
        self.requests = requests
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.model_id = model_id
        self.max_chars = max_chars

    def _one(self, system, user, tries=5):
        for k in range(tries):
            try:
                r = self.requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model_id, "max_tokens": 5, "temperature": 0,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                    timeout=60)
                text = r.json()["choices"][0]["message"]["content"]
                return int(re.search(r"\d", text).group()) / 9
            except Exception:
                time.sleep(2 ** k)
        raise RuntimeError("judge failed after retries")

    def style(self, texts):
        return [self._one(STYLE_RUBRIC, f"Evaluate this text: {t[:self.max_chars]}") for t in texts]

    def answers(self, questions, texts):
        return [self._one(ANSWER_RUBRIC, f"Question: {q[:self.max_chars]}\n\nResponse: {t[:self.max_chars]}")
                for q, t in zip(questions, texts)]


def make_judge(name, model_id=None):
    if name == "local":
        return LocalJudge(model_id or "Qwen/Qwen2.5-1.5B-Instruct")
    if name == "openrouter":
        return OpenRouterJudge(model_id or "google/gemini-2.5-flash")
    raise ValueError(name)


def combined_reward(judge, questions, texts, w_style=0.7, w_answer=0.3):
    s = judge.style(texts)
    a = judge.answers(questions, texts)
    return [w_style * si + w_answer * ai for si, ai in zip(s, a)], s, a


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
