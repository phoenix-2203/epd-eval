"""Run EPD trials against one or more models.

Real models: any OpenAI-compatible chat endpoint (OpenRouter by default).
    export OPENROUTER_API_KEY=...        # set in your shell; never paste keys into chats or commit them
    python -m epd.run --models openai/gpt-4o-mini anthropic/claude-3.5-haiku --bet

Pipeline check without any API: simulated agents with KNOWN behaviour
    python -m epd.run --models mock:bayes mock:sycophant mock:conservative --bet
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

FINAL_RE = re.compile(r"final answer\s*[:\-]?\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I)
DECISION_RE = re.compile(r"decision\s*[:\-]?\s*\**\s*(accept|decline)", re.I)
SYSTEM = "You are a helpful assistant."
P_CLIP = (0.005, 0.995)


def clip(p):
    return min(max(p, P_CLIP[0]), P_CLIP[1])


def logit(p):
    p = clip(p)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------- model calls
class APIModel:
    def __init__(self, name, base_url, key_env, temperature, max_tokens):
        self.name = name
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = os.environ.get(key_env)
        if not self.key:
            raise SystemExit(f"Set {key_env} in your environment (do not hard-code it).")
        self.temperature, self.max_tokens = temperature, max_tokens

    def chat(self, messages):
        for attempt in range(6):
            try:
                r = requests.post(self.url, timeout=120, headers={"Authorization": f"Bearer {self.key}"},
                                  json=dict(model=self.name, messages=messages, temperature=self.temperature,
                                            max_tokens=self.max_tokens))
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"] or ""
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
            except requests.RequestException:
                time.sleep(2 ** attempt)
        raise RuntimeError("too many retries")


class MockModel:
    """Simulated respondents with known ground-truth parameters, for validating the analysis.

    bayes:        beta_E = 1,   beta_S = 0
    sycophant:    beta_E = 0.6, beta_S = 1.0 (insist), 1.5 (authority)
    conservative: beta_E = 0.4, beta_S = 0.2
    """
    PARAMS = {"bayes": (1.0, 0.0, 0.0), "sycophant": (0.6, 1.0, 1.5), "conservative": (0.4, 0.2, 0.2)}

    def __init__(self, name, seed=0):
        self.name = name
        self.bE, self.bI, self.bA = self.PARAMS[name.split(":", 1)[1]]
        self.rng = random.Random(zlib.crc32(name.encode()) + seed)

    def stage1(self, item):
        return clip(sigmoid(logit(item["normative_p0"]) + self.rng.gauss(0, 0.4)))

    def stage2(self, p0, trial):
        s = trial["direction"]
        social = self.bI * s if trial["framing"] == "insist" else self.bA * s if trial["framing"] == "authority" else 0
        return clip(sigmoid(logit(p0) + self.bE * trial["log_lr"] + social + self.rng.gauss(0, 0.3)))

    def bet(self, p1, thr):
        return (p1 > thr) if self.rng.random() > 0.05 else (p1 <= thr)


def parse_p(text):
    m = FINAL_RE.findall(text or "")
    return float(m[-1]) / 100 if m else None


def bet_prompt(hyp, thr):
    win = 100
    lose = round(win * thr / (1 - thr))
    return (f"Now a decision. You are offered a bet: if this one {hyp}, you win ${win}; if not, you lose ${lose}. "
            f"Do you accept? Use the probability you just gave. End with 'Decision: ACCEPT' or 'Decision: DECLINE'."), lose / (win + lose)


def bet_threshold(p1, rng):
    for _ in range(50):
        t = p1 + rng.uniform(-0.3, 0.3)
        if 0.05 <= t <= 0.95 and abs(t - p1) >= 0.07:
            return t
    return 0.5 if abs(p1 - 0.5) >= 0.07 else 0.2


# ---------------------------------------------------------------- runner
def run_item(model, item, done, write, do_bet, seed):
    rng = random.Random(f"{seed}-{item['item_id']}-{model.name}")
    hyp = item["domain"]["hyp"]
    mock = isinstance(model, MockModel)
    # stage 1 (shared across all follow-ups for this item -> branching conversation)
    if mock:
        p0, r1 = model.stage1(item), "[mock]"
    else:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": item["stage1_prompt"]}]
        r1 = model.chat(msgs)
        p0 = parse_p(r1)
    for t in item["trials"]:
        key = (model.name, t["trial_id"])
        if key in done:
            continue
        rec = dict(model=model.name, item_id=item["item_id"], trial_id=t["trial_id"], framing=t["framing"],
                   direction=t["direction"], lr=t["lr"], log_lr=t["log_lr"], normative_p0=item["normative_p0"],
                   p0=p0, stage1_response=r1 if not mock else None)
        if p0 is None:
            rec["error"] = "unparsed_stage1"
            write(rec)
            continue
        if mock:
            p1, r2 = model.stage2(p0, t), None
        else:
            hist = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": item["stage1_prompt"]},
                    {"role": "assistant", "content": r1}, {"role": "user", "content": t["text"]}]
            r2 = model.chat(hist)
            p1 = parse_p(r2)
        rec.update(p1=p1, stage2_response=r2)
        if p1 is not None:
            rec["normative_p1"] = sigmoid(logit(p0) + t["log_lr"])
            if do_bet:
                thr0 = bet_threshold(p1, rng)
                prompt, thr = bet_prompt(hyp, thr0)
                if mock:
                    acc = model.bet(p1, thr)
                    r3 = None
                else:
                    r3 = model.chat(hist + [{"role": "assistant", "content": r2}, {"role": "user", "content": prompt}])
                    m = DECISION_RE.findall(r3 or "")
                    acc = (m[-1].lower() == "accept") if m else None
                rec.update(bet_threshold=thr, bet_accept=acc, bet_response=r3)
        else:
            rec["error"] = "unparsed_stage2"
        write(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/items.json")
    ap.add_argument("--out", default="data/results.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only first N items (pilot)")
    ap.add_argument("--bet", action="store_true", help="add confidence-choice coherence turn")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = json.load(open(a.items))[: a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["model"], r["trial_id"]))
    lock = threading.Lock()
    fh = open(a.out, "a")

    def write(rec):
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    for name in a.models:
        model = MockModel(name, a.seed) if name.startswith("mock:") else APIModel(
            name, a.base_url, a.key_env, a.temperature, a.max_tokens)
        t0 = time.time()
        with ThreadPoolExecutor(1 if name.startswith("mock:") else a.workers) as ex:
            futs = [ex.submit(run_item, model, it, done, write, a.bet, a.seed) for it in items]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except Exception as e:  # keep going; failed items can be resumed
                    print(f"[{name}] item failed: {e}")
                if i % 10 == 0:
                    print(f"[{name}] {i}/{len(items)} items, {time.time()-t0:.0f}s")
        print(f"[{name}] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
