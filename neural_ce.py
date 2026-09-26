"""NEURAL MATCHER: a multilingual transformer cross-encoder (Ditto-style).

    python -u neural_ce.py                  # train (if needed) + score every filtered pair
    python -u neural_ce.py --redo           # retrain and rescore

The model reads BOTH raw records together, e.g.
    "Digital Tech Pvt Ltd ; Plot No 184, Ganesh Nagar ... ; India"  [SEP]
    "డిజిటల్ టెక్ ప్రైవేట్ లిమిటెడ్ ; GANESH NAGAR PHASE-2 ... ; India"
and outputs the probability that they are the same business. Being pretrained on ~100
languages, it can relate scripts, transliterations and French address words that the
rule-based features only approximate.

  model     intfloat/multilingual-e5-small (MIT license, 118M parameters) by default
  training  pairs of the FIRST model's training entities: all kinds of positives plus the
            hardest negatives (top blocking ranks). It never sees the fresh / valA / valB /
            test entities it later scores, so its scores are honest features for stage 2.
  scoring   only the filtered candidates (first-model probability >= tau from prune.json),
            i.e. exactly the pairs stage 2 scores.
Outputs: work/models/cross_encoder/ and work/ce/{fresh,valA,valB,test_<country>}.parquet
GPU strongly recommended (about 1-2 hours on an RTX 3050); it also runs on CPU, slowly.
"""
import argparse
import contextlib
import json
import math
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import SEED, up_to_date
from finalize import macro_f05, one_to_one, rule_threshold
from preprocess import PREP_DIR
from run_pipeline import FEAT_DIR, PRED_DIR, WORK, countries_of
from run_stage2 import CE_DIR, CE_MODEL, S2_DIR, prune_tau

RAW = ["entity_id", "business_name", "business_address", "country"]


def log(m=""):
    print(f"{time.strftime('%H:%M:%S')}  {m}", flush=True)


# ---------------------------------------------------------------- text
def texts_for(split, country):
    """entity_id -> 'name ; address ; country' for S1 and pool of one country."""
    filt = [("country_key", "=", country)]
    parts = [pd.read_parquet(PREP_DIR / f"{split}_source{k}.parquet", columns=RAW, filters=filt)
             for k in (1, 2, 3)]
    df = pd.concat(parts, ignore_index=True)
    return pd.Series((df["business_name"] + " ; " + df["business_address"] + " ; " + df["country"]).to_numpy(),
                     index=df["entity_id"].to_numpy())


def pair_texts(pairs, text):
    return text.reindex(pairs["s1_id"]).to_numpy(object), text.reindex(pairs["cand_id"]).to_numpy(object)


# ---------------------------------------------------------------- batching
def length_batches(enc, idx, batch_size, shuffle, rng):
    """Group similar lengths together (much less padding = much faster)."""
    lens = np.array([len(enc["input_ids"][i]) for i in idx])
    order = idx[np.argsort(lens, kind="stable")]
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def to_device(tok, enc, rows, device):
    feats = tok.pad({k: [enc[k][i] for i in rows] for k in enc.keys()}, return_tensors="pt")
    return {k: v.to(device, non_blocking=True) for k, v in feats.items()}


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()


@torch.inference_mode()
def score(model, tok, a, b, device, batch_size, max_len, chunk=100_000):
    model.eval()
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), chunk):
        enc = tok(list(a[s:s + chunk]), list(b[s:s + chunk]), truncation="longest_first",
                  max_length=max_len)
        idx = np.arange(len(enc["input_ids"]))
        for rows in length_batches(enc, idx, batch_size, False, None):
            with autocast(device):
                logits = model(**to_device(tok, enc, rows, device)).logits.squeeze(-1)
            out[s + rows] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


# ---------------------------------------------------------------- training
def training_pairs(args):
    """Pairs of the first model's TRAINING entities: positives + hardest negatives."""
    cols = ["s1_id", "cand_id", "country", "label", "fwd_rank", "rev_rank"]
    tr = pd.read_parquet(FEAT_DIR / "train.parquet", columns=cols)
    hard = (tr["fwd_rank"] < 8) | (tr["rev_rank"] < 2)
    pos, neg = tr[tr["label"] == 1], tr[(tr["label"] == 0) & hard]
    n_pos = min(len(pos), int(args.max_train_pairs * 0.45))
    n_neg = min(len(neg), args.max_train_pairs - n_pos)
    sel = pd.concat([pos.sample(n_pos, random_state=SEED), neg.sample(n_neg, random_state=SEED)])
    log(f"training pairs: {len(sel):,} ({n_pos:,} matches, {n_neg:,} hard non-matches)")
    return sel.sample(frac=1, random_state=SEED).reset_index(drop=True)


def attach_texts(pairs, split):
    a = np.empty(len(pairs), dtype=object)
    b = np.empty(len(pairs), dtype=object)
    for c in sorted(pairs["country"].unique()):
        m = (pairs["country"] == c).to_numpy()
        a[m], b[m] = pair_texts(pairs[m], texts_for(split, c))
    return a, b


def train(args, device):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True).to(device)   # new 1-output head
    pairs = training_pairs(args)
    a, b = attach_texts(pairs, "train")
    y = pairs["label"].to_numpy(np.float32)

    # a small monitoring set from valA (hard pairs), only for the printed loss
    va = pd.read_parquet(FEAT_DIR / "valA.parquet",
                         columns=["s1_id", "cand_id", "country", "label", "fwd_rank", "rev_rank"])
    va = va[(va["fwd_rank"] < 8) | (va["rev_rank"] < 2)]
    va = va.sample(min(20_000, len(va)), random_state=SEED)
    va_a, va_b = attach_texts(va, "train")

    steps = math.ceil(len(pairs) / args.batch) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(pairs))
        for s in range(0, len(order), 50_000):
            idx = order[s:s + 50_000]
            enc = tok(list(a[idx]), list(b[idx]), truncation="longest_first", max_length=args.max_len)
            for rows in length_batches(enc, np.arange(len(idx)), args.batch, True, rng):
                with autocast(device):
                    logits = model(**to_device(tok, enc, rows, device)).logits.squeeze(-1)
                    loss = lossf(logits.float(), torch.from_numpy(y[idx[rows]]).to(device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % 500 == 0 or step == steps:
                    rate = step * args.batch / (time.time() - t0)
                    log(f"  epoch {epoch + 1} step {step:,}/{steps:,}  loss={loss.item():.4f}  "
                        f"{rate:,.0f} pairs/s  eta {(steps - step) * args.batch / max(rate, 1) / 60:.0f} min")
        p = score(model, tok, va_a, va_b, device, args.infer_batch, args.max_len)
        yv = va["label"].to_numpy()
        ll = -np.mean(yv * np.log(p + 1e-7) + (1 - yv) * np.log(1 - p + 1e-7))
        acc = np.mean((p >= 0.5) == (yv == 1))
        log(f"epoch {epoch + 1}: valA hard pairs  logloss={ll:.4f}  accuracy={acc:.4f}")
    CE_MODEL.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CE_MODEL)
    tok.save_pretrained(CE_MODEL)
    (CE_MODEL / "meta.json").write_text(json.dumps(vars(args), indent=2))
    return model, tok


# ---------------------------------------------------------------- scoring filtered pairs
def score_parts(model, tok, args, device, tau):
    CE_DIR.mkdir(parents=True, exist_ok=True)
    meta = CE_MODEL / "meta.json"
    jobs = [(part, S2_DIR / f"{part}_p1.parquet", "train") for part in ("fresh", "valA", "valB")]
    jobs += [(f"test_{c}", PRED_DIR / f"test_{c}.parquet", "test") for c in countries_of("test")]
    for name, src, split in jobs:
        dst = CE_DIR / f"{name}.parquet"
        if not args.redo and up_to_date([dst], [meta, src, WORK / "prune.json"]):
            log(f"  skip {name} (up to date)")
            continue
        t0 = time.time()
        col = "p1" if split == "train" else "prob"
        d = pq.read_table(src, columns=["s1_id", "cand_id", col] + (["country"] if split == "train" else []),
                          filters=[(col, ">=", tau)]).to_pandas()
        if split == "test":
            d["country"] = name.split("_", 1)[1]
        a, b = attach_texts(d, split)
        ce = score(model, tok, a, b, device, args.infer_batch, args.max_len)
        pd.DataFrame({"s1_id": d["s1_id"].to_numpy(object), "cand_id": d["cand_id"].to_numpy(object),
                      "ce": ce}).to_parquet(dst, index=False)
        log(f"  scored {name}: {len(d):,} pairs in {(time.time() - t0) / 60:.1f} min "
            f"({len(d) / max(time.time() - t0, 1e-9):,.0f} pairs/s)")


def report(tau):
    """How good is the cross-encoder ALONE on the official metric (valA tunes, valB reports)?"""
    truth = pd.read_parquet(FEAT_DIR / "truth.parquet")
    res = {}
    for v in ("valA", "valB"):
        p1 = pd.read_parquet(S2_DIR / f"{v}_p1.parquet", columns=["s1_id", "cand_id", "country", "p1"])
        ents = sorted(set(p1["s1_id"]) | set(truth.loc[truth["part"] == v, "s1"]))
        ce = pd.read_parquet(CE_DIR / f"{v}.parquet")
        d = p1[p1["p1"] >= tau].merge(ce, on=["s1_id", "cand_id"])
        t = truth[truth["part"] == v][["s1", "m"]]
        res[v] = {"ce": (one_to_one(d.assign(prob=d["ce"])), t, ents),
                  "first model": (one_to_one(d.assign(prob=d["p1"])), t, ents)}
    log("official metric on valB, each model ALONE (threshold tuned on valA):")
    for name in ("first model", "ce"):
        dA, tA, eA = res["valA"][name]
        dB, tB, eB = res["valB"][name]
        best = max(np.arange(0.10, 0.97, 0.02), key=lambda t: macro_f05(rule_threshold(dA, t), tA, eA))
        log(f"  {name:12} valB={macro_f05(rule_threshold(dB, best), tB, eB):.4f}  (threshold {best:.2f})")
    log("stage 2 will combine both; its validation decides how much the cross-encoder adds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--max-train-pairs", type=int, default=1_200_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--infer-batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    tau = prune_tau()
    if tau <= 0:
        raise SystemExit("no candidate filter yet (work/prune.json): run  python -u choose_prune.py  first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
                               else "  (no GPU found: this will be slow)"))
    log(f"candidate filter tau={tau}")

    if not args.redo and up_to_date([CE_MODEL / "meta.json"], [FEAT_DIR / "train.parquet"]):
        log("cross-encoder already trained: loading it")
        tok = AutoTokenizer.from_pretrained(CE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL).to(device)
    else:
        model, tok = train(args, device)
    score_parts(model, tok, args, device, tau)
    report(tau)


if __name__ == "__main__":
    main()
