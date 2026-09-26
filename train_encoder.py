"""Optional: fine-tune a multilingual sentence encoder with a contrastive loss.
Records of the same business are pulled together, other records in the batch pushed apart
(MultipleNegativesRankingLoss = in-batch-negatives InfoNCE). Best run on a GPU.

    python train_encoder.py                                   # small model, fine on CPU for tests
    python train_encoder.py --model intfloat/multilingual-e5-base --epochs 3   # on GPU

Only the 'enc' fold of Source 1 entities is used, so the validation fold stays unseen
and the matcher can be trained on a separate fold (see train_matcher.py).
"""
import argparse
import random
from collections import defaultdict
from itertools import combinations

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from common import (ENCODER_DIR, ENCODER_PREFIX, SEED, assign_folds, load_ground_truth,
                    load_split)


def corrupt(text, rng):
    """Cheap synthetic noise: gives extra positives, including for S1 singletons."""
    toks = text.split()
    if len(toks) < 2:
        return text
    op = rng.random()
    if op < 0.35 and len(toks) > 3:                         # drop a word
        del toks[rng.randrange(len(toks))]
    elif op < 0.7:                                          # typo: swap two letters
        t = rng.randrange(len(toks))
        w = toks[t]
        if len(w) > 3:
            p = rng.randrange(len(w) - 1)
            toks[t] = w[:p] + w[p + 1] + w[p] + w[p + 2:]
    else:                                                   # swap two adjacent words
        t = rng.randrange(len(toks) - 1)
        toks[t], toks[t + 1] = toks[t + 1], toks[t]
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--augment", type=int, default=1, help="synthetic pairs per S1 record")
    args = ap.parse_args()
    rng = random.Random(SEED)

    s1, pool = load_split("train")
    text = {}
    for df in (s1, pool):
        for eid, n, a, c in zip(df["entity_id"], df["business_name"],
                                df["business_address"], df["country"]):
            text[eid] = f"{n}, {a}, {c}"

    folds = assign_folds(s1["entity_id"])
    groups = defaultdict(list)
    for a, b in load_ground_truth():
        if folds.get(a) == "enc" and b in text:
            groups[a].append(b)

    pairs = []
    for a, ms in groups.items():
        pairs += [(text[a], text[m]) for m in ms]                  # S1 <-> its matches
        pairs += [(text[x], text[y]) for x, y in combinations(ms, 2)]  # matches with each other
    for a in (k for k, v in folds.items() if v == "enc"):
        pairs += [(text[a], corrupt(text[a], rng)) for _ in range(args.augment)]
    print(f"training pairs: {len(pairs):,} from {len(groups):,} matched S1 entities")

    examples = [InputExample(texts=[ENCODER_PREFIX + x, ENCODER_PREFIX + y]) for x, y in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch, drop_last=True)
    model = SentenceTransformer(args.model)
    model.max_seq_length = 96
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=int(0.1 * len(loader) * args.epochs), show_progress_bar=True)
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(ENCODER_DIR))
    print(f"saved encoder to {ENCODER_DIR}")


if __name__ == "__main__":
    main()
