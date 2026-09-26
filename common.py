"""Shared pieces: paths, file reading/writing, text normalization, folds, scoring."""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"          # expects dataset/train/... and dataset/test/...
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"
ENCODER_DIR = MODEL_DIR / "encoder"  # created by train_encoder.py (optional)

SEED = 42
# Fractions of the *training* Source 1 entities.
VAL_FRAC = 0.2   # held out for honest scoring, never trained on
ENC_FRAC = 0.4   # reserved for the contrastive encoder (only matters if you train one)


# ---------------------------------------------------------------- reading / writing
def read_tsv(path):
    # dtype=str keeps IDs and PIN codes as text; QUOTE_NONE because the files are unquoted
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE).fillna("")


def load_split(split):
    """Return (source1, pool) where pool = Source 2 + Source 3 stacked together."""
    d = DATA_DIR / split
    s1 = read_tsv(d / f"{split}_source1.tsv")
    s2 = read_tsv(d / f"{split}_source2.tsv")
    s3 = read_tsv(d / f"{split}_source3.tsv")
    return s1, pd.concat([s2, s3], ignore_index=True)


def load_ground_truth():
    """Set of (source1_id, matched_id) pairs."""
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    pairs = set()
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        for m in str(ids).split(","):
            if m.strip():
                pairs.add((s1_id.strip(), m.strip()))
    return pairs


def write_grouped(path, s1_ids, pairs, column):
    """One row per Source 1 id, comma-separated list (possibly empty), no quoting."""
    by_s1 = defaultdict(set)
    for a, b in pairs:
        by_s1[a].add(b)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{column}\n")
        for sid in s1_ids:
            fh.write(f"{sid}\t{','.join(sorted(by_s1.get(sid, ())))}\n")


# ---------------------------------------------------------------- normalization
# v2, based on the data report: Indian scripts are transliterated instead of shredded,
# addresses map to SHORT canonical forms (so French "St" = Saint and US "St" = Street both
# stay "st"), state names map to codes, house-number words and web junk are removed.
from anyascii import anyascii   # transliterates any script to Latin: 'డిజిటల్ టెక్' -> 'dijitl tek'

NAME_ABBR = {
    "pvt": "private", "pte": "private", "ltd": "limited", "lmtd": "limited",
    "corp": "corporation", "co": "company", "inc": "incorporated",
    "intl": "international", "mfg": "manufacturing", "svc": "service",
    "svcs": "services", "bros": "brothers", "assoc": "associates",
    "mgmt": "management", "ste": "societe", "cie": "compagnie",
}
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp",
    "plc", "lp", "pllc", "pc", "opc", "gmbh",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "selarl", "societe", "compagnie",
    # transliterated from Indian scripts (Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, ...)
    "praivet", "piraivet", "praivrr", "praibhet", "limitet", "limirrd", "limtid",
    "kmpni", "pra", "li",
}
# generic business words dropped from the *core* name only (the full name keeps them)
GENERIC = {"holdings", "holding", "group", "partners", "partner", "services", "service",
           "enterprises", "enterprise", "ventures", "associates"}
STOP = {"the", "and", "of", "a", "le", "la", "les", "l", "de", "des", "du", "d", "et",
        "www", "com", "net", "org"}

ADDR_CANON = {
    "road": "rd", "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "lane": "ln", "drive": "dr", "court": "ct",
    "highway": "hwy", "parkway": "pkwy", "square": "sq", "building": "bldg",
    "floor": "fl", "flr": "fl", "near": "nr", "opposite": "opp", "opp": "opp",
    "ngr": "nagar", "market": "mkt", "district": "dist", "distt": "dist",
    "village": "vill",
    "faubourg": "fbg", "route": "rte", "chemin": "chem", "impasse": "imp", "place": "pl",
    "rue": "r", "allee": "all", "passage": "pas", "quai": "qu", "cours": "crs", "square": "sq",
    "residence": "res", "res": "res", "lotissement": "lot", "ch": "chem", "av": "ave",
    "bangalore": "bengaluru", "bombay": "mumbai", "gurgaon": "gurugram",
    "calcutta": "kolkata", "madras": "chennai", "poona": "pune", "pondicherry": "puducherry",
}
# "Door No 184", "Plot No 184", "#184", "Unit APT 112" -> just the numbers
DROP_ADDR = {"no", "door", "plot", "flat", "house", "shop", "unit", "apt", "apartment",
             "suite", "ste", "number", "num",
             # French: numero, bis/ter suffixes and filler words that make different
             # addresses look alike ('rue de la paix' vs 'rue de gand')
             "n", "bis", "ter", "quater", "de", "du", "des", "le", "les", "null"}
STATE_CODES = {
    # India
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "ts": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la",
    "puducherry": "py", "chandigarh": "ch",
    # USA
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
# Note: some codes collide across countries (e.g. "tn", "ga"). That's fine: records are
# only ever compared within the same country.

# ordinal words -> plain numbers ('eleventh ave' and '11th ave' both become '11 ave')
_ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
        "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
        "seventeenth", "eighteenth", "nineteenth", "twentieth"]
ADDR_CANON.update({w: str(i + 1) for i, w in enumerate(_ORD)})
ADDR_CANON.update({"thirtieth": "30", "fortieth": "40", "fiftieth": "50"})
_ORD_SUFFIX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")   # 11th, 94st, 30rd -> 11, 94, 30
_LEAD_ZERO = re.compile(r"\b0+(\d)")                     # 090 -> 90

# French regions / departments: records mix them ('Nouvelle-Aquitaine' vs 'Gironde') and
# they carry almost no information once the city is present, so they are removed.
FR_AREAS = [
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire",
    "corse", "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine",
    "occitanie", "pays de la loire", "provence alpes cote d azur", "provence alpes cote dazur",
    "aisne", "nord", "oise", "pas de calais", "somme",
    "charente maritime", "charente", "correze", "creuse", "dordogne", "gironde", "landes",
    "lot et garonne", "pyrenees atlantiques", "deux sevres", "haute vienne",
    "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee",
    "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
    "val de marne", "val d oise", "val doise",
]
_FR_AREA_RE = re.compile(r"\b(" + "|".join(sorted(FR_AREAS, key=len, reverse=True)) + r")\b")

_URL = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|biz|info|co|in|fr|us|io)(?:\.[a-z]{2})?\b")
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")          # p.v.t. -> pvt
_SPACED_PIN = re.compile(r"\b(\d{3})\s(\d{3})\b")   # 560 001 -> 560001
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_CODES), key=len, reverse=True)) + r")\b")


def clean(text):
    s = re.sub(r"[°º]", " ", str(text))               # N°13 -> N 13 (anyascii would give 'Ndeg13')
    s = anyascii(s).lower()                           # any script / accents -> plain Latin
    s = re.sub(r"['`]", "", s)
    s = _URL.sub(r" \1 ", s)                          # amsholdings.com -> amsholdings
    s = _DOTTED.sub(lambda m: m.group().replace(".", ""), s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _map_tokens(s, table):
    return " ".join(table.get(t, t) for t in s.split())


_LEET = str.maketrans("013457", "oleast")   # c0nsultants, hea1th, anch0r, 5ervices


def _unleet(tok):
    """Digits inside a mostly-letter word are usually letters in disguise."""
    n_alpha = sum(ch.isalpha() for ch in tok)
    n_digit = len(tok) - n_alpha
    return tok.translate(_LEET) if n_digit and n_alpha >= 2 and n_digit <= n_alpha else tok


def norm_name(x):
    return _map_tokens(" ".join(_unleet(t) for t in clean(x).split()), NAME_ABBR)


def core_name(name_n, country_key=""):
    """Name without legal suffixes, generic words, stopwords, or the record's own country
    label: 'syndicat parents france sas' -> 'syndicat parents'."""
    drop = LEGAL | GENERIC | STOP | set(str(country_key).split())
    toks = [t for t in name_n.split() if t not in drop]
    return " ".join(toks) if toks else name_n


def _is_nonlatin(word):
    return any(ord(c) > 0x24F for c in word)          # beyond Latin + Latin Extended


def norm_addr(x):
    # A few native-script words in a mostly-English address are almost always the state
    # ("..., NEW DELHI, दिल्ली"); drop them. A fully native-script address is transliterated.
    words = str(x).split()
    flags = [_is_nonlatin(w) for w in words]
    if 0 < sum(flags) < len(words) / 2:
        x = " ".join(w for w, f in zip(words, flags) if not f)
    s = _SPACED_PIN.sub(r"\1\2", clean(x))
    s = _LEAD_ZERO.sub(r"\1", _ORD_SUFFIX.sub(r"\1", s))
    s = _FR_AREA_RE.sub(" ", s)
    s = _STATE_RE.sub(lambda m: STATE_CODES[m.group(0)], s)
    return " ".join(ADDR_CANON.get(t, t) for t in s.split() if t not in DROP_ADDR)


def prepare(df):
    """Add normalized columns used by blocking and features."""
    out = df.copy().reset_index(drop=True)
    out["country_key"] = out["country"].str.strip().str.lower()   # open set, never hard-coded
    out["name_n"] = out["business_name"].map(norm_name)
    out["name_core"] = [core_name(n, c) for n, c in zip(out["name_n"], out["country_key"])]
    out["addr_n"] = out["business_address"].map(norm_addr)
    out["full_n"] = out["name_core"] + " | " + out["addr_n"]
    out["nums"] = out["addr_n"].map(lambda s: frozenset(re.findall(r"\d+", s)))
    out["postal"] = out["nums"].map(lambda ns: frozenset(n for n in ns if len(n) in (5, 6)))
    out["raw"] = out["business_name"] + ", " + out["business_address"] + ", " + out["country"]
    return out


# ---------------------------------------------------------------- freshness
def up_to_date(outputs, inputs):
    """True if every output exists and is at least as new as every existing input.
    Stages use this (like a build tool) so leftovers from older runs are never reused."""
    outs = [Path(o) for o in outputs]
    ins = [Path(i) for i in inputs if Path(i).exists()]
    if not outs or not all(o.exists() for o in outs):
        return False
    return not ins or min(o.stat().st_mtime for o in outs) >= max(i.stat().st_mtime for i in ins)


# ---------------------------------------------------------------- folds & scoring
def assign_folds(s1_ids):
    """Deterministic split of training Source 1 ids into 'val', 'enc', 'match'.
    Splitting by Source 1 entity keeps all of an entity's matches on the same side."""
    ids = np.array(sorted(set(s1_ids)))
    np.random.default_rng(SEED).shuffle(ids)
    n_val, n_enc = int(len(ids) * VAL_FRAC), int(len(ids) * ENC_FRAC)
    return {x: ("val" if k < n_val else "enc" if k < n_val + n_enc else "match")
            for k, x in enumerate(ids)}


def f_beta(pred, true, beta=0.5):
    """Pair-level F-beta. Returns (f, precision, recall)."""
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    if p + r == 0:
        return 0.0, p, r
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r), p, r


# ---------------------------------------------------------------- optional encoder
ENCODER_PREFIX = "query: "   # e5-family models expect this prefix


def encode(texts, model_dir=ENCODER_DIR, batch_size=128):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_dir))
    emb = model.encode([ENCODER_PREFIX + t for t in texts], batch_size=batch_size,
                       normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True)
    return emb.astype(np.float32)
