"""
Rule-based labelling of FinanceBench responses
=============================================

Labels each model response as one of:
    Correct | Partially Correct | Incorrect | Rejected

Two kinds of analysis feed the rules:
  * Numerical analysis  - for answers that are a number (ratios, amounts, %)
  * Semantic analysis   - for answers that are text (yes/no + explanation, drivers, lists)

Every row also gets a `rule_fired` column saying WHICH rule decided the label,
so the decision model is transparent and easy to explain in the report.

How to use in Jupyter: copy each "# %%" block into its own cell, top to bottom.
Input : baseline_results.jsonl  (your 150 questions, each with
        baseline_answer = no evidence, evidence_answer = with evidence)
Output: labelled_responses.csv  (300 rows, one per response)
        manual_check_sample.csv (60 rows for you to label by hand)
"""

# %% [Cell 1] Load data and reshape to one row per response (300 rows)
import re
import numpy as np
import pandas as pd

INPUT_FILE = "baseline_results.jsonl"
OUTPUT_FILE = "labelled_responses.csv"

raw = pd.read_json(INPUT_FILE, lines=True)

rows = []
for _, r in raw.iterrows():
    for condition, column in [("no_evidence", "baseline_answer"),
                              ("with_evidence", "evidence_answer")]:
        rows.append({
            "financebench_id": r["financebench_id"],
            "question_type": r["question_type"],
            "condition": condition,
            "question": r["question"],
            "expected_answer": str(r["expected_answer"]),
            "response": re.sub(r"<think>.*?</think>", "", str(r[column]), flags=re.S).strip(),
        })
data = pd.DataFrame(rows)
print(data.shape)  # should be (300, 6)


# %% [Cell 2] Helpers: find refusals, find the model's final answer, read numbers

REFUSAL_PATTERNS = [
    r"\binsufficient\b",
    r"\bcannot\s+(?:be\s+)?(?:provide|determine|answer|calculate|compute|confirm|answered|determined|calculated)",
    r"\bcan't\s+(?:provide|determine|answer|calculate)",
    r"\bunable to\b",
    r"\bnot\s+(?:enough|sufficient)\b",
    r"\bno\s+(?:specific\s+|numerical\s+|financial\s+)?(?:data|information|figures)\s+(?:was\s+|were\s+|is\s+)?(?:provided|available|exists)",
    r"\b(?:data|information|figures|statements?|details)\s+(?:was\s+|were\s+|is\s+|are\s+)?not\s+(?:provided|available|included)",
    r"\bdo(?:es)? not have (?:access|real-time|live)",
    r"\bnot\s+(?:calculable|determinable|publicly available)",
    r"\bimpossible to (?:determine|calculate|answer|compute)",
    r"\bevidence\s+(?:provided\s+)?(?:does not|doesn't)\s+(?:contain|provide|include|address|specify|mention|indicate)",
    r"\bevidence\s+(?:provided\s+)?(?:is\s+)?(?:only|exclusively)\s+(?:covers?|contains?|shows?|for)\b",
    r"\b(?:question|query)\s+(?:does not|doesn't)\s+(?:provide|include|contain)",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)

ANSWER_MARKER_RE = re.compile(
    r"(?:(?:final\s+)?answer\**\s*[:：]|final\s+answer\**\s*\n)\s*\**\s*([^\n]*)", re.I)


def final_answer_segment(text):
    """The part of the response that states the answer.
    Priority: the last 'Answer:' line  ->  the opening + closing sentences."""
    matches = list(ANSWER_MARKER_RE.finditer(text))
    if matches:
        m = matches[-1]
        seg = m.group(1).strip(" *")
        if not seg:  # 'Answer:' followed by a newline -> take the next non-empty line
            after = [ln.strip(" *-") for ln in text[m.end():].splitlines() if ln.strip(" *-")]
            seg = after[0] if after else ""
        return seg, True
    return text[:300] + " ... " + text[-300:], False


# Saying "this metric is not useful here" IS an answer (FinanceBench asks for it), not a refusal
NOT_USEFUL_RE = re.compile(r"\bnot\s+(?:a\s+)?(?:useful|relevant|meaningful|applicable)\b", re.I)


def is_refusal(text):
    if NOT_USEFUL_RE.search(text[:600]):
        return False
    seg, has_marker = final_answer_segment(text)
    if has_marker and REFUSAL_RE.search(seg):
        return True
    # otherwise judge by how the response OPENS (a refusal word near the end is
    # usually a side remark, e.g. "evidence is insufficient for FY2021 only")
    return bool(REFUSAL_RE.search(text[:300]))


NUM_RE = re.compile(r"(\(?)(-?)\s?\$?\s?(\d[\d,]*\.?\d*)\s?(\)?)\s?(%?)")


def parse_numbers(text):
    """All numbers in a text as floats. (1,577) -> -1577. Skips plain years like 2018."""
    out = []
    for m in NUM_RE.finditer(text):
        open_p, minus, digits, close_p, pct = m.groups()
        clean = digits.replace(",", "").rstrip(".")
        if not clean:
            continue
        try:
            val = float(clean)
        except ValueError:
            continue
        is_year = ("," not in digits and "." not in clean and 1990 <= val <= 2035)
        if is_year:
            continue
        # skip the '2' in 'Q2', 'FY22', 'Note 16' style references
        before = text[max(0, m.start() - 4):m.start()].lower()
        if before.endswith(("q", "fy", "note ", "h")):
            continue
        if minus or (open_p and close_p):
            val = -val
        out.append(val)
    return out


def predicted_number(text):
    """The single number the model gives as its answer."""
    seg, has_marker = final_answer_segment(text)
    if has_marker:
        nums = parse_numbers(seg)
        if nums:
            return nums[-1]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    for b in reversed(boxed):
        nums = parse_numbers(b)
        if nums:
            return nums[-1]
    bolds = [b for b in re.findall(r"\*\*(.+?)\*\*", text, flags=re.S)
             if re.search(r"\d", b) and not re.match(r"\s*(?:step|cell|note|q\d)", b, re.I)]
    for b in reversed(bolds):
        nums = parse_numbers(b)
        if nums:
            return nums[-1]
    nums = parse_numbers(text)
    return nums[-1] if nums else None


print(final_answer_segment("blah\nAnswer: 0.66")[0], predicted_number("Ratio is **1.73**"),
      parse_numbers("capex (1,577) in FY2018, margin 23.5%"))


# %% [Cell 3] Numerical analysis

GOLD_NUMBER_RE = re.compile(r"^\s*-?\$?\s?-?\d[\d,]*\.?\d*\s?%?\s*$")
SCALES = [1, 100, 0.01, 1000, 0.001]   # handles % vs decimal, millions vs billions


def is_numeric_gold(gold):
    return bool(GOLD_NUMBER_RE.match(gold))


def decimals_in(gold):
    m = re.search(r"\.(\d+)", gold)
    return len(m.group(1)) if m else 0


def relative_error(pred, gold_val):
    """Smallest relative error across unit scalings (sign ignored here)."""
    return min(abs(abs(pred) * s - abs(gold_val)) / max(abs(gold_val), 1e-9) for s in SCALES)


def rounds_to_gold(pred, gold_val, decimals):
    """True if the prediction, rounded like the gold answer, equals it (e.g. 0.0147 -> 0.01)."""
    return any(round(abs(pred) * s, decimals) == round(abs(gold_val), decimals)
               and abs(abs(pred) * s - abs(gold_val)) / max(abs(gold_val), 1e-9) <= 0.5
               for s in SCALES)


def numerical_label(response, gold):
    gold_val = parse_numbers(gold)[0]
    pred = predicted_number(response)
    if pred is None:
        return "Incorrect", "NUM: no number in response", np.nan
    err = relative_error(pred, gold_val)
    sign_wrong = (gold_val < 0) != (pred < 0) and gold_val != 0
    if err <= 0.01 or rounds_to_gold(pred, gold_val, decimals_in(gold)):
        if sign_wrong:
            return "Partially Correct", "NUM: right size, wrong sign", err
        return "Correct", "NUM: within 1% (or rounds to gold)", err
    if err <= 0.10:
        return "Partially Correct", "NUM: within 10%", err
    # the right number appears somewhere, but the model picked another one
    if any(abs(abs(n) - abs(gold_val)) <= 0.01 * abs(gold_val) for n in parse_numbers(response)):
        return "Partially Correct", "NUM: gold number present but not chosen", err
    return "Incorrect", "NUM: off by more than 10%", err


# %% [Cell 4] Semantic analysis
# Uses a sentence-embedding model if installed (pip install sentence-transformers),
# otherwise falls back to TF-IDF word overlap. Thresholds differ per backend.
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def similarities(gold, chunks):
        emb = _model.encode([gold] + chunks, normalize_embeddings=True)
        return emb[1:] @ emb[0]
    BACKEND = "embeddings"
    SIM_HIGH, SIM_LOW = 0.70, 0.45
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def similarities(gold, chunks):
        vec = TfidfVectorizer(stop_words="english").fit([gold] + chunks)
        m = vec.transform([gold] + chunks)
        return cosine_similarity(m[1:], m[0]).ravel()
    BACKEND = "tfidf"
    SIM_HIGH, SIM_LOW = 0.35, 0.18
print("Semantic backend:", BACKEND)


def best_similarity(gold, response):
    """Max similarity between the gold answer and any 1-2 sentence chunk of the response."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", response) if len(s.strip()) > 15]
    if not sents:
        return 0.0
    chunks = sents + [sents[i] + " " + sents[i + 1] for i in range(len(sents) - 1)]
    return float(np.max(similarities(gold, chunks)))


def polarity(text):
    m = re.search(r"\b(yes|no)\b", text, re.I)
    if m:
        return m.group(1).lower()
    if re.search(r"\b(?:did|does|do|has|have|is|was|were|are)\s+not\b|\b(?:didn't|doesn't|hasn't|isn't|wasn't)\b", text, re.I):
        return "no"
    return None


def gold_polarity(gold):
    m = re.match(r"\s*(yes|no)\b", gold, re.I)
    return m.group(1).lower() if m else None


def number_overlap(gold, response):
    """Share of the gold answer's key numbers that also appear in the response."""
    g_nums = parse_numbers(gold)
    if not g_nums:
        return None
    r_nums = parse_numbers(response)
    found = sum(any(relative_error(r, g) <= 0.02 for r in r_nums) for g in g_nums)
    return found / len(g_nums)


NONE_GOLD_RE = re.compile(r"^\s*(?:there (?:are|were|is) none|none|not applicable)\b", re.I)
NONE_RESP_RE = re.compile(r"\b(?:none|no\s+(?:\w+\s+){0,3}(?:securities|acquisitions|such)|did not make any|were no|are no)\b", re.I)


def semantic_label(response, gold):
    sim = best_similarity(gold, response)
    overlap = number_overlap(gold, response)
    gp = gold_polarity(gold)

    if NONE_GOLD_RE.match(gold):  # the right answer is "none"
        if NONE_RESP_RE.search(response[:400]):
            return "Correct", "SEM: correctly says none", sim
        return "Incorrect", "SEM: gold is none, response names something", sim

    if gp:  # yes/no question: the verdict decides first
        seg, has_marker = final_answer_segment(response)
        pp = polarity(seg) if has_marker else polarity(response[:200])
        if pp and pp != gp:
            return "Incorrect", f"SEM: yes/no verdict wrong ({pp} vs {gp})", sim
        if pp == gp:
            if overlap == 0:
                return "Partially Correct", "SEM: verdict right, key figures wrong/missing", sim
            return "Correct", "SEM: verdict right", sim
        # no clear yes/no given -> judge by meaning only (below)

    if sim >= SIM_HIGH and (overlap is None or overlap >= 0.5):
        return "Correct", "SEM: high similarity", sim
    if sim >= SIM_LOW or (overlap or 0) > 0:
        return "Partially Correct", "SEM: medium similarity / some key numbers", sim
    return "Incorrect", "SEM: low similarity", sim


# %% [Cell 5] The rule-based decision model
def decide(row):
    """Rules are checked in order; the first one that applies sets the label."""
    resp, gold = row["response"], row["expected_answer"]

    # Rule 1: the model declined to answer
    if is_refusal(resp):
        return pd.Series({"label": "Rejected", "rule_fired": "R1: refusal phrase", "score": np.nan})

    # Rule 2: the gold answer is a number -> numerical analysis
    if is_numeric_gold(gold):
        label, rule, err = numerical_label(resp, gold)
        return pd.Series({"label": label, "rule_fired": "R2 " + rule, "score": err})

    # Rule 3: the gold answer is text -> semantic analysis
    label, rule, sim = semantic_label(resp, gold)
    return pd.Series({"label": label, "rule_fired": "R3 " + rule, "score": sim})


data[["label", "rule_fired", "score"]] = data.apply(decide, axis=1)
data["answer_kind"] = np.where(data["expected_answer"].apply(is_numeric_gold), "numeric", "text")
data.to_csv(OUTPUT_FILE, index=False)
print("Saved", OUTPUT_FILE)


# %% [Cell 6] Results tables
ORDER = ["Correct", "Partially Correct", "Incorrect", "Rejected"]

print("\n=== Counts by condition ===")
counts = pd.crosstab(data["condition"], data["label"]).reindex(columns=ORDER, fill_value=0)
print(counts)

print("\n=== Percent by condition ===")
print((counts.div(counts.sum(axis=1), axis=0) * 100).round(1))

print("\n=== By condition and answer kind (numeric vs text) ===")
print(pd.crosstab([data["condition"], data["answer_kind"]], data["label"])
      .reindex(columns=ORDER, fill_value=0))

print("\n=== By condition and FinanceBench question type ===")
print(pd.crosstab([data["condition"], data["question_type"]], data["label"])
      .reindex(columns=ORDER, fill_value=0))

print("\n=== Which rules fired ===")
print(data["rule_fired"].value_counts())


# %% [Cell 7] Make a sample for hand-checking (do this before trusting the numbers)
parts = [g.sample(min(len(g), 8), random_state=42)
         for _, g in data.groupby(["condition", "label"])]
sample = pd.concat(parts)[["financebench_id", "condition", "question", "expected_answer",
                 "response", "label", "rule_fired"]].copy()
sample["human_label"] = ""  # fill in: Correct / Partially Correct / Incorrect / Rejected
sample.to_csv("manual_check_sample.csv", index=False)
print(len(sample), "rows saved to manual_check_sample.csv - open it and fill human_label")


# %% [Cell 8] After hand-labelling: how well do the rules agree with you?
def agreement_report(path="manual_check_sample.csv"):
    s = pd.read_csv(path).dropna(subset=["human_label"])
    s = s[s["human_label"].str.strip() != ""]
    acc = (s["label"] == s["human_label"].str.strip()).mean()
    # Cohen's kappa (agreement beyond chance)
    labels = ORDER
    p_obs = acc
    p_exp = sum((s["label"] == l).mean() * (s["human_label"].str.strip() == l).mean() for l in labels)
    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 1.0
    print(f"Rows checked: {len(s)}   Agreement: {acc:.1%}   Cohen's kappa: {kappa:.2f}")
    print(pd.crosstab(s["human_label"], s["label"], rownames=["human"], colnames=["rules"]))

# agreement_report()   # run after you fill in human_label
