import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .index import DocId, InvertedIndex, Vector, check_slope, log_tf, vector_norm
from .preprocess import STOPWORDS, preprocess, tokenize
from .scoring import Ranking, cosine_scores, pivoted_scores, rank

SEED = 15
K = 10
REPETITIONS = (5, 10, 25, 50)
PADDING_PCTS = (25, 50, 100, 200)
GROUPS = ("short", "medium", "long")

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Attack:
    kind: str
    repeats: int = 0
    pad_pct: int = 0

    @property
    def label(self) -> str:
        if self.kind == "repeat":
            return f"repeat x{self.repeats}"
        if self.kind == "pad":
            return f"pad {self.pad_pct}%"
        return f"repeat x{self.repeats} + pad {self.pad_pct}%"


def attack_grid() -> list[Attack]:
    return (
        [Attack("repeat", repeats=k) for k in REPETITIONS]
        + [Attack("pad", pad_pct=p) for p in PADDING_PCTS]
        + [Attack("combined", repeats=k, pad_pct=p) for k in REPETITIONS for p in PADDING_PCTS]
    )


def top_k(ranking: Ranking, k: int = K) -> set:
    return {d for d, _ in ranking[:k]}


def select_targets(
    index: InvertedIndex,
    queries: Mapping[str, str],
    qrels: Mapping[str, set],
    groups: Mapping[DocId, str],
    s: float,
    per_group: int = 2,
    seed: int = SEED,
    k: int = K,
) -> dict[str, list[DocId]]:
    # One generator over all queries in file order keeps the targets reproducible
    rng = random.Random(seed)
    targets: dict[str, list[DocId]] = {}
    for qid, text in queries.items():
        qvec = index.vectorize(text)
        top = top_k(rank(index, cosine_scores(index, qvec)), k) | top_k(
            rank(index, pivoted_scores(index, qvec, s)), k)
        chosen: list[DocId] = []
        for g in GROUPS:
            # Non-relevant and outside the top k under both cosine and pivoted
            pool = sorted(d for d in index.doc_ids
                          if groups[d] == g and d not in qrels[qid] and d not in top)
            if len(pool) < per_group:
                raise ValueError(f"{qid}: only {len(pool)} eligible {g} articles")
            chosen.extend(rng.sample(pool, per_group))
        targets[qid] = chosen
    return targets


def query_terms(query_text: str) -> list[str]:
    # Query words without stopwords, unstemmed
    return [tok for tok in tokenize(query_text) if tok not in STOPWORDS]


def repetition_text(query_text: str, repeats: int) -> str:
    return " ".join(query_terms(query_text) * repeats)


@dataclass(frozen=True)
class Sentence:
    doc_id: DocId
    text: str
    stems: frozenset


def split_sentences(articles: Sequence[dict]) -> list[Sentence]:
    out = []
    for a in sorted(articles, key=lambda a: a["document_id"]):
        for sent in _SENTENCE_BREAK.split(a["text"]):
            if sent.strip():
                out.append(Sentence(a["document_id"], sent, frozenset(preprocess(sent))))
    return out


def padding_stream(
    sentences: Sequence[Sentence],
    query_text: str,
    relevant: set,
    target: DocId,
    seed: int = SEED,
    qid: str = "",
) -> list[str]:
    q_stems = set(preprocess(query_text))
    # Sentences from other non-relevant articles that share no stem with the query
    pool = [x.text for x in sentences
            if x.doc_id != target and x.doc_id not in relevant and not q_stems & x.stems]
    random.Random(f"{seed}:{qid}:{target}").shuffle(pool)
    return [w for sent in pool for w in sent.split()]


def padding_words(word_count: int, pct: int) -> int:
    # Round half up
    return math.floor(word_count * pct / 100 + 0.5)


def padding_text(stream: Sequence[str], n_words: int, query_text: str) -> str:
    if n_words > len(stream):
        raise ValueError(f"padding pool has {len(stream)} words, {n_words} needed")
    text = " ".join(stream[:n_words])
    overlap = set(preprocess(query_text)) & set(preprocess(text))
    # Padding must never add a query term
    assert not overlap, f"padding contains query stems {sorted(overlap)}"
    return text


def added_text(attack: Attack, query_text: str, stream: Sequence[str], word_count: int) -> str:
    parts = []
    if attack.repeats:
        parts.append(repetition_text(query_text, attack.repeats))
    if attack.pad_pct:
        parts.append(padding_text(stream, padding_words(word_count, attack.pad_pct), query_text))
    return " ".join(parts)


@dataclass
class EditedDocument:
    weights: Vector
    norm: float
    added_words: int
    added_tokens: int


def edited_document(base_counts: Counter, extra: str) -> EditedDocument:
    extra_tokens = preprocess(extra) if extra else []
    # Only the edited article's weights and norm are recomputed; IDF stays fixed
    counts = base_counts + Counter(extra_tokens)
    weights = {t: log_tf(c) for t, c in counts.items()}
    return EditedDocument(weights, vector_norm(weights), len(extra.split()), len(extra_tokens))


def edited_score(
    index: InvertedIndex, qvec: Vector, doc: EditedDocument, s: Optional[float] = None
) -> float:
    nq = vector_norm(qvec)
    if nq == 0 or doc.norm == 0:
        return 0.0
    dot = 0.0
    for term in sorted(qvec):
        wd = doc.weights.get(term, 0.0)
        if wd > 0:
            dot += qvec[term] * wd
    if s is None:
        return dot / (nq * doc.norm)
    check_slope(s)
    # The pivot stays at its clean-collection value
    return dot / (nq * ((1.0 - s) * index.pivot + s * doc.norm))


def attacked_ranking(
    index: InvertedIndex, clean_scores: Sequence[float], target: DocId, new_score: float
) -> Ranking:
    scores = list(clean_scores)
    scores[index.position[target]] = new_score
    return rank(index, scores)


def rank_of(ranking: Ranking, doc_id: DocId, unretrieved: int) -> int:
    for i, (d, _) in enumerate(ranking, start=1):
        if d == doc_id:
            return i
    # Not retrieved (score 0)
    return unretrieved
