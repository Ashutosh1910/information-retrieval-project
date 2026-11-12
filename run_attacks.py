import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from irengine import InvertedIndex, cosine_scores, pivoted_scores, preprocess, rank
from irengine.attacks import (
    GROUPS,
    PADDING_PCTS,
    REPETITIONS,
    SEED,
    added_text,
    attack_grid,
    attacked_ranking,
    edited_document,
    edited_score,
    padding_stream,
    query_terms,
    rank_of,
    select_targets,
    split_sentences,
)
from irengine.data import index_documents, load_articles
from irengine.metrics import evaluate
from run_epc import LABEL_SETS, length_groups, load_query_set

ROOT = Path(__file__).resolve().parent
K = 10
PER_GROUP = 2
METHODS = ("cosine", "pivoted")


def mean(xs):
    # Queries with no relevant article give None for AP and P@10
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def quality(clean, attacked):
    # No AP or P@10 for a query with no relevant article
    if clean is None:
        return {k: None for k in ("ap_clean", "ap_attacked", "d_ap", "p10_clean", "p10_attacked", "d_p10")}
    return {"ap_clean": clean.average_precision, "ap_attacked": attacked.average_precision,
            "d_ap": attacked.average_precision - clean.average_precision,
            "p10_clean": clean.precision, "p10_attacked": attacked.precision,
            "d_p10": attacked.precision - clean.precision}


def aggregate(rows):
    # Score ratio only for targets retrieved before the edit
    ratios = [r["new_score"] / r["old_score"] for r in rows if r["old_score"] > 0]
    return {
        "trials": len(rows),
        "successes": sum(r["success"] for r in rows),
        "success_pct": 100 * mean(r["success"] for r in rows),
        "median_new_rank": statistics.median(r["new_rank"] for r in rows),
        "median_rank_change": statistics.median(r["rank_change"] for r in rows),
        "mean_rank_change": mean(r["rank_change"] for r in rows),
        "mean_score_ratio": mean(ratios),
        "mean_added_words": mean(r["added_words"] for r in rows),
        "mean_added_tokens": mean(r["added_tokens"] for r in rows),
        "mean_d_ap": mean(r["d_ap"] for r in rows),
        "mean_d_p10": mean(r["d_p10"] for r in rows),
        "mean_d_map": mean(r["d_map"] for r in rows),
    }


def sign_test(a, b):
    n = a + b
    if n == 0:
        return 1.0
    # Exact two-sided sign test, ties dropped
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(cos_rows, piv_rows):
    pairs = list(zip(cos_rows, piv_rows))
    assert all((c["query_id"], c["document_id"]) == (p["query_id"], p["document_id"]) for c, p in pairs)
    more_p = sum(p["rank_change"] > c["rank_change"] for c, p in pairs)
    more_c = sum(c["rank_change"] > p["rank_change"] for c, p in pairs)
    return {
        "moved_more_pivoted": more_p,
        "moved_more_cosine": more_c,
        "moved_same": len(pairs) - more_p - more_c,
        "moved_more_sign_p": sign_test(more_p, more_c),
        "pivoted_higher": sum(p["new_rank"] < c["new_rank"] for c, p in pairs),
        "cosine_higher": sum(c["new_rank"] < p["new_rank"] for c, p in pairs),
        "same_rank": sum(c["new_rank"] == p["new_rank"] for c, p in pairs),
        "success_only_cosine": sum(c["success"] and not p["success"] for c, p in pairs),
        "success_only_pivoted": sum(p["success"] and not c["success"] for c, p in pairs),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", choices=sorted(LABEL_SETS), default="main",
                        help="main: Q1-Q12; extended: Q1-Q40 (adds queries_extended.json)")
    parser.add_argument("--out", default=None,
                        help="default results/attacks (main) or results/attacks_extended (extended)")
    args = parser.parse_args(argv)

    s = float(json.loads((ROOT / "frozen_slope.json").read_text(encoding="utf-8"))["s"])
    slopes = {"cosine": None, "pivoted": s}

    articles = load_articles(ROOT / "articles.json")
    by_id = {a["document_id"]: a for a in articles}
    words = {d: int(a["word_count"]) for d, a in by_id.items()}
    texts = dict(index_documents(articles))
    index = InvertedIndex(index_documents(articles))
    queries, qrels, labels = load_query_set(args.queries, set(index.doc_ids))
    empty = [q for q in queries if not qrels[q]]
    group, _ = length_groups(articles)
    # Rank given to an article with score 0
    unretrieved = index.N + 1

    # Clean collection: scores, rankings and metrics per method
    qvecs = {q: index.vectorize(t) for q, t in queries.items()}
    clean_scores = {"cosine": {q: cosine_scores(index, v) for q, v in qvecs.items()},
                    "pivoted": {q: pivoted_scores(index, v, s) for q, v in qvecs.items()}}
    clean_rank = {m: {q: rank(index, sc) for q, sc in clean_scores[m].items()} for m in METHODS}
    clean_eval = {m: evaluate({q: [d for d, _ in r] for q, r in clean_rank[m].items()}, qrels, k=K)
                  for m in METHODS}
    clean_q = {m: {x.query_id: x for x in ev.per_query} for m, ev in clean_eval.items()}
    clean_map = {m: ev.map for m, ev in clean_eval.items()}
    print(f"EPC: {index.N} articles, {len(queries)} queries ({len(empty)} with no relevant article), labels "
          + ", ".join(f"{x['file']} (frozen {x['frozen_on']})" for x in labels)
          + f", pivot p = {index.pivot:.6f} (fixed), s = {s:g}")
    print(f"Clean MAP: cosine {clean_map['cosine']:.4f}, pivoted {clean_map['pivoted']:.4f}")

    # The same targets are used for both methods
    targets = select_targets(index, queries, qrels, group, s, per_group=PER_GROUP, seed=SEED, k=K)
    target_rows = []
    for q, docs in targets.items():
        for d in docs:
            pos = index.position[d]
            target_rows.append({
                "query_id": q, "document_id": d, "words": words[d], "length_group": group[d],
                "norm": index.norms[pos],
                **{f"rank_{m}": rank_of(clean_rank[m][q], d, unretrieved) for m in METHODS},
                **{f"score_{m}": clean_scores[m][q][pos] for m in METHODS},
            })
    n_unret = sum(r["rank_cosine"] == unretrieved for r in target_rows)
    print(f"\nTargets: {len(target_rows)} ({PER_GROUP} per length group per query, seed {SEED}); "
          f"{n_unret} are not retrieved at all before the attack (score 0, rank {unretrieved})")

    # One edited article per trial; every query is re-ranked and re-evaluated with it
    sentences = split_sentences(articles)
    grid = attack_grid()
    rows = []
    for t in target_rows:
        q, d = t["query_id"], t["document_id"]
        stream = padding_stream(sentences, queries[q], qrels[q], d, seed=SEED, qid=q)
        base = Counter(preprocess(texts[d]))
        for attack in grid:
            ed = edited_document(base, added_text(attack, queries[q], stream, words[d]))
            for m in METHODS:
                new_scores = {q2: edited_score(index, qvecs[q2], ed, slopes[m]) for q2 in queries}
                rankings = {q2: attacked_ranking(index, clean_scores[m][q2], d, new_scores[q2])
                            for q2 in queries}
                ev = evaluate({q2: [x for x, _ in r] for q2, r in rankings.items()}, qrels, k=K)
                att_q = {x.query_id: x for x in ev.per_query}
                old_rank = t[f"rank_{m}"]
                new_rank = rank_of(rankings[q], d, unretrieved)
                rows.append({
                    "query_id": q, "document_id": d, "words": words[d], "length_group": group[d],
                    "attack": attack.label, "kind": attack.kind,
                    "repeats": attack.repeats, "pad_pct": attack.pad_pct, "method": m,
                    "added_words": ed.added_words, "added_tokens": ed.added_tokens,
                    "old_norm": t["norm"], "new_norm": ed.norm,
                    "old_score": t[f"score_{m}"], "new_score": new_scores[q],
                    "old_rank": old_rank, "new_rank": new_rank, "rank_change": old_rank - new_rank,
                    "started_unretrieved": int(old_rank == unretrieved),
                    "success": int(new_rank <= K),
                    "query_has_relevant": int(q in att_q),
                    **quality(clean_q[m].get(q), att_q.get(q)),
                    "map_clean": clean_map[m], "map_attacked": ev.map, "d_map": ev.map - clean_map[m],
                    "other_queries_ap_changed": sum(
                        att_q[q2].average_precision != clean_q[m][q2].average_precision
                        for q2 in att_q if q2 != q),
                })

    def select(m, **conds):
        return [r for r in rows if r["method"] == m and all(r[k] == v for k, v in conds.items())]

    level_rows, group_rows, query_rows = [], [], []
    for attack in grid:
        sel = {m: select(m, attack=attack.label) for m in METHODS}
        pr = paired(sel["cosine"], sel["pivoted"])
        for m in METHODS:
            level_rows.append({"attack": attack.label, "kind": attack.kind, "repeats": attack.repeats,
                               "pad_pct": attack.pad_pct, "method": m, **aggregate(sel[m]), **pr})
            for g in GROUPS:
                group_rows.append({"attack": attack.label, "kind": attack.kind, "length_group": g,
                                   "method": m, **aggregate([r for r in sel[m] if r["length_group"] == g])})
            for q in queries:
                query_rows.append({"attack": attack.label, "kind": attack.kind, "query_id": q,
                                   "method": m, **aggregate([r for r in sel[m] if r["query_id"] == q])})

    def lv(label, m, key, table=level_rows, **extra):
        return next(r[key] for r in table
                    if r["attack"] == label and r["method"] == m and all(r[k] == v for k, v in extra.items()))

    def level_table(title, labels):
        print(f"\n{title}")
        print(f"  {'attack':<24}{'+words':>7}{'success % c/p':>16}{'median new rank':>18}"
              f"{'new/old score c/p':>20}{'mean dMAP c/p':>20}{'moved up more piv/cos/same':>29}"
              f"{'sign p':>9}")
        for a in labels:
            c = next(r for r in level_rows if r["attack"] == a and r["method"] == "cosine")
            p = next(r for r in level_rows if r["attack"] == a and r["method"] == "pivoted")
            print(f"  {a:<24}{c['mean_added_words']:>7.0f}{c['success_pct']:>9.1f}/{p['success_pct']:<6.1f}"
                  f"{c['median_new_rank']:>10.1f}/{p['median_new_rank']:<7.1f}"
                  f"{c['mean_score_ratio']:>12.3f}/{p['mean_score_ratio']:<7.3f}"
                  f"{c['mean_d_map']:>+11.4f}/{p['mean_d_map']:<+8.4f}"
                  f"{c['moved_more_pivoted']:>10}/{c['moved_more_cosine']}/{c['moved_same']:<8}"
                  f"{c['moved_more_sign_p']:>9.4f}")

    reps = [a.label for a in grid if a.kind == "repeat"]
    pads = [a.label for a in grid if a.kind == "pad"]
    diag = [a.label for a in grid if a.kind == "combined" and
            REPETITIONS.index(a.repeats) == PADDING_PCTS.index(a.pad_pct)]
    n = len(target_rows)
    print(f"\nEach row: {n} trials per method. drank = old rank - new rank (positive = moved up).")
    level_table("Keyword repetition", reps)
    level_table("Unrelated padding", pads)
    level_table("Combined (diagonal of the 4 x 4 grid)", diag)

    combined_grid = {}
    for m in METHODS:
        combined_grid[m] = {str(k): {str(p): lv(f"repeat x{k} + pad {p}%", m, "success_pct")
                                     for p in PADDING_PCTS} for k in REPETITIONS}
    print("\nCombined: success % (rows: repetitions, columns: padding %)")
    for m in METHODS:
        print(f"  {m:<8}" + "".join(f"{p:>8}%" for p in PADDING_PCTS))
        for k in REPETITIONS:
            print(f"    x{k:<5}" + "".join(f"{combined_grid[m][str(k)][str(p)]:>9.1f}" for p in PADDING_PCTS))

    print(f"\nBy length group ({PER_GROUP * len(queries)} trials per group and method): success % c/p "
          "for repetition and combined, median drank c/p for padding")
    print(f"  {'attack':<24}" + "".join(f"{g:>15}" for g in GROUPS))
    for a in reps + diag:
        print(f"  {a:<24}" + "".join(
            f"{lv(a, 'cosine', 'success_pct', group_rows, length_group=g):>8.0f}/"
            f"{lv(a, 'pivoted', 'success_pct', group_rows, length_group=g):<6.0f}" for g in GROUPS))
    for a in pads:
        print(f"  {a:<24}" + "".join(
            f"{lv(a, 'cosine', 'median_rank_change', group_rows, length_group=g):>8.1f}/"
            f"{lv(a, 'pivoted', 'median_rank_change', group_rows, length_group=g):<6.1f}" for g in GROUPS))

    per_q = PER_GROUP * len(GROUPS)
    print(f"\nBy query ({per_q} targets each): successes c/p out of {per_q}")
    print(f"  {'query':<6}" + "".join(f"{a:>12}" for a in [f"x{k}" for k in REPETITIONS])
          + "".join(f"{a:>14}" for a in [f"x{k}+{p}%" for k, p in zip(REPETITIONS, PADDING_PCTS)]))
    for q in queries:
        cells = [f"{lv(a, 'cosine', 'successes', query_rows, query_id=q)}/"
                 f"{lv(a, 'pivoted', 'successes', query_rows, query_id=q)}" for a in reps + diag]
        print(f"  {q:<6}" + "".join(f"{c:>12}" for c in cells[:4]) + "".join(f"{c:>14}" for c in cells[4:]))
    print(f"\nBy query: median drank c/p under padding (padding alone cannot help, so all are <= 0)")
    print(f"  {'query':<6}" + "".join(f"{f'{p}%':>14}" for p in PADDING_PCTS))
    for q in queries:
        print(f"  {q:<6}" + "".join(
            f"{lv(a, 'cosine', 'median_rank_change', query_rows, query_id=q):>7.1f}/"
            f"{lv(a, 'pivoted', 'median_rank_change', query_rows, query_id=q):<6.1f}" for a in pads))

    # Queries with a relevant article vs queries without one
    type_rows = []
    if empty:
        for attack in grid:
            for m in METHODS:
                for has in (1, 0):
                    sel = [r for r in select(m, attack=attack.label) if r["query_has_relevant"] == has]
                    type_rows.append({"attack": attack.label, "kind": attack.kind, "method": m,
                                      "query_has_relevant": has, **aggregate(sel)})
        n_with = PER_GROUP * len(GROUPS) * (len(queries) - len(empty))
        n_without = PER_GROUP * len(GROUPS) * len(empty)
        print(f"\nBy query type: with a relevant article ({n_with} targets) vs without one ({n_without}); "
              "success % c/p, and median drank c/p for padding")
        print(f"  {'attack':<24}{'with relevant':>18}{'no relevant':>18}")
        for a in reps + diag:
            print(f"  {a:<24}" + "".join(
                f"{lv(a, 'cosine', 'success_pct', type_rows, query_has_relevant=h):>11.1f}/"
                f"{lv(a, 'pivoted', 'success_pct', type_rows, query_has_relevant=h):<6.1f}" for h in (1, 0)))
        for a in pads:
            print(f"  {a:<24}" + "".join(
                f"{lv(a, 'cosine', 'median_rank_change', type_rows, query_has_relevant=h):>11.1f}/"
                f"{lv(a, 'pivoted', 'median_rank_change', type_rows, query_has_relevant=h):<6.1f}" for h in (1, 0)))

    # Per target over the 16 combined cells, so the target is the unit of the sign test
    cells = {}
    for r in rows:
        if r["kind"] == "combined":
            cells.setdefault((r["query_id"], r["document_id"]), {m: 0 for m in METHODS})[r["method"]] += r["success"]
    per_target = {
        "pivoted_more_cells": sum(v["pivoted"] > v["cosine"] for v in cells.values()),
        "cosine_more_cells": sum(v["cosine"] > v["pivoted"] for v in cells.values()),
        "same": sum(v["cosine"] == v["pivoted"] for v in cells.values()),
    }
    per_target["sign_p"] = sign_test(per_target["pivoted_more_cells"], per_target["cosine_more_cells"])
    print(f"\nCombined, per target over the 16 cells: entered the top 10 in more cells under pivoted "
          f"for {per_target['pivoted_more_cells']} targets, under cosine for {per_target['cosine_more_cells']}, "
          f"same for {per_target['same']} (sign test p = {per_target['sign_p']:.2g})")

    # Padding adds no query term, so it can never raise a score
    pad_rows = [r for r in rows if r["kind"] == "pad"]
    pad_success = sum(r["success"] for r in pad_rows)
    pad_raised = sum(r["new_score"] > r["old_score"] for r in pad_rows)
    print(f"\nCheck: padding-only trials that entered the top 10: {pad_success}; "
          f"that raised the target's score: {pad_raised} (both must be 0)")

    out = Path(args.out or ROOT / "results" / ("attacks" if args.queries == "main" else "attacks_extended"))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "targets.csv", target_rows)
    write_csv(out / "trials.csv", rows)
    write_csv(out / "levels.csv", level_rows)
    write_csv(out / "by_group.csv", group_rows)
    write_csv(out / "by_query.csv", query_rows)
    if type_rows:
        write_csv(out / "by_query_type.csv", type_rows)
    summary = {
        "query_set": args.queries, "labels": labels, "queries_without_relevant": empty,
        "slope": s, "pivot_fixed": index.pivot, "k": K, "seed": SEED, "unretrieved_rank": unretrieved,
        "targets": {"total": len(target_rows), "per_group_per_query": PER_GROUP,
                    "not_retrieved_before_attack": n_unret},
        "attacks": {"repetitions": list(REPETITIONS), "padding_pct": list(PADDING_PCTS),
                    "combined": "full 4 x 4 grid; repetition then padding; padding % of original length",
                    "repetition_text": "the query's words after stopword removal, unstemmed",
                    "padding_source": "body sentences of other articles not relevant for the query, "
                                      "dropping any sentence that contains a query stem"},
        "query_terms": {q: query_terms(t) for q, t in queries.items()},
        "clean_map": clean_map,
        "levels": level_rows,
        "combined_success_pct": combined_grid,
        "combined_per_target": per_target,
        "by_query_type": type_rows,
        "checks": {"padding_successes": pad_success, "padding_raised_score": pad_raised},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/targets.csv, trials.csv ({len(rows)} rows), levels.csv, by_group.csv, "
          f"by_query.csv{', by_query_type.csv' if type_rows else ''}, summary.json")
    if pad_success or pad_raised:
        sys.exit("padding alone changed a target for the better: check the attack code")


if __name__ == "__main__":
    main()
