import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from irengine import InvertedIndex, search
from irengine.data import index_documents, load_articles, load_qrels, load_queries
from irengine.metrics import evaluate
from irengine.significance import RESAMPLES, paired_randomization_test

ROOT = Path(__file__).resolve().parent
K = 10
GROUPS = ("short", "medium", "long")


# Query files and their frozen label records; extended adds Q13-Q40
LABEL_SETS = {
    "main": [("queries.json", "labels_frozen.json")],
    "extended": [("queries.json", "labels_frozen.json"),
                 ("queries_extended.json", "labels_frozen_extended.json")],
}


def load_frozen_labels(record_name="labels_frozen.json"):
    record = json.loads((ROOT / record_name).read_text(encoding="utf-8"))
    path = ROOT / record["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Refuse to run if the frozen labels changed
    if digest != record["sha256"]:
        sys.exit(f"{path.name} does not match the checksum frozen in {record_name}; "
                 "the labels changed after the freeze.")
    return path, record


def load_query_set(name, known_doc_ids):
    queries, qrels, labels = {}, {}, []
    for qfile, record_name in LABEL_SETS[name]:
        # Every label file must still match its frozen checksum
        path, record = load_frozen_labels(record_name)
        part_q = load_queries(ROOT / qfile)
        part_r = load_qrels(path, known_doc_ids=known_doc_ids)
        if set(part_q) & set(queries) or set(part_r) & set(qrels):
            sys.exit(f"{qfile} repeats a query ID from an earlier file")
        if set(part_r) != set(part_q):
            sys.exit(f"{path.name} does not label exactly the queries in {qfile}")
        queries.update(part_q)
        qrels.update(part_r)
        labels.append({k: record[k] for k in ("file", "sha256", "frozen_on", "relevant")})
    return queries, qrels, labels


def length_groups(articles):
    order = sorted(articles, key=lambda a: (int(a["word_count"]), a["document_id"]))
    n = len(order)
    # Word-count tertiles
    cuts = (round(n / 3), round(2 * n / 3))
    group = {}
    for i, a in enumerate(order):
        group[a["document_id"]] = GROUPS[0] if i < cuts[0] else GROUPS[1] if i < cuts[1] else GROUPS[2]
    bounds = {g: [min(int(a["word_count"]) for a in order if group[a["document_id"]] == g),
                  max(int(a["word_count"]) for a in order if group[a["document_id"]] == g)]
              for g in GROUPS}
    return group, bounds


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", choices=sorted(LABEL_SETS), default="main",
                        help="main: Q1-Q12; extended: Q1-Q40 (adds queries_extended.json)")
    parser.add_argument("--out", default=None,
                        help="default results/epc (main) or results/epc_extended (extended)")
    args = parser.parse_args(argv)

    s = float(json.loads((ROOT / "frozen_slope.json").read_text(encoding="utf-8"))["s"])

    articles = load_articles(ROOT / "articles.json")
    by_id = {a["document_id"]: a for a in articles}
    words = {d: int(a["word_count"]) for d, a in by_id.items()}
    index = InvertedIndex(index_documents(articles))
    queries, qrels, labels = load_query_set(args.queries, set(index.doc_ids))
    print(f"EPC: {index.N} articles, {len(queries)} queries, labels "
          + ", ".join(f"{x['file']} (frozen {x['frozen_on']}, {x['relevant']} relevant)" for x in labels)
          + f", pivot p = {index.pivot:.6f}, frozen s = {s:g}")

    systems = {"cosine": None, "pivoted": s}
    # Full rankings for both methods
    runs = {name: {qid: search(index, text, s=sl) for qid, text in queries.items()}
            for name, sl in systems.items()}
    rankings = {name: {q: [d for d, _ in r] for q, r in run.items()} for name, run in runs.items()}

    evals = {name: evaluate(rankings[name], qrels, k=K) for name in systems}
    per_q = {name: {q.query_id: q for q in ev.per_query} for name, ev in evals.items()}
    evaluated = [q.query_id for q in evals["cosine"].per_query]
    ap = {name: [per_q[name][q].average_precision for q in evaluated] for name in systems}
    p10 = {name: [per_q[name][q].precision for q in evaluated] for name in systems}
    # Paired randomization tests on per-query AP and P@10
    p_ap = paired_randomization_test(ap["pivoted"], ap["cosine"])
    p_p10 = paired_randomization_test(p10["pivoted"], p10["cosine"])
    better = sum(a > b for a, b in zip(ap["pivoted"], ap["cosine"]))
    worse = sum(a < b for a, b in zip(ap["pivoted"], ap["cosine"]))

    empty = [q.query_id for q in evals["cosine"].no_relevant]
    print(f"\nAverages over {len(evaluated)} queries"
          + (f"; {len(empty)} with no relevant article are left out: {', '.join(empty)}" if empty
             else " (every query has a relevant article)"))
    print(f"  {'system':<10}{'P@10':>8}{'R@10':>8}{'F1@10':>8}{'MAP':>8}")
    for name, ev in evals.items():
        print(f"  {name:<10}{ev.mean_precision:>8.4f}{ev.mean_recall:>8.4f}{ev.mean_f1:>8.4f}{ev.map:>8.4f}")
    print(f"  AP: pivoted better on {better}, worse on {worse}, equal on {len(evaluated) - better - worse}; "
          f"randomization p = {p_ap:.4f} (AP), {p_p10:.4f} (P@10), {RESAMPLES} resamples")

    print("\nPer query")
    print(f"  {'query':<6}{'rel':>4}  {'P@10 c/p':>11}  {'R@10 c/p':>11}  {'AP cos':>7}{'AP piv':>8}{'dAP':>8}")
    for q in evaluated:
        c, p = per_q["cosine"][q], per_q["pivoted"][q]
        print(f"  {q:<6}{c.num_relevant:>4}  {c.precision:>5.2f}/{p.precision:<5.2f}  "
              f"{c.recall:>5.2f}/{p.recall:<5.2f}  {c.average_precision:>7.4f}{p.average_precision:>8.4f}"
              f"{p.average_precision - c.average_precision:>+8.4f}")

    # Length groups and their share of top-10 slots
    group, bounds = length_groups(articles)

    # Queries with no relevant article have no metrics; report their top 10 instead
    empty_rows = []
    for q in empty:
        tc, tp = rankings["cosine"][q][:K], rankings["pivoted"][q][:K]
        empty_rows.append({
            "query_id": q, "text": queries[q], "retrieved": len(rankings["cosine"][q]),
            "top10_overlap": len(set(tc) & set(tp)),
            **{f"top10_mean_words_{name}": mean(words[d] for d in t) for name, t in (("cosine", tc), ("pivoted", tp))},
            **{f"top10_{g}_{name}": sum(group[d] == g for d in t)
               for name, t in (("cosine", tc), ("pivoted", tp)) for g in GROUPS},
        })
    if empty_rows:
        print("\nQueries with no relevant article (no metrics; what each method puts in the top 10)")
        print(f"  {'query':<6}{'retrieved':>10}{'overlap':>9}{'mean words c/p':>18}"
              f"{'short/medium/long cos':>23}{'piv':>9}")
        for r in empty_rows:
            print(f"  {r['query_id']:<6}{r['retrieved']:>10}{r['top10_overlap']:>6}/10"
                  f"{r['top10_mean_words_cosine'] or 0:>10,.0f}/{r['top10_mean_words_pivoted'] or 0:<7,.0f}"
                  f"{'/'.join(str(r[f'top10_{g}_cosine']) for g in GROUPS):>16}"
                  f"{'/'.join(str(r[f'top10_{g}_pivoted']) for g in GROUPS):>16}")
    sizes = {g: sum(1 for d in group if group[d] == g) for g in GROUPS}
    print("\nLength groups (body word count, before stopword removal)")
    for g in GROUPS:
        print(f"  {g:<7} {sizes[g]:>3} articles, {bounds[g][0]:,}-{bounds[g][1]:,} words")

    relevant_pairs = [(q, d) for q in evaluated for d in sorted(qrels[q])]
    top10 = {name: [(q, d) for q in evaluated for d in rankings[name][q][:K]] for name in systems}
    group_rows = []
    for g in GROUPS:
        row = {"group": g, "articles": sizes[g], "min_words": bounds[g][0], "max_words": bounds[g][1],
               "relevant_pairs": sum(group[d] == g for _, d in relevant_pairs)}
        for name in systems:
            row[f"top10_{name}"] = sum(group[d] == g for _, d in top10[name])
            row[f"relevant_in_top10_{name}"] = sum(group[d] == g and d in qrels[q] for q, d in top10[name])
        group_rows.append(row)
    total_slots = {name: len(top10[name]) for name in systems}
    print("\nShare of top-10 slots vs share of relevant documents, by length group")
    print(f"  {'group':<8}{'relevant':>10}{'cosine top10':>14}{'pivoted top10':>15}"
          f"{'rel found cos':>15}{'rel found piv':>15}")
    for r in group_rows:
        print(f"  {r['group']:<8}{r['relevant_pairs']:>4} {r['relevant_pairs'] / len(relevant_pairs):>5.0%}"
              f"{r['top10_cosine']:>8} {r['top10_cosine'] / total_slots['cosine']:>5.0%}"
              f"{r['top10_pivoted']:>9} {r['top10_pivoted'] / total_slots['pivoted']:>5.0%}"
              f"{r['relevant_in_top10_cosine']:>11}/{r['relevant_pairs']:<3}"
              f"{r['relevant_in_top10_pivoted']:>11}/{r['relevant_pairs']:<3}")
    mean_words = {
        "collection": mean(words.values()),
        "relevant": mean(words[d] for _, d in relevant_pairs),
        **{f"top10_{name}": mean(words[d] for _, d in top10[name]) for name in systems},
    }
    print("  mean words: " + ", ".join(f"{k} {v:,.0f}" for k, v in mean_words.items()))

    # Where cosine and pivoted disagree
    disagreements = []
    for q in evaluated:
        tc, tp = rankings["cosine"][q][:K], rankings["pivoted"][q][:K]
        d_ap = per_q["pivoted"][q].average_precision - per_q["cosine"][q].average_precision
        rank = {name: {d: i for i, d in enumerate(rankings[name][q], 1)} for name in systems}
        disagreements.append({
            "query_id": q, "text": queries[q], "delta_ap": d_ap,
            "ap_cosine": per_q["cosine"][q].average_precision, "ap_pivoted": per_q["pivoted"][q].average_precision,
            "retrieved": len(rankings["cosine"][q]),
            "top10_size": len(tc),
            "top10_overlap": len(set(tc) & set(tp)),
            "moved_within_top10": sum(1 for i, d in enumerate(tc) if d in tp and tp.index(d) != i),
            "entered_top10": [{"doc": d, "words": words[d], "group": group[d], "relevant": d in qrels[q],
                               "rank_cosine": rank["cosine"].get(d), "rank_pivoted": rank["pivoted"][d]}
                              for d in tp if d not in tc],
            "left_top10": [{"doc": d, "words": words[d], "group": group[d], "relevant": d in qrels[q],
                            "rank_cosine": rank["cosine"][d], "rank_pivoted": rank["pivoted"].get(d)}
                           for d in tc if d not in tp],
            "relevant_ranks": [{"doc": d, "words": words[d], "group": group[d],
                                "rank_cosine": rank["cosine"].get(d), "rank_pivoted": rank["pivoted"].get(d)}
                               for d in sorted(qrels[q])],
        })
    # Largest AP change first, then swaps, then reorders
    disagreements.sort(key=lambda r: (-abs(r["delta_ap"]), r["top10_overlap"] - r["top10_size"],
                                      -r["moved_within_top10"], r["query_id"]))
    print("\nQueries by disagreement (|dAP|, then documents swapped into or out of the top 10, "
          "then documents reordered within it)")
    for r in disagreements:
        print(f"  {r['query_id']:<4} dAP {r['delta_ap']:+.4f}  overlap {r['top10_overlap']:>2}/{r['top10_size']:<2} "
              f"in {len(r['entered_top10'])} out {len(r['left_top10'])}  reordered {r['moved_within_top10']:>2}  "
              f"retrieved {r['retrieved']}")
    for r in disagreements[:3]:
        print(f"\n  {r['query_id']} \"{r['text']}\"  AP {r['ap_cosine']:.4f} -> {r['ap_pivoted']:.4f}")
        for x in r["relevant_ranks"]:
            print(f"    relevant {x['doc']} ({x['words']:,} words, {x['group']}): rank "
                  f"{x['rank_cosine']} -> {x['rank_pivoted']}")
        for label, xs in (("entered", r["entered_top10"]), ("left", r["left_top10"])):
            for x in xs:
                print(f"    {label:<7} {x['doc']} ({x['words']:,} words, {x['group']}, "
                      f"{'relevant' if x['relevant'] else 'not relevant'}): rank {x['rank_cosine']} -> {x['rank_pivoted']}")

    out = Path(args.out or ROOT / "results" / ("epc" if args.queries == "main" else "epc_extended"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "num_relevant", "system", "P@10", "R@10", "F1@10", "AP"])
        for q in evaluated:
            for name in systems:
                x = per_q[name][q]
                w.writerow([q, x.num_relevant, name, x.precision, x.recall, x.f1, x.average_precision])
    with open(out / "top10.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "system", "rank", "document_id", "score", "words", "length_group", "relevant"])
        for name in systems:
            for q in queries:
                for i, (d, sc) in enumerate(runs[name][q][:K], 1):
                    w.writerow([q, name, i, d, sc, words[d], group[d], int(d in qrels[q])])
    with open(out / "length_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(group_rows)
    with open(out / "article_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["document_id", "words", "length_group", "norm"])
        for d in index.doc_ids:
            w.writerow([d, words[d], group[d], index.norms[index.position[d]]])
    summary = {
        "query_set": args.queries, "labels": labels,
        "slope": s, "pivot": index.pivot, "k": K, "queries_evaluated": len(evaluated),
        "no_relevant_queries": empty_rows,
        "systems": {name: {"P@10": ev.mean_precision, "R@10": ev.mean_recall, "F1@10": ev.mean_f1, "MAP": ev.map}
                    for name, ev in evals.items()},
        "ap_better_worse_equal": [better, worse, len(evaluated) - better - worse],
        "randomization_p": {"AP": p_ap, "P@10": p_p10, "resamples": RESAMPLES},
        "length_groups": {"bounds_words": bounds, "sizes": sizes, "rows": group_rows,
                          "mean_words": mean_words, "top10_slots": total_slots,
                          "relevant_pairs": len(relevant_pairs)},
        "disagreements": disagreements,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/summary.json, per_query.csv, top10.csv, length_groups.csv, article_groups.csv")


if __name__ == "__main__":
    main()
