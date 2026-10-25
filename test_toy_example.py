# Toy collection checked against hand-calculated weights, scores, ties and metrics.
import math
import unittest

from irengine import InvertedIndex, cosine_scores, pivoted_scores, rank, search
from irengine.metrics import evaluate

L = math.log(2)

DOCS = [
    (1, "Apple apple banana."),
    (2, "Apple, cherry!"),
    (3, "Banana cherry date"),
    # Only stopwords: a zero-norm document
    (4, "The and of it."),
]

N1 = math.sqrt((1 + L) ** 2 + 1)
N2 = math.sqrt(2)
N3 = math.sqrt(3)
# Pivot leaves out the zero-norm document
P = (N1 + N2 + N3) / 3
NQ = L * math.sqrt(2)

Q1 = "apple banana"
# Hand-derived dot products per document
Q1_DOT = {1: L * (2 + L), 2: L, 3: L}
Q2 = "banana cherry"
Q2_DOT = {1: L, 2: L, 3: 2 * L}
NORMS = {1: N1, 2: N2, 3: N3}


# Closed-form score; s=None means cosine
def expected(dots, s=None):
    out = {}
    for d, dot in dots.items():
        factor = NORMS[d] if s is None else (1 - s) * P + s * NORMS[d]
        out[d] = dot / (NQ * factor)
    return out


class ToyExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ix = InvertedIndex(DOCS)

    def scores_by_id(self, scores):
        return {self.ix.doc_ids[i]: sc for i, sc in enumerate(scores)}

    def test_postings_and_df(self):
        self.assertEqual(self.ix.N, 4)
        self.assertEqual(self.ix.postings["appl"], [(0, 2), (1, 1)])
        self.assertEqual(self.ix.postings["banana"], [(0, 1), (2, 1)])
        self.assertEqual(self.ix.postings["date"], [(2, 1)])
        self.assertEqual(self.ix.df, {"appl": 2, "banana": 2, "cherri": 2, "date": 1})
        self.assertEqual(self.ix.doc_lengths, [3, 2, 3, 0])

    def test_idf_uses_all_documents(self):
        for t in ("appl", "banana", "cherri"):
            self.assertAlmostEqual(self.ix.idf[t], L, places=15)
        self.assertAlmostEqual(self.ix.idf["date"], 2 * L, places=15)

    def test_document_weights_have_no_idf(self):
        self.assertAlmostEqual(self.ix.document_vector(1)["appl"], 1 + L, places=15)
        self.assertEqual(self.ix.document_vector(1)["banana"], 1.0)
        self.assertEqual(self.ix.document_vector(3)["date"], 1.0)
        self.assertEqual(self.ix.document_vector(4), {})

    def test_query_weights_include_idf(self):
        self.assertEqual(self.ix.vectorize("apple banana").keys(), {"appl", "banana"})
        q = self.ix.vectorize("apple apple date")
        self.assertAlmostEqual(q["appl"], (1 + L) * L, places=15)
        self.assertAlmostEqual(q["date"], 2 * L, places=15)

    def test_norms(self):
        for pos, n in enumerate((N1, N2, N3)):
            self.assertAlmostEqual(self.ix.norms[pos], n, places=14)
        self.assertEqual(self.ix.norms[3], 0.0)
        self.assertAlmostEqual(self.ix.norms[0], 1.966405, places=6)
        self.assertAlmostEqual(self.ix.norms[1], 1.414214, places=6)
        self.assertAlmostEqual(self.ix.norms[2], 1.732051, places=6)

    def test_pivot_excludes_zero_norm_document(self):
        self.assertAlmostEqual(self.ix.pivot, P, places=14)
        self.assertAlmostEqual(self.ix.pivot, 1.704223, places=6)
        self.assertEqual(self.ix.num_nonzero_docs, 3)
        self.assertNotAlmostEqual(self.ix.pivot, (N1 + N2 + N3) / 4, places=3)

    def test_cosine_scores_q1(self):
        got = self.scores_by_id(cosine_scores(self.ix, self.ix.vectorize(Q1)))
        for d, v in expected(Q1_DOT).items():
            self.assertAlmostEqual(got[d], v, places=14)
        self.assertAlmostEqual(got[1], 0.968439, places=6)
        self.assertAlmostEqual(got[2], 0.5, places=14)
        self.assertAlmostEqual(got[3], 1 / math.sqrt(6), places=14)
        self.assertEqual(got[4], 0.0)

    def test_cosine_ranking(self):
        self.assertEqual([d for d, _ in search(self.ix, Q1)], [1, 2, 3])
        self.assertEqual([d for d, _ in search(self.ix, Q2)], [3, 2, 1])

    def test_unseen_query_terms_ignored(self):
        base = cosine_scores(self.ix, self.ix.vectorize(Q1))
        with_unseen = cosine_scores(self.ix, self.ix.vectorize(Q1 + " zebra quokka"))
        self.assertEqual(base, with_unseen)
        self.assertEqual(self.ix.vectorize("zebra"), {})

    def test_zero_norm_document_scores_zero_everywhere(self):
        for q in (Q1, Q2, "apple cherry date"):
            qv = self.ix.vectorize(q)
            self.assertEqual(cosine_scores(self.ix, qv)[3], 0.0)
            for s in (0.0, 0.3, 1.0):
                self.assertEqual(pivoted_scores(self.ix, qv, s)[3], 0.0)
            self.assertNotIn(4, [d for d, _ in search(self.ix, q)])

    def test_zero_norm_query_scores_zero(self):
        for q in ("", "the of and", "zebra", "The zebra!"):
            qv = self.ix.vectorize(q)
            self.assertEqual(cosine_scores(self.ix, qv), [0.0] * 4)
            self.assertEqual(pivoted_scores(self.ix, qv, 0.0), [0.0] * 4)
            self.assertEqual(search(self.ix, q), [])

    def test_term_in_every_document_has_zero_query_weight(self):
        ix = InvertedIndex([(1, "common alpha"), (2, "common beta"), (3, "common")])
        self.assertEqual(ix.idf["common"], 0.0)
        self.assertEqual(ix.norms, [math.sqrt(2), math.sqrt(2), 1.0])
        self.assertEqual(ix.vectorize("common"), {})
        self.assertEqual(search(ix, "common"), [])
        self.assertEqual(search(ix, "common alpha"), search(ix, "alpha"))

    def test_pivoted_scores_intermediate_slope(self):
        for q, dots in ((Q1, Q1_DOT), (Q2, Q2_DOT)):
            got = self.scores_by_id(pivoted_scores(self.ix, self.ix.vectorize(q), 0.5))
            for d, v in expected(dots, s=0.5).items():
                self.assertAlmostEqual(got[d], v, places=14)
        got = self.scores_by_id(pivoted_scores(self.ix, self.ix.vectorize(Q1), 0.5))
        self.assertAlmostEqual(got[1], 1.037611, places=6)
        self.assertAlmostEqual(got[2], 0.453501, places=6)
        self.assertAlmostEqual(got[3], 0.411554, places=6)

    def test_pivot_raises_long_lowers_short(self):
        qv = self.ix.vectorize(Q2)
        cos = cosine_scores(self.ix, qv)
        piv = pivoted_scores(self.ix, qv, 0.5)
        self.assertGreater(N1, P)
        self.assertLess(N2, P)
        self.assertGreater(piv[0], cos[0])
        self.assertLess(piv[1], cos[1])

    def test_s1_equals_cosine(self):
        for q in (Q1, Q2, "apple", "date cherry cherry", "zebra"):
            qv = self.ix.vectorize(q)
            self.assertEqual(pivoted_scores(self.ix, qv, 1.0), cosine_scores(self.ix, qv))
            self.assertEqual(search(self.ix, q, s=1.0), search(self.ix, q))

    def test_s0_gives_equal_factors(self):
        factors = self.ix.pivoted_factors(0.0)
        self.assertEqual(factors[:3], [self.ix.pivot] * 3)
        self.assertEqual(factors[3], 0.0)
        got = self.scores_by_id(pivoted_scores(self.ix, self.ix.vectorize(Q1), 0.0))
        for d, dot in Q1_DOT.items():
            self.assertAlmostEqual(got[d], dot / (NQ * P), places=14)
        # At s = 0 equal dot products tie exactly
        self.assertEqual(got[2], got[3])
        self.assertAlmostEqual(got[2], 0.414914, places=6)

    def test_slope_out_of_range_rejected(self):
        qv = self.ix.vectorize(Q1)
        for s in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                pivoted_scores(self.ix, qv, s)

    def test_ties_broken_by_document_id(self):
        self.assertEqual([d for d, _ in search(self.ix, Q1, s=0.0)], [1, 2, 3])
        self.assertEqual([d for d, _ in search(self.ix, Q2, s=0.0)], [3, 1, 2])

    def test_tie_break_independent_of_insertion_order(self):
        shuffled = InvertedIndex([DOCS[2], DOCS[3], DOCS[1], DOCS[0]])
        for q in (Q1, Q2):
            for s in (None, 0.0, 0.5):
                self.assertEqual(
                    [d for d, _ in search(shuffled, q, s=s)],
                    [d for d, _ in search(self.ix, q, s=s)],
                )

    def test_floating_point_noise_counts_as_tie(self):
        ix = InvertedIndex([(7, "x"), (5, "y")])
        self.assertNotEqual(0.1 + 0.2, 0.3)
        # Equal to 12 decimals, so the lower ID wins
        ranked = rank(ix, [0.3, 0.1 + 0.2])
        self.assertEqual([d for d, _ in ranked], [5, 7])
        self.assertEqual(ranked[0][1], 0.1 + 0.2)

    # q4 has no relevant documents and q5 has no labels
    QRELS = {"q1": {1, 3}, "q2": {2}, "q3": {1}, "q4": set()}
    QUERIES = {"q1": Q1, "q2": Q2, "q3": "the zebra", "q4": "apple", "q5": "cherry"}

    def run_queries(self, s):
        rankings = {qid: [d for d, _ in search(self.ix, q, s=s)] for qid, q in self.QUERIES.items()}
        return evaluate(rankings, self.QRELS, k=10)

    def test_metrics_cosine(self):
        ev = self.run_queries(None)
        per = {q.query_id: q for q in ev.per_query}
        self.assertEqual(list(per), ["q1", "q2", "q3"])
        self.assertAlmostEqual(per["q1"].precision, 2 / 10)
        self.assertAlmostEqual(per["q1"].recall, 1.0)
        self.assertAlmostEqual(per["q1"].f1, 1 / 3)
        self.assertAlmostEqual(per["q1"].average_precision, (1 / 1 + 2 / 3) / 2)
        self.assertAlmostEqual(per["q2"].precision, 1 / 10)
        self.assertAlmostEqual(per["q2"].f1, 2 / 11)
        self.assertAlmostEqual(per["q2"].average_precision, 1 / 2)
        self.assertEqual((per["q3"].num_retrieved, per["q3"].average_precision), (0, 0.0))
        self.assertEqual(per["q3"].f1, 0.0)
        self.assertAlmostEqual(ev.mean_precision, 0.1)
        self.assertAlmostEqual(ev.mean_recall, 2 / 3)
        self.assertAlmostEqual(ev.mean_f1, 17 / 99)
        self.assertAlmostEqual(ev.map, 4 / 9)

    def test_metrics_pivoted(self):
        self.assertAlmostEqual(self.run_queries(0.5).map, 4 / 9)
        ev0 = self.run_queries(0.0)
        per = {q.query_id: q for q in ev0.per_query}
        self.assertAlmostEqual(per["q2"].average_precision, 1 / 3)
        self.assertAlmostEqual(ev0.map, 7 / 18)

    def test_queries_without_relevant_documents_reported_separately(self):
        ev = self.run_queries(None)
        no_rel = {q.query_id: q for q in ev.no_relevant}
        self.assertEqual(set(no_rel), {"q4", "q5"})
        self.assertEqual(no_rel["q4"].num_retrieved, 2)
        self.assertTrue(no_rel["q4"].labelled)
        self.assertFalse(no_rel["q5"].labelled)
        self.assertEqual(len(ev.per_query), 3)


if __name__ == "__main__":
    unittest.main()
