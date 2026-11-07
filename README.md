# Correcting Length Bias: Pivoted Document Length Normalization

**Group 15** | Information Retrieval course project | Anushka Doshi, Ashutosh Desai

A TF–IDF retrieval engine that compares cosine normalization with pivoted cosine normalization on 100 English Press Club articles.

## Setup

Requires Python 3.10+.

```
pip install -r requirements.txt
```

## Running

```
python run_engine.py stats
python run_engine.py search "students union election commission" --k 10
python run_cranfield.py
python check_labels.py
python run_epc.py
python run_attacks.py
python plot_attacks.py
```

- `run_engine.py stats` shows statistics for the collection and the index.
- `run_engine.py search` ranks the articles for one query with both methods.
- `run_cranfield.py` chooses the pivoted slope on Cranfield and saves it to
  `frozen_slope.json`.
- `check_labels.py` checks the relevance labels and writes `qrels.csv`.
- `run_epc.py` compares cosine and pivoted on the EPC queries and writes `results/epc/`.
- `run_attacks.py` runs the attack experiments (keyword repetition, unrelated padding
  and both combined) and writes `results/attacks/`.
- `plot_attacks.py` draws the attack figures from `results/attacks/` into `docs/figures/`.

## Repository

```
irengine/
  preprocess.py, stopwords.py   tokenising, stopwords
  porter.py                     Porter stemmer
  index.py                      inverted index, weights, norms, pivot
  scoring.py                    cosine and pivoted scoring, ranking
  metrics.py                    P@10, R@10, F1@10, MAP
  data.py                       loading articles, queries and labels
  cranfield.py                  Cranfield data, query splits, slope choice
  significance.py               paired randomization test
  attacks.py                    attack targets, edits and rescoring
run_engine.py                   command-line runner
run_cranfield.py                slope selection on Cranfield
run_epc.py                      main EPC evaluation
run_attacks.py                  attack experiments
plot_attacks.py                 attack figures
check_labels.py                 label checks
articles.json, articles.csv     the 100 articles
queries.json                    the 12 queries with descriptions
results/llm_labels.jsonl        LLM relevance labels (Claude Opus 5.5)
results/llm_labels.csv          the same labels as a table
manual_labels.csv               labels reviewed by hand
qrels.csv                       working relevance labels
qrels_final.csv                 frozen labels used for evaluation
labels_frozen.json              checksum of the frozen labels
frozen_slope.json               the chosen slope
```
