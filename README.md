# Diagnosing RAG retrieval failures: a working log

A small local RAG pipeline, measured and diagnosed on NeoQA, a fully fictional
news dataset. One part per completed phase. A lab log, not a product.

## Where RAG retrieval is known to break

Known problems, all of which this project hits:

- **Multi-hop questions.** Retrieval finds THE most relevant passage. A
  question needing two documents usually gets one.
- **Exact words underused.** The name that pinpoints the right document may
  not win the embedding score. Hence hybrid search.
- **Near-duplicates.** Reworded copies of the same content fill the top ranks
  and starve everything else.
- **Contaminated benchmarks.** Strong LLMs have memorized the public corpora;
  many benchmarks are answerable with retrieval switched off.

## What this project does

Build a deliberately simple pipeline: one open embedder, each article embedded
whole in one pass, cosine top-10. Measure it on clean data. Diagnose the
failures from evidence before picking a fix. Then build the fix and count how
many failures come back.

## The parts

| part | what it covers | status |
|---|---|---|
| [Part 1: Diagnosing retrieval failures](part1-diagnosing-retrieval-failures.ipynb) | The honest failure count (216 becomes 171), which properties separate failures from successes, a measurement that cheated, three fix previews | published |
| [Part 2: Why the failures fail](part2-why-the-failures-fail.ipynb) | Reading all 171 failures and naming six causes; two checks (counting each cause over the successes, and counterfactuals) that killed one guess and shrank another; which fix each cause points to | published |
| Part 3: Counts and payoffs | How often each cause occurs; what each fix would recover | planned |
| Part 4: The fix | Keyword-mix built and measured against the 171 failures | planned |

## Results so far (Part 1)

Setup: NeoQA dev, 266 answerable multi-hop and time-span questions,
`gte-modernbert-base` embedder (149M parameters, long-window), whole articles
as the retrieval unit, top-10.

- **Real failures: 171, not 216.** The naive count demanded one official
  evidence pair; the dataset accepts many. A fifth of the "failures" were
  scoring artifacts.
- **The strongest signal is a keyword gap.** In failures, plain word overlap
  (BM25) ranks the needed article ~21 places better than the embedder. In
  successes, no gap.
- **Near-duplicates bury the misses.** Failures: ~2.1 near-copies above the
  needed article. Successes: ~0.7.

Fix previews, measured on the 171 failures:

| fix | rescues |
|---|---|
| BM25 keyword ranking | 101 (59%) |
| an embedder 4x bigger | 34 (20%) |
| a cross-encoder re-ranker | 62 (36%) |

A stronger model is not the lever. The keyword signal is.

## Results so far (Part 2)

Same pipeline. We read all 171 failures in shuffled order, turned each
suspected cause into a frozen rule, and labeled every question -- successes
included -- so a cause has to separate failures from successes to count.

Six causes came out: the question leans to one of its two articles; a wrong
near-identical event captures the words; the needed fact drowns in a
whole-article vector; the question never names its second article; a few
broken questions; and near-copies crowd the top-10.

Two checks tested each one. Counting each cause over the successes killed
one candidate outright. A counterfactual -- remove the cause and re-score --
showed the most common one, near-copy crowding, fixes far fewer failures
than it appears in. Co-occurrence is not cause.

The numbers, one worked example, and which fix each cause points to are in
the notebook.

## Why NeoQA (and why there is no data here)

The first two candidate datasets were contaminated. On MultiHop-RAG (real
news), Llama-3.3-70B scored 0.803 with retrieval switched off, 0.895 with it:
it answers from memory, so the benchmark cannot measure retrieval. CLAPnq
(Wikipedia-based) had the same disease: ~52% of its dev questions answered
with no evidence.

NeoQA, by Amazon Science, is a fully fictional generated newswire: 1,800
articles on invented timelines, 7,348 questions. Nothing in it can be answered
from memory. It ships encrypted, on purpose, to stay out of training crawls:
[github.com/amazon-science/NeoQA](https://github.com/amazon-science/NeoQA) /
[huggingface.co/datasets/mglockner/neoqa](https://huggingface.co/datasets/mglockner/neoqa) /
[arXiv:2505.05949](https://arxiv.org/abs/2505.05949) (Findings of ACL 2025).

The dataset is CC BY-ND, and we keep it out of crawls too. This repo has **no
dataset files and no decrypted text**: aggregate numbers, our own code
excerpts, our own words. Question examples are paraphrased, never quoted. To
work with the data, use the dataset's own repo, which documents the
decryption.

## The models used

Two kinds of models appear here. Don't mix them up:

- **Under test:** open embedders (`gte-modernbert-base` as the pipeline's
  embedder, `Qwen3-Embedding-0.6B` as a stronger comparison), a cross-encoder
  re-ranker, and Llama-3.3-70B as the answer model. Each part states which
  ones it used.
- **Doing the work:** the code, analysis, and text were built in
  [Claude Code](https://claude.com/claude-code) sessions with Anthropic's
  **Claude Opus 4.8** and **Claude Fable 5**, directed and reviewed by the
  project's operator.

## What this repo is not

- Not a framework, not a runnable pipeline. The notebooks are **reports,
  written to be read**: code excerpts with real outputs saved in, so
  everything renders on GitHub with no GPU and no data.
- Not finished research. Parts publish as work completes, and a later part may
  revise an earlier one. That's the point of the log.
