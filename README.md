# multi-turn-rag-pipeline

A RAG pipeline for multi-turn chat that runs on a CPU. On every user turn it rewrites the question so it makes sense without the chat history, searches two indexes, reranks the hits, and then picks one of three moves. It answers from the documents, says it does not know, or asks the user which thing they mean.

I tested it on IBM's MTRAG benchmark. Dev is the MTRAG human set (842 tasks) and test is the public MTRAG-UN human set (507 tasks), the data behind SemEval 2026 Task 8. All models are small and open. The LLM is Qwen3.5-2B (4-bit GGUF on llama.cpp), and retrieval uses a learned sparse encoder and a cross-encoder with about 23M parameters each, both as int8 ONNX.

## Results

Everything here was measured with this repo on a 2 core Linux machine with no GPU. I picked every setting on dev.

### Retrieval

Scored on the tasks that have gold passages, 332 on test and 777 on dev.

Test:

| System | nDCG@5 | Recall@5 | nDCG@10 |
|---|---|---|---|
| BM25 on the last turn | 0.290 | 0.319 | 0.326 |
| Learned sparse on the last turn | 0.405 | 0.430 | 0.438 |
| Learned sparse on the Qwen rewrite | 0.440 | 0.476 | 0.485 |
| Fusion of 8 ranked lists | 0.511 | 0.552 | 0.563 |
| **Fusion and rerank (default)** | **0.517** | **0.575** | **0.564** |

For reference, SemEval 2026 Task 8 Subtask A ranked teams by nDCG@5 on a 507 task MTRAG-UN test set over the same four domains. The strongest organizer baseline (ELSER with GPT-OSS-20B rewrites) scored 0.480 and the winning team scored 0.578.

Dev:

| System | nDCG@5 | Recall@5 | nDCG@10 |
|---|---|---|---|
| BM25 on the last turn | 0.191 | 0.211 | 0.221 |
| BM25 on the Mixtral rewrite | 0.217 | 0.245 | 0.255 |
| Learned sparse on the last turn | 0.328 | 0.356 | 0.368 |
| Learned sparse on the Mixtral rewrite | 0.358 | 0.390 | 0.402 |
| Learned sparse on the Qwen rewrite | 0.356 | 0.387 | 0.399 |
| Fusion of 8 ranked lists | 0.375 | 0.409 | 0.431 |
| **Fusion and rerank (default)** | **0.429** | **0.467** | **0.474** |

The Mixtral rewrites are the rewrite files that ship with the data, which the MTRAG paper made with Mixtral 8x7B. The BM25 rows line up with Table 3 of the MTRAG paper, which has 0.20 Recall@5 and 0.21 nDCG@10 for the last turn and 0.25 and 0.25 with those rewrites. The rewrites from the 2B model work about as well as the Mixtral ones. The best retriever in that paper, ELSER with rewrites, gets 0.52 Recall@5 and 0.54 nDCG@10 on dev.

### Answers

Answers are scored with MTRAG's RB_alg, the harmonic mean of BERTScore recall against the reference answer, ROUGE-L against the reference, and BERTScore precision against the passages. The score is conditioned on answerability. An answerable task gets 0 if the reply is an I don't know. An unanswerable task gets 1 for an I don't know and 0 for anything else, and an underspecified task gets 1 only for a clarifying question.

Test, all 507 tasks:

| Answerability | Tasks | Pipeline | Gate off |
|---|---|---|---|
| Answerable | 285 | 0.372 | 0.373 |
| Partial | 47 | 0.220 | 0.229 |
| Unanswerable | 97 | 0.485 | 0.443 |
| Underspecified | 78 | 0.141 | 0.026 |
| **All** | **507** | **0.344** | **0.320** |

Gate off is the same run with `gate.enabled = false`, so every turn gets an answer. The two columns only differ on the 30 test turns where the clarify step fired, since the fitted gate never refuses. Those turns lift underspecified tasks from 0.026 to 0.141. They also lift unanswerable tasks from 0.443 to 0.485, because the I don't know judge counted 9 of the 10 clarifying replies there as an I don't know. Answerable tasks barely move (0.373 to 0.372). When the bot does answer an answerable or partial question, RB_alg averages 0.408.

What the pipeline did on test, with I don't know as judged by the local judge:

| Answerability | Answered | I don't know | Clarify |
|---|---|---|---|
| Answerable | 254 | 23 | 8 |
| Partial | 28 | 16 | 3 |
| Unanswerable | 49 | 38 | 10 |
| Underspecified | 55 | 14 | 9 |

For a rough sense of scale, the MTRAG-UN paper reports conditioned RB_alg of 0.31 for Granite-4.0-tiny and 0.38 for Llama-3.3-70B in its RAG setting. That is on its own 666 tasks over six domains, so it is not the same task set.

## How it works

1. Rewrite. Qwen3.5-2B reads the last six messages and writes JSON with a standalone question, a keyword query and a flag for questions that are too vague to answer. In one test chat the user names the Gordon D. Schaber courthouse and then asks "Is there a Small Claims Court?", which becomes "Is there a Small Claims Court at the Gordon D. Schaber Sacramento County Courthouse?". I take the probability of the vague flag from the token logprobs, because the plain true or false came out true for almost a quarter of all questions.
2. Retrieve. Four queries go to two indexes. The queries are the last turn, the last three user turns weighted 1, 0.5 and 0.25, the rewrite, and the keywords. The indexes are BM25 and a learned sparse index. The sparse model only runs on documents, and query tokens are weighted by idf, so a search needs no model call. The 8 ranked lists are merged with weighted reciprocal rank fusion, with the BM25 lists at a quarter of the weight.
3. Rerank. The top 30 go through the cross-encoder. The final order mixes the rerank score and the fusion score at 0.8 and 0.2, after z-scoring both.
4. Decide. If the vague probability is above the 95th percentile of dev, the bot asks which thing the user means. Otherwise a small logistic regression over retrieval features and a one-token yes or no probe from the LLM decides between answering and saying I don't know. Its cutoff is the one with the best expected score on dev. The gate only reached 0.59 AUC in cross validation, so no cutoff beat always answering, and the fitted gate stays open. The I don't know replies come from the LLM itself, which is told to say so when the passages do not cover the question.
5. Answer. Each of the top 5 passages gives the sentences that share the most rare words with the rewrite, 600 tokens in total, and the LLM answers in under 150 words. Citations are added afterward by matching answer sentences to passages.

The API is FastAPI with server-sent events. The chat page first shows what the bot searched for and which move it picked, then streams the reply.

## Run it

You need Python 3.11 or newer and a llama.cpp build that has `llama-server`.

```
git clone https://github.com/adwitiyashukla/multi-turn-rag-pipeline
cd multi-turn-rag-pipeline
pip install -e ".[export]"
```

To try one domain first, set `collections = ["fiqa"]` in config.toml. Then download the data and models, export the sparse encoder and build the indexes.

```
turnrag fetch
turnrag models --export
turnrag export
turnrag index
```

Start the LLM in a second terminal.

```
llama-server -m models/qwen3.5-2b/Qwen3.5-2B-Q4_K_M.gguf --port 8080 --jinja --alias qwen3.5-2b -c 16384 -np 2
```

Then start the API and open http://127.0.0.1:8000 for the chat page.

```
turnrag serve
```

`turnrag chat --collection fiqa` does the same in the terminal. On my laptop (Intel Core 5 120U, on battery) a follow-up turn took 33 seconds end to end, of which the rewrite took 6, search and rerank 9, and the answer 10. Every setting is in config.toml, and any of them can be overridden with an environment variable such as `TURNRAG__RERANK__BLEND=0.3`.

To rerun the evaluation, keep all four collections, keep llama-server running and run these in order.

```
pip install -e ".[export,eval]"
turnrag models --export --eval
turnrag rewrites --split dev
turnrag rewrites --split test
turnrag retrieval --split dev
turnrag retrieval --split test
turnrag answers --split dev --sample 100
turnrag judge --split dev
turnrag score --split dev --bf16
turnrag gate
turnrag answers --split test
turnrag judge --split test
turnrag score --split test --bf16
turnrag check-judge --sample 150
turnrag check-metrics --sample 60 --bf16
```

`answers --split dev --sample 100` takes 100 answerable dev tasks (every 7th) plus all 65 that are not answerable. `score` runs the 750M DeBERTa model behind BERTScore, so it is the slow step on a CPU.

## How I checked the numbers

- Retrieval metrics. A test compares my nDCG and Recall code with pytrec_eval on random runs, and they agree exactly. The BM25 baselines above also land within about 0.01 of the MTRAG paper.
- RB_alg. `check-metrics` rescores 60 model responses that IBM already scored in their analytics file and compares each part with IBM's value. ROUGE-L matches exactly, BERTScore recall is off by 0.0015 on average, and 59 of the 60 K-precision values are within 0.02. Over the 57 answered responses, RB_alg is off by at most 0.033.
- Judges. The I don't know judge and the clarification judge use IBM's exact prompts, read from their repo files at runtime, but they run on the local 2B model instead of IBM's judge. `check-judge` compares the local I don't know judge with IBM's labels on 150 responses. They agree on 87% of them (Cohen's kappa 0.41), but the local judge calls 18% of the replies an I don't know where IBM's calls 7%. So my unanswerable scores lean high and my answerable scores lean low.
- Reruns. Running the retrieval step again gives the same numbers to the last digit, and on my Windows laptop the 32 tests pass and the sparse encoder export comes out byte for byte the same as on Linux (same SHA-256).
- Dev only. The fusion weights, the rerank blend, the vague cutoff and the gate were all picked on dev.

## Project layout

```
config.toml              every setting, validated in config.py
gate.json                the gate fitted on dev
src/turnrag/
  config.py              loads config.toml and TURNRAG__ environment overrides
  data.py                pinned downloads, corpus store, task loading
  text.py                tokenizing, stemming, sentence splitting
  index.py               impact index and BM25
  encoders.py            ONNX Runtime wrappers for the sparse encoder and the reranker
  export.py              exports the sparse document encoder to int8 ONNX
  build.py               resumable index build
  llm.py                 llama-server client with a response cache
  rewrite.py             query rewriting and the vague probability
  retrieve.py            fusion and reranking
  answer.py              gate features, probe, evidence, prompts, citations
  pipeline.py            one chat turn, plain or streamed
  metrics.py             retrieval metrics and the conditioned score
  scoring.py             IBM's judge prompts and RB_alg
  evaluate.py            evaluation runs and reports
  api.py                 FastAPI app with server-sent events
  static/index.html      chat page
  cli.py                 the turnrag command
tests/                   pytest, runs in CI
```

## Data and models

| What | Used for | License |
|---|---|---|
| [MTRAG](https://github.com/IBM/mt-rag-benchmark) | corpora, dev and test tasks, judge prompts | Apache 2.0 |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), Q4_K_M GGUF by unsloth | rewrite, probe, answer, judges | Apache 2.0 |
| [opensearch-neural-sparse-encoding-doc-v2-mini](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini) | learned sparse index | Apache 2.0 |
| [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | reranker | Apache 2.0 |
| [deberta-xlarge-mnli](https://huggingface.co/microsoft/deberta-xlarge-mnli) | BERTScore inside RB_alg, evaluation only | MIT |

Every download is pinned to a commit and checked against a SHA-256 in data.py.
