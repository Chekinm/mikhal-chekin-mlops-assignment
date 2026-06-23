# LLM Inference + Observability — Report

Text-to-SQL PoC: Qwen3-30B-A3B served on 1× H100 via vLLM, with a LangGraph
verify→revise agent on top. 
Stack: vLLM + Prometheus/Grafana for serving o11y,
LangGraph + Langfuse for agent o11y.

---

## 1. Serving configuration (Phase 1)

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, 30B total / 3B active). Hardware: 1× H100 80GB.

| Flag | Value | Rationale |
|---|---|---|
| `--max-model-len` | 8192 | Prompts are 1.5–3K tokens (schema + question) with short SQL output; 8K leaves headroom without wasting KV. I set this parameter to 16384 initially to fit long answer, but finaly after soem fix in agent logic and adding some restirction for big shcema I relized that we don't need such a big context, so reduce it back to 8192.|
| `--max-num-seqs` | 64 | Each agent run fans out to ~2–3 dependent vLLM calls; 64 covers 10 RPS × fan-out. Confirmed sufficient: queue stayed empty under load. Played with this number during experiements. Making it smaller  affect the performance, while 32 to 64 make quite a small difference. Just suggest to stay at 64.|
| `--max-num-batched-tokens` | 4096 | [VERIFY rationale — chunked-prefill batch budget] |
| `--gpu-memory-utilization` | 0.90 | Leaves ~10% for CUDA kernels / torch overhead. |
| `--enable-prefix-caching` | on | Schema prefix is identical across calls to the same DB → prefill cache hits. O tried to organzie prompts so that it will have longest common part (including DB schema right after system prompt). Confirmed: TTFT settled to ~80ms under load. |
| `--enable-chunked-prefill` | true | [VERIFY — overlaps prefill with decode to smooth latency] |
| `--disable-log-requests` | on | Reduces CPU overhead under load. |

**Dashboard-grounded justification.** Under the 10 RPS load test, queue depth stayed
at 0, KV-cache utilization peaked at ~5–7%, and TTFT settled around 80ms. This shows
concurrency (`max-num-seqs`) and memory (`max-model-len`) were *not* the constraint —
the chosen flags carry comfortable headroom for this workload.

---

## 2. Baseline eval results (Phase 5)

Eval signal: execution accuracy — agent's final SQL vs gold SQL, comparing
canonicalized row sets over 30 BIRD questions.

| Metric | Value |
|---|---|
| Overall accuracy | 36.7% (11/30) |
| Pass rate after iter 0 | 33.3% |
| Pass rate after iter 1 | 36.7% |
| Pass rate after iter 2 | 36.7% |

**Commentary.** With implemented agent flow. the revise loop barely earns its keep: iter 1 adds exactly one correct
answer (+3.3pp), iter 2 adds zero. The architecture is doing minimal work on this set.
So as you will see later we set max iteration to 2 as we not 3 stpe run achive any better results, while make the P99 latency significanly slow.

---

## 3. Hitting the SLO (Phase 6)

**Target:** P95 end-to-end agent latency < 5s at 10+ RPS over a 5-minute window.

**Baseline (MAX_ITERATIONS=3):**

| Metric | Value |
|---|---|
| Achieved RPS | 9.52 |
| P50 | 1.85s |
| P95 | 9.39s |
| P99 | 14.08s |

P95 was ~1.9× over the SLO. Profile: good median, heavy tail.

### Diagnosis

Dashboard under load: **queue depth = 0** (requests scheduled immediately),
**KV cache ~5%** (no eviction pressure), **TTFT ~80ms** (fast prefill, prefix cache
working). Conclusion: the bottleneck is **not** the serving layer. The tail comes from
**number of vLLM calls per agent run** — a request that goes the full N iterations
stacks N call latencies. Cross-confirmed by eval: iter 2 adds no accuracy, so the 3rd
iteration is pure latency cost. More imporvments can come from agent imporving. 
We need to realize how to ahndel huge SCHEMA. Probably to sue some small and fast model to optimize the schema which we feed to the main model to generate SQL.



### Iteration log

I made quite a lot of iterations. 
It includes vLLM parameters tuning, LLM call in agent pararamter tuning. Prompt tuning. Some bags fix of the code suggested in the task.
The metrics a bit mixed ans we start with 5RPS and only final runs was raned with 10 RPS. 

** Baseline **
```
  "summary": {
    "requested_rps": 5.0,
    "duration_seconds": 120,
    "wall_clock_seconds": 163.27163717700023,
    "total_requests": 600,
    "achieved_rps": 3.674857497444883,
    "ok": 506,
    "timeouts": 2,
    "http_errors": 77,
    "client_errors": 15,
    "latency_p50": 4.205336914999862,
    "latency_p95": 22.046403835000092,
    "latency_p99": 30.513695796999855,
    "latency_max": 83.0203100409999
  },
```
We see here 2 timeout and 77 https error. Which is not good at all. 
Most of HTTP error was casued the bug in thsi funcion. It hadn't harness for None value in the begning. So it jsut fails. Adding default for None value get rid of the HTTP errors.
```python
def _q(ident) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    if ident is None:
        return "NULL"
    return '"' + ident.replace('"', '""') + '"'
```
The base accuray is 
```
"summary": {
    "total": 30,
    "overall_accuracy": 0.4,
    "iter_pass_rates": {
      "iter_0": 0.3333,
      "iter_1": 0.3667,
      "iter_2": 0.4
    }
  },  
```

**Iteration #1 - #9.**

After fix of the python code I still run into timeout problem. Which gives some agent call to run like 3 to 4 minutes. Investigation and debugging lead me to this three solutions. 
1. Trancate huge DB schemas (expesialy the football one)
2. Trancate long lines to 4000 characters
3. One of the DB was tended to produce long-long SQL on the first stroke, which case out of the context errro, and timeoutn in the same time. 
Which was repeting AND ... NOT LIKE" repetition loop
Adding frequency_penalty=0.5 and presence_penalty=0.3 I handle this case. 
So after this I end up with something like this, with proud zeros near timeout and errors.
```
  "summary": {
    "requested_rps": 5.0,
    "duration_seconds": 120,
    "wall_clock_seconds": 125.6342896770002,
    "total_requests": 600,
    "achieved_rps": 4.775766246162346,
    "ok": 600,
    "timeouts": 0,
    "http_errors": 0,
    "client_errors": 0,
    "latency_p50": 1.4641172829979041,
    "latency_p95": 12.699359417001688,
    "latency_p99": 20.220609576001152,
    "latency_max": 32.14765773399995
  },
```
**Iteration #10 - #14.**
So after fixeing the flow I start to optimizing the preformance.
I check how changeing --max-num-seqs affect the timings. 
So that setting smallnumbers like 16, we have our calls queued whihc increase the time even for P50 to 5 sec.
Stop at 64. 
Then I play with lenght of the verify_node LLM call. 
Originally all call to llm has 1024 max_len limit. 
Seting it to 128 for verify call (which is simple ok not ok, with soem explanation). Reduce overall timeing significantly. 
```"summary": {
    "requested_rps": 5.0,
    "duration_seconds": 120,
    "wall_clock_seconds": 122.71593655599645,
    "total_requests": 600,
    "achieved_rps": 4.8893405114193484,
    "ok": 600,
    "timeouts": 0,
    "http_errors": 0,
    "client_errors": 0,
    "latency_p50": 1.2930918269994436,
    "latency_p95": 7.154788054001983,
    "latency_p99": 9.855944494996947,
    "latency_max": 14.0938160279984
  },
```

**Iteration #15 - #20.**
Here I decided to chanlange to 10RPS loads. So the values getting sligly wrost.
In this step we the most improvments comes from runing uvcorn on 4 workers.
And reducing --max-model-len to 8192 (which was set to 16384 during fighting with timeouse in the begning)
This to thing elads us to thi numbers

```
"summary": {
    "requested_rps": 10.0,
    "duration_seconds": 120,
    "wall_clock_seconds": 127.4737918030005,
    "total_requests": 1200,
    "achieved_rps": 9.4136997340951,
    "ok": 1200,
    "timeouts": 0,
    "http_errors": 0,
    "client_errors": 0,
    "latency_p50": 3.26914804399712,
    "latency_p95": 13.013870126997062,
    "latency_p99": 17.863356554000347,
    "latency_max": 29.389899894002156
  },
```

**Iteration #21**

Here I found missing slash in model configuraiton. 
Bascily we work without chucked_preffill. 
So seting it to true add reduce the latency by 1.5 seconds. 

```
"summary": {
    "requested_rps": 10.0,
    "duration_seconds": 120,
    "wall_clock_seconds": 126.04410947900033,
    "total_requests": 1200,
    "achieved_rps": 9.520476640758265,
    "ok": 1200,
    "timeouts": 0,
    "http_errors": 0,
    "client_errors": 0,
    "latency_p50": 1.8494768579985248,
    "latency_p95": 9.38669773500078,
    "latency_p99": 14.07742765200237,
    "latency_max": 26.42177825100225
  },
```
I suggest this is the final number.

Here I run the final evalutaion of the agent.
```
  "summary": {
    "total": 30,
    "overall_accuracy": 0.3667,
    "iter_pass_rates": {
      "iter_0": 0.3333,
      "iter_1": 0.3667,
      "iter_2": 0.3667
    }
  },
```
Which get's us understanding the no 3 run imporve the results. While third run of the loop affects the latency. 

I run on more load test. with max_iteration=2. And get this final number for latency. Which is quite close to goal 5 sec. But I reach this by cutting the number of loops to 2. Which is not fair  by my opinion.
```
"latency_p50": 1.670117248002498,
"latency_p95": 7.074469617000432,
"latency_p99": 10.450049170001876,
```


### Final result
I consider as a initial point the first run with 10RPS. So real improvment is a bit higher.

| Metric | Baseline | Final | SLO |
|---|---|---|---|
| P95 | 10.35s | **7.07s** | < 5s |
| P99 | 20.13s | 10.45s | — |
| Achieved RPS | 8.97 | 9.73 | 10+ |

**Verdict: SLO missed by ~2s on P95.** The gap is loop-bound, not serving-bound:
serving has headroom (empty queue, 5% KV), but each revise round adds a full vLLM-call
latency. Closing the remaining gap requires reducing the *rate* of revise rounds
(better first-pass generation), not tuning vLLM flags.


---

## 4. Agent value

The verify→revise loop added exactly one correct
answer out of 30 (iter 0: 33.3% → iter 1: 36.7%, iter 2: +0) at the cost of roughly
doubling the latency tail (P95 driven by multi-iteration runs). On this workload the
loop is a poor trade: marginal accuracy gain, significant latency cost. It earns its
keep only if first-pass generation quality improves enough that revise fires on genuine
defects rather than on verifier false-positives.
Need aditional work on agent logic itself to imporve first generation, and imporve working with massice SHEMAS, which affets timing significantly.

---

## 5. What I'd do with more time

As it was discussed. Som approach left behind.
- **Few-shot exemplars** in `generate_sql`: 2–3 schema-matched SQL examples to cut
  hallucinated columns (model invented `totalSetSize`) and reduce revise rounds.
- **Schema linking / retrieval:** surface only relevant tables+columns per question
  instead of the full schema — less context, fewer wrong-table joins. Less token to feed ot the model
- **Two-model split:** a stronger/larger model for verify (catch real defects) and a
  fast model for generate — better revise signal without latency on the common path.
- **Fix wasted revise rounds:** tighten verify so it stops sending valid-but-empty
  results to revise — double win, raises accuracy *and* lowers latency (fewer iterations).
---

## Appendix — known issues

- `/answer` returns `ok: true` whenever the SQL executes, even if the final result is
  empty and verify was unhappy. `ok` reflects execution success, not verify verdict.
  Eval is unaffected (it re-executes `sql` and compares row sets, ignoring `ok`).