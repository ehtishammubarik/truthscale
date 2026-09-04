# truthscale

**Your GPU is not 90% busy. Here is the real number.**

![Status](https://img.shields.io/badge/status-alpha-orange)
![Go](https://img.shields.io/badge/go-1.27-00ADD8?logo=go&logoColor=white)
![Dependencies](https://img.shields.io/badge/runtime%20dependencies-none-success)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

`nvidia-smi` reports "GPU utilization". NVIDIA defines it as *the percent of time
over the sample period during which one or more kernels was executing*. It is a
time-based occupancy flag, not a measure of work.

One small kernel, on 1 of an H100's 132 streaming multiprocessors, running for
the whole window, reads **100%**. The other 131 sit idle. Your dashboard says
"fully utilized".

Autoregressive LLM decode is the pathological case. It launches one small kernel
per token, so the flag pins at 100% while SM occupancy sits near 15% and the
tensor cores are close to idle. Production Kubernetes clusters average around
**5% real GPU utilization** while their dashboards read healthy, and KEDA and
Karpenter then autoscale on the number that caused the confusion.

## Install

```bash
go install github.com/ehtishammubarik/truthscale/cmd/truthscale@v0.1.0
```

Go 1.27 or newer. No runtime dependencies, no CGO, one static binary. Or clone
the repo and run the commands below with `go run` against the committed traces.

## Try it without a GPU

Every command works against a recorded trace, so you can form an opinion in
thirty seconds on a laptop:

```bash
go run ./cmd/truthscale top --replay traces/mixed-fleet.jsonl --last
```

```
NODE         GPU  VERDICT    UTIL  OCCUPANCY  GAP  CEILING  KV CACHE  QUEUE  BATCH
gpu-node-01  0    STARVED    99%   13%        +86  n/a      8%        0      1
gpu-node-02  0    BUSY       97%   86%        +11  n/a      63%       2      48
gpu-node-03  0    BLOCKED    99%   18%        +81  n/a      97%       34     12
gpu-node-04  0    THROTTLED  96%   41%        +55  n/a      -         -      -
gpu-node-05  0    UNKNOWN    87%   n/a        n/a  n/a      -         -      -

5 GPU(s): 1 starved, 1 blocked, 1 throttled, 1 busy, 1 unknown

1 GPU(s) report high utilization while the SMs are mostly idle.
Run `truthscale explain` on one of them to see why.

1 GPU(s) could not be judged because a counter was unavailable.
These are reported as unknown rather than assumed healthy. See docs/metrics.md.
```

Five GPUs, five different truths, and one of them admits it does not know.

## Why the verdicts differ

```bash
go run ./cmd/truthscale explain --replay traces/h100-vllm-decode.jsonl
```

```
gpu-node-01 gpu0 (NVIDIA H100 80GB HBM3): STARVED

  what the dashboard says   utilization 98%
  what the SMs say          occupancy   15%
  the gap                   83 points of apparent busyness that is not work
  against measured ceiling  n/a (no trustworthy ceiling for this node yet)

  - utilization reads 98% while only 15% of the SMs are occupied: a kernel is resident, the machine is mostly idle
  - the batch is 1: one small kernel per token keeps utilization pinned while the tensor cores idle
  - the KV cache is only 8% used, so there is room to batch more work onto this GPU
```

That last section is the point. `gpu-node-03` also reads 99% utilization, and it
is **not** starved: its KV cache is 97% full with 34 requests queued, so it is
fenced off rather than underused. Adding a GPU raises capacity; raising the cache
or shortening the max sequence length may be cheaper. Those two nodes look
identical on every dashboard in common use.

## The metric that matters

`ceiling_ratio`: **delivered divided by this node's measured ceiling.** It is
the only number here that answers "should this node exist?", which is the
question a GPU fleet is actually asking.

Measured, never from a datasheet. A datasheet figure is taken at a clock, a
power limit, and a thermal envelope you probably do not have, so a ratio against
it is wrong in an unknown direction. In this release no ceiling is established
yet, so the column reads `n/a` rather than guessing. Establishing it is
[v0.2](ROADMAP.md).

## What it reports

The pairing is the argument. Each row is the number people watch, next to the
number that means something.

| Commonly watched | truthscale reports | Source |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | `sm_occupancy` | `DCGM_FI_PROF_SM_OCCUPANCY` |
| GPU memory used | `dram_active`, bandwidth actually moved | `DCGM_FI_PROF_DRAM_ACTIVE` |
| "it is busy" | `tensor_active` | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` |
| requests/sec | `output_tokens_per_sec`, `batch_size` | vLLM `/metrics` |
| nothing | `kv_cache_utilization` | vLLM `/metrics` |
| nothing | `queued_requests`, `queue_seconds` | vLLM `/metrics` |
| nothing | `throttle_reason` | `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` |
| nothing | **`ceiling_ratio`** | derived |

Full definitions, units, and the support matrix: [docs/metrics.md](docs/metrics.md).

## Four rules it will not break

These are enforced in code and in review, not encouraged in a doc.

1. **A measurement that was not taken is never rendered as zero.** An
   unavailable counter is `n/a`, and the reason is recorded. An idle GPU and an
   unsupported driver must not look identical, and they do on most dashboards.
2. **No throughput decline is attributed to software before power and thermal
   throttling are ruled out** from the sample taken at that same instant. A
   throttled GPU short-circuits every other verdict.
3. **Raw records, not averages.** Samples are written as taken. Percentiles are
   computed on read, never at capture time.
4. **Measured and spec-sheet figures never share a column.**

## Status

**Alpha. v0.1.0 is tagged and pinnable.** It reads traces, computes the honest
signal, and explains it. The one live path is the vLLM collector, which reads a
serving endpoint's `/metrics` for KV cache utilization, queue depth, and token
throughput. The GPU side is still trace-only, so `--replay` is required on `top`
and `explain`; running without it prints why rather than returning an empty
table that would let someone conclude their fleet is fine.

Live DCGM collection, a ceiling measurement, the DaemonSet and Helm chart, and
the scaling decision engine are [v0.2 and v0.3](ROADMAP.md), in that order and
for a reason: nobody should act on a signal they have not learned to trust.
[CHANGELOG.md](CHANGELOG.md) records what each tag changed.

## Prior art, and what is different

[NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator),
[DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter), and NVIDIA's GPU Usage
Monitor all collect the right raw fields and are the correct foundation. This
reads DCGM rather than reimplementing it.

What none of them do is pair the misleading number with the honest one, express
the gap as a fraction of measured capability, or explain in words why the two
differ. That gap is discussed in a great many blog posts and shipped by nobody,
which is the entire reason this exists.

If you want scheduling and queueing, use [Kueue](https://github.com/kubernetes-sigs/kueue)
or [Volcano](https://github.com/volcano-sh/volcano). This is not a scheduler.

## Traces are synthetic, and say so

The traces in `traces/` are generated by `cmd/gen-traces` from the published
behaviour of decode and the counter semantics in `docs/metrics.md`. They contain
no data from any real deployment. The generator is committed and deterministic,
so you can see exactly how each shape was chosen and regenerate them yourself:

```bash
go run ./cmd/gen-traces && git diff --exit-code traces/
```

CI runs that, so a trace cannot drift from its generator.

## Build

```bash
go build ./... && go test ./...
```

No runtime dependencies, no CGO, one static binary.

## License

Apache 2.0. See [LICENSE](LICENSE).

---

If truthscale changed a number you were about to act on, a star helps the next
person find it. Contributions welcome, and [ROADMAP.md](ROADMAP.md) says which
work is load-bearing.
