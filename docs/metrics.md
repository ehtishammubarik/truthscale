# Metrics: every field, its source, and what it does not tell you

Every number here is traceable to a DCGM field, a vLLM metric, or a formula in
this file. Nothing is estimated, and nothing is filled in when the source did
not answer.

## The pair the project exists for

### `gpu_util` — the misleading one

- **Source:** `DCGM_FI_DEV_GPU_UTIL`, equivalently `nvidia-smi --query-gpu=utilization.gpu`
- **Units:** fraction, 0 to 1
- **NVIDIA's definition:** the percent of time over the past sample period
  during which one or more kernels was executing on the GPU

**What it does not tell you:** how much of the GPU was used. It is a
time-based occupancy flag. One kernel occupying 1 of an H100's 132 SMs for the
whole window reads 1.0. This field is reported here because people watch it and
because the gap against occupancy is the finding, not because it is useful
alone.

### `sm_occupancy` — the honest one

- **Source:** `DCGM_FI_PROF_SM_OCCUPANCY`
- **Units:** fraction, 0 to 1
- **Meaning:** the fraction of resident warp slots actually occupied, averaged
  over the SMs and the sample window

**What it does not tell you:** whether the occupied warps are doing useful
arithmetic. A memory-bound kernel can hold high occupancy while stalled on DRAM.
Read it with `dram_active` and `tensor_active`.

**Requires DCGM profiling**, which needs a recent driver and is not available on
every GPU. When absent it is reported absent.

## Hardware fields

| Field | DCGM source | Units | What it does not tell you |
|---|---|---|---|
| `sm_active` | `DCGM_FI_PROF_SM_ACTIVE` | fraction | Whether the active SMs are efficient. High `sm_active` with low `sm_occupancy` means few warps per SM |
| `tensor_active` | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | fraction | Anything about non-tensor work. A correct FP32 workload legitimately reads near zero |
| `dram_active` | `DCGM_FI_PROF_DRAM_ACTIVE` | fraction | Whether the bandwidth was usefully spent |
| `memory_used_bytes` | `DCGM_FI_DEV_FB_USED` | bytes | Anything about activity. A framework caching allocator holds memory it is not using, so this is close to useless as a load signal |
| `power_watts` | `DCGM_FI_DEV_POWER_USAGE` | watts | Efficiency. Useful mainly against `power_limit_watts` to spot a cap |
| `temp_celsius` | `DCGM_FI_DEV_GPU_TEMP` | Celsius | Whether throttling is happening. Check `throttles` for that |
| `sm_clock_mhz` | `DCGM_FI_DEV_SM_CLOCK` | MHz | Read against the boost clock: a large shortfall means capability is being withheld |
| `throttles` | `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` | flags | Nothing about software. It is what has to be excluded before blaming software |

## Serving fields

Present only when an inference server is on the GPU. Absent for training and for
a bare node, which is normal and not a failure.

**Source:** vLLM's Prometheus endpoint, scraped by `internal/vllm`. The endpoint
is configurable; `http://127.0.0.1:8000/metrics` is vLLM's default. TGI and
Triton expose the same shape under different names and are not read yet.

Three of these are **derived from counters and are absent on a first scrape**,
because a counter carries no rate until it has been observed twice. See
[Rates and why the first scrape is empty](#rates-and-why-the-first-scrape-is-empty).

| Field | vLLM metric | Units | Why it matters |
|---|---|---|---|
| `kv_cache_utilization` | `vllm:gpu_cache_usage_perc` | fraction | Explains the paradox. A full cache blocks admission while a resident kernel pins `gpu_util` at 1.0 |
| `running_requests` | `vllm:num_requests_running` | count | The real concurrency, as against apparent busyness |
| `queued_requests` | `vllm:num_requests_waiting` | count | Work that exists and cannot start |
| `queue_seconds` | `vllm:request_queue_time_seconds` sum and count | seconds | What the queue costs a caller. Mean over the interval, not since server start |
| `output_tokens_per_sec` | derived from `vllm:generation_tokens_total` | tokens/s | Delivered work. Cannot be faked by a resident kernel, which is why the ceiling ratio prefers it |
| `prompt_tokens_per_sec` | derived from `vllm:prompt_tokens_total` | tokens/s | Prefill throughput. Separate from decode because they load the GPU differently |
| `batch_size` | `vllm:num_requests_running` | count | A GPU at `gpu_util` 1.0 with `batch_size` 1 is the canonical wasted node. Instantaneous, not a windowed mean: see the note below |

### Rates and why the first scrape is empty

`vllm:generation_tokens_total` and `vllm:prompt_tokens_total` are counters. A
counter is a total, not a rate, so a rate requires two observations and the
interval between them. The collector therefore reports
`output_tokens_per_sec` and `prompt_tokens_per_sec` **absent on the first
scrape**, with that as the recorded reason.

This is not a limitation to work around. Emitting a rate from one scrape means
dividing a lifetime total by an arbitrary interval, which produces a number that
looks like throughput and is not.

Four cases return absent rather than a number:

| Case | Why not a number |
|---|---|
| First scrape | No previous observation, so no interval |
| Counter decreased | vLLM restarted inside the interval. The tokens served before the restart are gone, so the interval's true rate is unknowable. Not zero, and not the post-restart total over the whole interval, which understates it arbitrarily |
| Interval is zero or negative | Dividing produces `+Inf` or a negative rate from a positive delta. Both render as measurements |
| Metric missing from the exposition page | Some builds omit metrics. Absent with the metric named |

A counter that genuinely did not move is a **rate of zero**, which is a
measurement and is reported as one. Absence and zero are different answers and
the collector keeps them apart.

`queue_seconds` follows the same rule with one addition: it is
`delta(sum) / delta(count)` over the interval, and when `delta(count)` is zero no
request finished queueing, so there is no observation in the window. That is
absent, not zero. Zero would claim an instant queue, which is a far more
reassuring statement than "nothing to report".

### `batch_size` is instantaneous

vLLM exposes no mean-batch metric, and a mean cannot be derived from token
counters without knowing sequence lengths. `batch_size` is
`vllm:num_requests_running` at the instant of the scrape.

It is reported rather than withheld because the instantaneous value is what makes
the decode case legible: "utilization is 1.0 and the batch is 1" is the sentence
this project exists to be able to say. A windowed mean arrives with the ceiling
store, which is where the history to compute one will already live.

### Multiple models on one GPU

vLLM normally serves one model per process, but the metrics carry a `model_name`
label and a multi-model process exposes several series per name. The collector
aggregates:

- **counts and counters sum**, because two models on one GPU do contend for it
- **`kv_cache_utilization` takes the maximum**, because it is a fraction of one
  physical cache. Summing fractions would exceed 1.0, and a mean would hide a
  full cache behind an empty one

### What the collector does not do

- **No engine other than vLLM.** TGI and Triton are the same shape under
  different names, and one engine read properly is worth more than three read
  partially
- **No `max_batch_size`.** vLLM carries `max_num_seqs` in its engine
  configuration and does not put it on `/metrics`. Absent rather than inferred
  from the highest running count seen, which is a high-water mark presented as a
  limit
- **No timestamp of its own.** The instant is passed in by the caller so the GPU
  read and the serving read sit on one clock. Two pollers each calling
  `time.Now()` cannot promise that, and without it a throughput dip cannot be
  attributed to the GPU behaviour that caused it
- **`NaN` is not a value.** Prometheus uses it for "no observation yet", which is
  exactly the case this project refuses to render as a number. It parses as
  absent


## Derived

### `util_gap`

```
util_gap = gpu_util - sm_occupancy
```

Absent if either input is absent. **Never computed by treating an absent input
as zero**, which would manufacture a large gap on any node whose driver lacks
profiling and is exactly the failure this project is against.

Signed on purpose. A negative gap should not occur and is worth surfacing rather
than hiding behind an absolute value.

### `ceiling_ratio`

```
ceiling_ratio = delivered / measured_ceiling
```

Preferring `output_tokens_per_sec` as the basis when a serving ceiling exists,
because it is what a serving node is for and a resident kernel cannot inflate
it. Falling back to `sm_occupancy`.

**Absent unless a trustworthy ceiling exists.** A ceiling is refused as a
denominator when it was:

- declared by an operator rather than measured on this node
- measured while the GPU was throttled, which understates capability, inflates
  every ratio computed against it, and makes a starved node look healthy
- recorded without a timestamp, so it cannot be aged

Not established in v0.1, so the column reads `n/a`.

## Verdicts

One instant supports a shape, not a recommendation. Recommendations need history
and policy and arrive in v0.3.

| Verdict | Condition | Why it is separate |
|---|---|---|
| `throttled` | any throttle flag set | Short-circuits everything. Rule 2: no software conclusion from a sample where hardware was holding back |
| `blocked` | `kv_cache_utilization` >= 0.90 and `queued_requests` >= 1 | Not an underused GPU, a fenced-off one. Adding capacity may be the wrong fix |
| `starved` | `gpu_util` >= 0.80 and `sm_occupancy` < 0.30 | The case this exists for |
| `busy` | high util backed by occupancy | Genuine work |
| `idle` | `gpu_util` < 0.05 | Honest about doing nothing |
| `unknown` | a needed counter was absent | Reported, never assumed healthy |

Order matters and is tested. `throttled` before `blocked` before `starved`.

## Thresholds, and why you may want to change them

`UtilBusy` 0.80 and `OccupancyLow` 0.30 come from the documented shape of
autoregressive decode, where `gpu_util` pins high while occupancy sits near 15%.

**They will misfire on some real workloads.** A workload of large fused kernels
can legitimately run at low occupancy and high efficiency, and would be called
starved here when it is not. That is a threshold problem, not a measurement
problem: change the thresholds and keep the measurement.

## Support matrix

| GPU | `gpu_util` | Profiling fields | Note |
|---|---|---|---|
| H100, H200 | yes | yes | Driver 525+ |
| A100, A30 | yes | yes | Driver 470+ for most fields |
| L4, L40S | yes | driver dependent | 520+ for profiling |
| T4, V100 | yes | partial | Several profiling fields absent; reported absent |
| MIG instances | yes | partial | Per-instance profiling is limited. Treat MIG numbers with care |
| Consumer (RTX) | yes | mostly no | DCGM support is limited outside data-centre parts |

Where a field is unavailable, `truthscale` records why and reports `n/a`. The
`l4-no-profiling` trace exercises exactly this path so the behaviour is tested
rather than asserted.
