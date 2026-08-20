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

| Field | vLLM metric | Units | Why it matters |
|---|---|---|---|
| `kv_cache_utilization` | `vllm:gpu_cache_usage_perc` | fraction | Explains the paradox. A full cache blocks admission while a resident kernel pins `gpu_util` at 1.0 |
| `running_requests` | `vllm:num_requests_running` | count | The real concurrency, as against apparent busyness |
| `queued_requests` | `vllm:num_requests_waiting` | count | Work that exists and cannot start |
| `queue_seconds` | `vllm:request_queue_time_seconds` | seconds | What the queue costs a caller |
| `output_tokens_per_sec` | derived from `vllm:generation_tokens_total` | tokens/s | Delivered work. Cannot be faked by a resident kernel, which is why the ceiling ratio prefers it |
| `prompt_tokens_per_sec` | derived from `vllm:prompt_tokens_total` | tokens/s | Prefill throughput. Separate from decode because they load the GPU differently |
| `batch_size` | derived from `running_requests` | count | A GPU at `gpu_util` 1.0 with `batch_size` 1 is the canonical wasted node |

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
