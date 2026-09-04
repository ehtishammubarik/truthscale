# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0: the CLI surface and the JSON trace schema may change
between minor versions, and any change to either will be called out here.

## [Unreleased]

## [0.1.0] - 2026-09-01

First tagged release. The code and tests existed before this tag; what was
missing was a version anyone could pin, which is why `go install` did not work
and the ROADMAP's claim that v0.1 was "shipped" was not true. It is now.

### The argument

`nvidia-smi` reports "utilization" and means "a kernel was resident". One kernel
occupying 1 of an H100's 132 SMs for a whole sample window reads 100%. Fleets are
sized on that number. `truthscale` reports it beside `sm_occupancy`, which is the
fraction of resident warp slots actually occupied, and the gap between the two is
the finding.

### Added

- **The honest signal.** `gpu_util` beside `sm_occupancy`, with `util_gap` as the
  signed difference. Absent if either input is absent, never computed by treating
  a missing input as zero, which would manufacture a large gap on any node whose
  driver lacks profiling.
- **Six verdicts** in a tested precedence: `throttled`, `blocked`, `starved`,
  `busy`, `idle`, `unknown`. `throttled` short-circuits the rest, because a
  conclusion about software drawn from a thermally limited sample is worse than
  no conclusion.
- **`truthscale top`**, one row per GPU, the misleading number beside the honest
  one.
- **`truthscale explain`**, why the two numbers differ, in prose, per GPU.
- **`--replay` on every command**, reading committed JSON Lines traces, so the
  tool runs on a laptop with no GPU. Six traces ship in `traces/`, including
  `l4-no-profiling` which exercises the absent-counter path.
- **Absent is never zero.** Every optional field is a `collect.Metric` that knows
  whether it was measured. An unavailable counter renders `n/a` with a reason
  attached, and absence survives a JSON round trip, which is asserted by a test.
  An idle GPU and an unsupported driver must not look identical, and on most
  dashboards they do.
- **vLLM collector** reading a live `/metrics` endpoint: KV cache utilization,
  queue depth, running requests, batch size, and token throughput
  ([#2](https://github.com/ehtishammubarik/truthscale/issues/2)). Token rates are
  derived from counters, so they are **absent on a first scrape and absent across
  a server restart**, never zero and never negative. A counter that genuinely did
  not move is a rate of zero, which is a measurement, and the two are kept apart.
  The scrape instant is supplied by the caller so the serving read and the GPU
  read sit on one clock.
- **Deterministic trace generator**, `go run ./cmd/gen-traces`, with CI asserting
  the committed traces are reproducible from the committed generator. Traces of
  unknown provenance are the thing this project is against.

### Known limitations

Stated here rather than discovered later.

- **No live GPU collection.** The DCGM collector is
  [#1](https://github.com/ehtishammubarik/truthscale/issues/1). Until it lands the
  GPU side is trace-only, so a fully live sample is not yet possible.
- **`ceiling_ratio` always reads `n/a`.** Establishing a measured ceiling is
  [#3](https://github.com/ehtishammubarik/truthscale/issues/3) and storing one is
  [#8](https://github.com/ehtishammubarik/truthscale/issues/8). A declared ceiling
  is refused as a denominator on purpose.
- **No Helm chart or exporter yet**
  ([#4](https://github.com/ehtishammubarik/truthscale/issues/4)).
- **No recommendations.** A verdict describes one instant; a recommendation needs
  history and a policy someone can argue with, and it is
  [#5](https://github.com/ehtishammubarik/truthscale/issues/5), deliberately last.
  Nobody should act on a signal they have not learned to trust.
- **The thresholds will misfire on some real workloads.** `UtilBusy` 0.80 and
  `OccupancyLow` 0.30 come from the documented shape of autoregressive decode. A
  workload of large fused kernels can legitimately run at low occupancy and high
  efficiency, and would be called `starved` here when it is not. That is a
  threshold problem, not a measurement problem: change the thresholds and keep the
  measurement. Every default and the workload where it is wrong is in
  [docs/metrics.md](docs/metrics.md).
- **vLLM only.** TGI is
  [#10](https://github.com/ehtishammubarik/truthscale/issues/10); Triton is not
  scheduled.

### Not planned

Actuation without explicit opt-in. A tool that resizes a GPU fleet on the day it
is installed is a tool nobody installs. It is also not a scheduler, not a hosted
service, and not a replacement for DCGM.

[Unreleased]: https://github.com/ehtishammubarik/truthscale/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ehtishammubarik/truthscale/releases/tag/v0.1.0
