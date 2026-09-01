# Roadmap

Honest about status: **v0.1 collects serving state but not GPU state.** The vLLM
collector reads a live endpoint; the GPU side is still trace-only until the DCGM
collector lands ([#1](https://github.com/ehtishammubarik/truthscale/issues/1)),
so a full live sample is not yet possible. Recorded traces remain how every
command is demonstrated and tested. The ordering is deliberate and the reason is
in [Sequencing](#why-this-order).

## Vision

**Make the number people scale GPUs on mean something.**

Four commitments follow, and the first is why the rest are possible.

**A measurement that was not taken is never rendered as zero.** An unavailable
counter reports `n/a` with a reason attached. An idle GPU and an unsupported
driver must not look identical, and on most dashboards they do. Every optional
field is a type that knows whether it was set, and absence survives a JSON round
trip, which is asserted by a test.

**No throughput decline is attributed to software until hardware is ruled out**
from the sample taken at that same instant. A throttled GPU short-circuits every
other verdict, because a conclusion about software drawn from a thermally
limited sample is worse than no conclusion.

**Measured, never declared.** A ceiling is what this GPU did on this node under
this driver. Datasheet figures and measured figures never share a column, and a
ceiling captured while throttled is refused as a denominator.

**It reads DCGM; it does not reimplement it.** NVIDIA's exporter is the right
foundation and this is a layer on top, not a competitor to it.

**Where it stops.** Not a scheduler: Kueue and Volcano exist and are good. Not a
hosted service. Not a replacement for DCGM. And it will not actuate anything
without explicit opt-in, because a tool that resizes a GPU fleet on the day it
is installed is a tool nobody installs.

## Now (v0.1)

Shipped and tested.

| Capability | |
| :--- | :--- |
| **The honest signal** | `sm_occupancy` beside `gpu_util`, and the gap between them |
| **Six verdicts** | throttled, blocked, starved, busy, idle, unknown, in that precedence |
| `truthscale top` | one row per GPU, the lie beside the truth |
| `truthscale explain` | why the two numbers differ, in prose, per GPU |
| **`--replay` on every command** | committed traces, so it runs on a laptop with no GPU |
| **Absent is not zero** | typed optional metrics, absence survives JSON, tested |
| **Serving-aware** | KV cache, queue depth, batch size, token throughput, read from vLLM |
| **vLLM collector** | reads a real vLLM `/metrics` endpoint. Rates absent on a first scrape and across a restart, never zero ([#2](https://github.com/ehtishammubarik/truthscale/issues/2)) |
| Deterministic trace generator | `go run ./cmd/gen-traces`, verified in CI |

## Next (v0.2): live, and worth alerting on

[Milestone](https://github.com/ehtishammubarik/truthscale/milestone/1).

| Item | Note |
| :--- | :--- |
| [DCGM collector](https://github.com/ehtishammubarik/truthscale/issues/1) | Read real fields. Detect and report unavailable ones rather than substituting a zero |
| [`truthscale measure`](https://github.com/ehtishammubarik/truthscale/issues/3) | Establish the ceiling without disrupting what is running. Refuses on a busy GPU |
| [Helm chart and exporter](https://github.com/ehtishammubarik/truthscale/issues/4) | DaemonSet, ServiceMonitor, Grafana dashboard. An unavailable field is omitted, never exported as 0 |
| [Worked example](https://github.com/ehtishammubarik/truthscale/issues/6) | Real cluster, real numbers, including a false positive |
| Ceiling store and ageing | Re-measure on a schedule, on driver change, and on MIG profile change |
| Alerting rules and a Slack sink | With the rationale for each, because a rule nobody can justify gets muted |

## Later (v0.3): the decision engine

The intellectual centre, and last on purpose.

[Milestone](https://github.com/ehtishammubarik/truthscale/milestone/2).

| Item | Note |
| :--- | :--- |
| [The decision function](https://github.com/ehtishammubarik/truthscale/issues/5) | Three of the five common cases make the naive action wrong. That table is the product |
| `truthscale recommend` | Dry run, explaining every recommendation |
| CRD and controller | So a recommendation can become an action, behind opt-in |
| RKE2 and Cluster Autoscaler integration | |
| MIG-aware rightsizing | A fragmented A100 is a different problem from an idle one |

## Why this order

The signal has to be trustworthy before anything acts on it. A tool that
autoscaled a GPU fleet on day one, using a ceiling it had not measured and
counters it could not verify, would be a more confident version of the problem
it set out to solve.

So v0.1 ships no operator at all, and `truthscale top` without `--replay` prints
why rather than returning an empty table. An empty table would let someone
conclude their fleet was fine.

## Not planned

- **A scheduler.** Kueue and Volcano exist and are good
- **Reimplementing DCGM.** This reads it
- **Actuation without opt-in.** Permanent
- **A datasheet-relative ratio.** A denominator nobody measured produces a
  number that is wrong in an unknown direction
- **Rendering an unavailable counter as a number.** Permanent, and the reason
  this project exists

## How this file stays true

A PR that closes a roadmap item moves its row into **Now** in the same PR,
naming the PR that delivered it. Reviewers may block on it.

The claim most likely to rot is "v0.1 does no live collection". When that stops
being true, this file is wrong in the most misleading possible direction.

## Influencing this list

The order is a guess and a real fleet beats a guess. Open an issue describing
what you run, which counters your driver exposes, and what decision you are
trying to make. Concrete beats abstract.
