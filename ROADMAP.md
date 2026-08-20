# Roadmap

Honest about status: **v0.1 does no live collection.** It reads recorded traces,
computes the honest signal, and explains it. That is a deliberate first release,
not an unfinished one, and the reason is in [Sequencing](#why-this-order).

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
| **Serving-aware** | KV cache, queue depth, batch size, token throughput |
| Deterministic trace generator | `go run ./cmd/gen-traces`, verified in CI |

## Next (v0.2): live, and worth alerting on

| Item | Note |
| :--- | :--- |
| DCGM collector | Read real fields from `dcgm-exporter` or the DCGM socket. Detect and report unavailable fields rather than substituting |
| vLLM collector | Scrape `/metrics`, correlate against the GPU sample on one monotonic clock |
| `truthscale measure` | Establish this node's ceiling with a bounded, cancellable microbenchmark. Refuses to run when another process holds the GPU |
| Ceiling store and ageing | Re-measure on a schedule, on driver change, and on MIG profile change |
| `truthscale export` | Prometheus endpoint, with `ceiling_ratio` as the headline series |
| DaemonSet and Helm chart | Plus a ServiceMonitor and a Grafana dashboard as a real artifact |
| Alerting rules | With the rationale for each, because a rule nobody can justify gets muted |
| Slack and webhook sink | |

## Later (v0.3): the decision engine

The intellectual centre, and last on purpose.

| Item | Note |
| :--- | :--- |
| Decision function | Given ceiling ratio, queue depth, cache pressure, and throttle state: scale up, scale down, rightsize, or do nothing. Documented and testable, not a magic number |
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
