// Package synth generates the traces that ship in traces/.
//
// These are synthetic. They are shaped from the published behaviour of
// autoregressive decode and from the counter semantics documented in
// docs/metrics.md, and they contain no data from any real deployment. That
// boundary is deliberate: the measurement research this project draws on was
// done under a client pilot, and methodology is transferable where data is not.
//
// Synthetic traces are honest as long as the README says they are synthetic and
// the generator is committed so anyone can see how the shape was chosen. Both
// are true.
package synth

import (
	"math"
	"math/rand"
	"time"

	"github.com/ehtishammubarik/truthscale/internal/collect"
)

// Scenario is one recognisable GPU situation worth having a trace of.
type Scenario string

const (
	// DecodeStarved is the case this project exists for: vLLM serving a small
	// batch, utilization pinned high, SMs mostly idle, cache barely used.
	DecodeStarved Scenario = "h100-vllm-decode"
	// PrefillBusy is genuine work: large fused kernels, high occupancy.
	PrefillBusy Scenario = "h100-vllm-prefill"
	// CacheBlocked has queued requests that cannot be admitted.
	CacheBlocked Scenario = "h100-vllm-kv-blocked"
	// ThermalThrottled is being held back by hardware, so no software
	// conclusion may be drawn.
	ThermalThrottled Scenario = "a100-thermal-throttle"
	// NoProfiling is a driver without DCGM profiling fields, which must report
	// unknown rather than a fabricated zero.
	NoProfiling Scenario = "l4-no-profiling"
	// MixedFleet is several nodes at once, for the `top` demo.
	MixedFleet Scenario = "mixed-fleet"
)

// Generate builds a deterministic trace for a scenario. The seed is fixed by
// the caller so a committed trace is byte-reproducible and a diff to it is
// meaningful rather than noise.
func Generate(sc Scenario, samples int, seed int64) []collect.Sample {
	rng := rand.New(rand.NewSource(seed))
	start := time.Date(2026, 8, 20, 9, 0, 0, 0, time.UTC)

	switch sc {
	case MixedFleet:
		var out []collect.Sample
		for i, part := range []Scenario{DecodeStarved, PrefillBusy, CacheBlocked, ThermalThrottled, NoProfiling} {
			for _, s := range Generate(part, samples, seed+int64(i)) {
				s.Node = nodeFor(part)
				out = append(out, s)
			}
		}
		return out
	}

	out := make([]collect.Sample, 0, samples)
	for i := 0; i < samples; i++ {
		at := start.Add(time.Duration(i) * time.Second)
		out = append(out, one(sc, at, i, rng))
	}
	return out
}

func nodeFor(sc Scenario) string {
	switch sc {
	case DecodeStarved:
		return "gpu-node-01"
	case PrefillBusy:
		return "gpu-node-02"
	case CacheBlocked:
		return "gpu-node-03"
	case ThermalThrottled:
		return "gpu-node-04"
	case NoProfiling:
		return "gpu-node-05"
	}
	return "gpu-node-00"
}

func one(sc Scenario, at time.Time, i int, rng *rand.Rand) collect.Sample {
	// A little jitter so a trace looks like a measurement rather than a
	// constant, and so percentile code has something to chew on.
	j := func(scale float64) float64 { return (rng.Float64() - 0.5) * scale }
	clamp := func(v float64) float64 { return math.Max(0, math.Min(1, v)) }

	s := collect.Sample{
		Taken: at, Node: nodeFor(sc), GPU: 0,
		MemoryTotalBytes: collect.Known(80 * 1024 * 1024 * 1024),
		PowerLimitWatts:  collect.Known(700),
	}

	switch sc {
	case DecodeStarved:
		// The signature: one small kernel per token keeps a kernel resident, so
		// the utilization flag saturates while the SMs and tensor cores idle.
		s.Model = "NVIDIA H100 80GB HBM3"
		s.GPUUtil = collect.Known(clamp(0.99 + j(0.02)))
		s.SMOccupancy = collect.Known(clamp(0.14 + j(0.04)))
		s.SMActive = collect.Known(clamp(0.21 + j(0.05)))
		s.TensorActive = collect.Known(clamp(0.04 + j(0.02)))
		s.DRAMActive = collect.Known(clamp(0.38 + j(0.06)))
		s.MemoryUsedBytes = collect.Known(41 * 1024 * 1024 * 1024)
		s.PowerWatts = collect.Known(228 + j(24))
		s.TempCelsius = collect.Known(58 + j(4))
		s.SMClockMHz = collect.Known(1755)
		s.Serving = &collect.ServingState{
			Engine:             "vllm",
			KVCacheUtilization: collect.Known(clamp(0.07 + j(0.03))),
			RunningRequests:    collect.Known(1),
			QueuedRequests:     collect.Known(0),
			QueueSeconds:       collect.Known(0.002 + j(0.001)),
			PromptTokensPerSec: collect.Known(0),
			OutputTokensPerSec: collect.Known(47 + j(8)),
			BatchSize:          collect.Known(1),
			MaxBatchSize:       collect.Known(256),
		}

	case PrefillBusy:
		s.Model = "NVIDIA H100 80GB HBM3"
		s.GPUUtil = collect.Known(clamp(0.97 + j(0.03)))
		s.SMOccupancy = collect.Known(clamp(0.83 + j(0.06)))
		s.SMActive = collect.Known(clamp(0.91 + j(0.04)))
		s.TensorActive = collect.Known(clamp(0.62 + j(0.08)))
		s.DRAMActive = collect.Known(clamp(0.71 + j(0.07)))
		s.MemoryUsedBytes = collect.Known(68 * 1024 * 1024 * 1024)
		s.PowerWatts = collect.Known(648 + j(30))
		s.TempCelsius = collect.Known(71 + j(3))
		s.SMClockMHz = collect.Known(1755)
		s.Serving = &collect.ServingState{
			Engine:             "vllm",
			KVCacheUtilization: collect.Known(clamp(0.62 + j(0.08))),
			RunningRequests:    collect.Known(48 + math.Trunc(j(6))),
			QueuedRequests:     collect.Known(2),
			QueueSeconds:       collect.Known(0.09 + j(0.03)),
			PromptTokensPerSec: collect.Known(18400 + j(2200)),
			OutputTokensPerSec: collect.Known(1920 + j(180)),
			BatchSize:          collect.Known(48),
			MaxBatchSize:       collect.Known(256),
		}

	case CacheBlocked:
		s.Model = "NVIDIA H100 80GB HBM3"
		s.GPUUtil = collect.Known(clamp(0.99 + j(0.01)))
		s.SMOccupancy = collect.Known(clamp(0.19 + j(0.04)))
		s.SMActive = collect.Known(clamp(0.26 + j(0.05)))
		s.TensorActive = collect.Known(clamp(0.06 + j(0.02)))
		s.DRAMActive = collect.Known(clamp(0.44 + j(0.05)))
		s.MemoryUsedBytes = collect.Known(79 * 1024 * 1024 * 1024)
		s.PowerWatts = collect.Known(276 + j(20))
		s.TempCelsius = collect.Known(64 + j(3))
		s.SMClockMHz = collect.Known(1755)
		s.Serving = &collect.ServingState{
			Engine:             "vllm",
			KVCacheUtilization: collect.Known(clamp(0.97 + j(0.02))),
			RunningRequests:    collect.Known(12),
			QueuedRequests:     collect.Known(31 + math.Trunc(j(8))),
			QueueSeconds:       collect.Known(4.2 + j(1.1)),
			PromptTokensPerSec: collect.Known(0),
			OutputTokensPerSec: collect.Known(310 + j(40)),
			BatchSize:          collect.Known(12),
			MaxBatchSize:       collect.Known(256),
		}

	case ThermalThrottled:
		s.Model = "NVIDIA A100-SXM4-40GB"
		s.GPUUtil = collect.Known(clamp(0.96 + j(0.03)))
		s.SMOccupancy = collect.Known(clamp(0.44 + j(0.06)))
		s.SMActive = collect.Known(clamp(0.51 + j(0.05)))
		s.TensorActive = collect.Known(clamp(0.29 + j(0.05)))
		s.DRAMActive = collect.Known(clamp(0.55 + j(0.06)))
		s.MemoryUsedBytes = collect.Known(33 * 1024 * 1024 * 1024)
		s.MemoryTotalBytes = collect.Known(40 * 1024 * 1024 * 1024)
		s.PowerWatts = collect.Known(398 + j(8))
		s.PowerLimitWatts = collect.Known(400)
		s.TempCelsius = collect.Known(87 + j(2))
		s.SMClockMHz = collect.Known(1005) // well under the 1410 boost
		s.Throttles = []collect.ThrottleReason{collect.ThrottleThermal, collect.ThrottleSWLimit}

	case NoProfiling:
		// An older driver that exposes the utilization flag and none of the
		// profiling fields. The tool must say unknown, not zero.
		s.Model = "NVIDIA L4"
		s.GPUUtil = collect.Known(clamp(0.88 + j(0.06)))
		s.SMOccupancy = collect.Absent()
		s.SMActive = collect.Absent()
		s.TensorActive = collect.Absent()
		s.DRAMActive = collect.Absent()
		s.MemoryUsedBytes = collect.Known(14 * 1024 * 1024 * 1024)
		s.MemoryTotalBytes = collect.Known(24 * 1024 * 1024 * 1024)
		s.PowerWatts = collect.Known(58 + j(6))
		s.PowerLimitWatts = collect.Known(72)
		s.TempCelsius = collect.Known(63 + j(3))
		s.SMClockMHz = collect.Known(2040)
		s.Unavailable = map[string]string{
			"sm_occupancy":  "DCGM profiling fields require driver >= 520 and are unavailable here",
			"sm_active":     "DCGM profiling fields require driver >= 520 and are unavailable here",
			"tensor_active": "DCGM profiling fields require driver >= 520 and are unavailable here",
			"dram_active":   "DCGM profiling fields require driver >= 520 and are unavailable here",
		}
	}
	return s
}
