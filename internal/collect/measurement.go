// Package collect defines what a single correlated observation of a GPU looks
// like, and the types that keep "we did not measure this" distinguishable from
// "we measured this and it was zero".
//
// That distinction is the whole reason this package exists. A dashboard that
// renders an unavailable counter as 0.0 is not missing a feature, it is lying,
// and the lie is invisible: an idle GPU and an unsupported driver look
// identical. Every optional field here is a Metric, and a Metric knows whether
// it was ever set.
package collect

import (
	"encoding/json"
	"fmt"
	"time"
)

// Metric is a float that may be absent.
//
// Deliberately not a *float64. A pointer is easy to dereference by accident and
// the zero value of the pointer is nil, which panics rather than reporting.
// This type makes the caller ask.
type Metric struct {
	value   float64
	present bool
}

// Known returns a Metric that was measured.
func Known(v float64) Metric { return Metric{value: v, present: true} }

// Absent returns a Metric that was not measured. The reason belongs in
// Sample.Unavailable, so a reader can find out why rather than guessing.
func Absent() Metric { return Metric{} }

// Get returns the value and whether it was measured. There is no Value()
// accessor on purpose: every read has to acknowledge the second return.
func (m Metric) Get() (float64, bool) { return m.value, m.present }

// Or returns the measured value, or the fallback if absent. Use it for
// rendering, never for arithmetic that feeds a decision.
func (m Metric) Or(fallback float64) float64 {
	if m.present {
		return m.value
	}
	return fallback
}

// IsPresent reports whether this metric was measured.
func (m Metric) IsPresent() bool { return m.present }

// String renders an absent metric as "n/a" rather than as a number, because
// every path that shows a number to a human goes through here eventually.
func (m Metric) String() string {
	if !m.present {
		return "n/a"
	}
	return fmt.Sprintf("%.4g", m.value)
}

// MarshalJSON writes null for an absent metric. A trace round-trips through
// JSON, so absence has to survive the trip: writing 0 here would launder a
// missing measurement into a real one on the next read.
func (m Metric) MarshalJSON() ([]byte, error) {
	if !m.present {
		return []byte("null"), nil
	}
	return json.Marshal(m.value)
}

// UnmarshalJSON reads null and a missing key as absent.
func (m *Metric) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*m = Absent()
		return nil
	}
	var v float64
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*m = Known(v)
	return nil
}

// ThrottleReason is why the GPU is running below its clocks. Rule 2 of this
// project: a throughput decline is not attributed to software until these have
// been ruled out from the sample taken at the same instant.
type ThrottleReason string

const (
	ThrottlePower      ThrottleReason = "power"
	ThrottleThermal    ThrottleReason = "thermal"
	ThrottleSyncBoost  ThrottleReason = "sync_boost"
	ThrottleHWSlowdown ThrottleReason = "hw_slowdown"
	ThrottleSWLimit    ThrottleReason = "sw_power_cap"
)

// Sample is one instant, on one GPU, with the serving stack's view of the same
// instant attached.
//
// The single Taken timestamp is the point. GPU state and serving state are read
// in one pass against one monotonic clock, so a throughput dip can be matched
// to the GPU behaviour that caused it. Two independently-polled sources cannot
// answer that question, however good each one is.
type Sample struct {
	Taken time.Time `json:"taken"`
	Node  string    `json:"node"`
	GPU   int       `json:"gpu"`
	Model string    `json:"model,omitempty"`

	// The number everyone watches. NVIDIA defines it as the fraction of the
	// sample window in which at least one kernel was resident. It says nothing
	// about how much of the GPU that kernel used.
	GPUUtil Metric `json:"gpu_util"`

	// The numbers that mean something. All four are DCGM profiling fields and
	// are not available on every GPU or driver, which is why they are Metrics.
	SMOccupancy  Metric `json:"sm_occupancy"`
	SMActive     Metric `json:"sm_active"`
	TensorActive Metric `json:"tensor_active"`
	DRAMActive   Metric `json:"dram_active"`

	MemoryUsedBytes  Metric `json:"memory_used_bytes"`
	MemoryTotalBytes Metric `json:"memory_total_bytes"`
	PowerWatts       Metric `json:"power_watts"`
	PowerLimitWatts  Metric `json:"power_limit_watts"`
	TempCelsius      Metric `json:"temp_celsius"`
	SMClockMHz       Metric `json:"sm_clock_mhz"`

	Throttles []ThrottleReason `json:"throttles,omitempty"`

	// Serving state, when an inference server is present on this GPU. Absent is
	// the normal case for training or for a bare node.
	Serving *ServingState `json:"serving,omitempty"`

	// Why a field above is absent, keyed by JSON field name. A reader who finds
	// n/a can find out whether the driver is too old, the GPU lacks the
	// counter, or the scrape failed.
	Unavailable map[string]string `json:"unavailable,omitempty"`
}

// ServingState is what the inference server reports about the same instant.
// These are the metrics that make an LLM node's real behaviour legible, and
// none of them are visible in GPU counters alone.
type ServingState struct {
	Engine string `json:"engine"` // vllm, tgi, triton

	// The one that explains the paradox. Decode keeps a kernel resident, so
	// GPUUtil pins at 100%, while the batch is tiny because the KV cache is
	// full and no new sequence can be admitted.
	KVCacheUtilization Metric `json:"kv_cache_utilization"`

	RunningRequests Metric `json:"running_requests"`
	QueuedRequests  Metric `json:"queued_requests"`
	QueueSeconds    Metric `json:"queue_seconds"`

	PromptTokensPerSec Metric `json:"prompt_tokens_per_sec"`
	OutputTokensPerSec Metric `json:"output_tokens_per_sec"`

	// Mean batch size over the window. A GPU at 100% GPUUtil serving a batch of
	// 1 is the canonical wasted node.
	BatchSize Metric `json:"batch_size"`

	MaxBatchSize Metric `json:"max_batch_size"`
}

// Throttled reports whether anything is holding this GPU below its clocks, and
// what. Callers use it before attributing a slowdown to software.
func (s Sample) Throttled() (bool, []ThrottleReason) {
	return len(s.Throttles) > 0, s.Throttles
}
