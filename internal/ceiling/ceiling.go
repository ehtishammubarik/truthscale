// Package ceiling holds what a GPU was measured to be capable of.
//
// Not what the datasheet claims. A datasheet number is measured at a clock and
// a power limit and a thermal envelope you probably do not have, and comparing
// live throughput against it produces a ratio that is wrong in an unknown
// direction. A ceiling here is always something this GPU actually did, on this
// node, under this driver.
package ceiling

import (
	"fmt"
	"time"
)

// Source records how a ceiling was arrived at. Kept alongside the number
// because a measured ceiling and a declared one must never be compared as
// though they were the same kind of thing.
type Source string

const (
	// SourceMeasured came from `truthscale measure` on this node.
	SourceMeasured Source = "measured"
	// SourceObserved is the best the node has been seen to do while serving
	// real traffic. A lower bound on the true ceiling, and honest about it.
	SourceObserved Source = "observed"
	// SourceDeclared was supplied by an operator who knows something we do not.
	SourceDeclared Source = "declared"
)

// Ceiling is one GPU's measured capability.
type Ceiling struct {
	Node  string `json:"node"`
	GPU   int    `json:"gpu"`
	Model string `json:"model"`

	Source Source    `json:"source"`
	Taken  time.Time `json:"taken"`

	// How the measurement was made, so a reader can reproduce or distrust it.
	DriverVersion string  `json:"driver_version,omitempty"`
	Method        string  `json:"method,omitempty"`
	DurationSec   float64 `json:"duration_sec,omitempty"`

	// The ceilings themselves. Zero means not established for that dimension,
	// and callers check Has* rather than reading the field directly.
	SMOccupancyCeiling  float64 `json:"sm_occupancy_ceiling,omitempty"`
	TensorActiveCeiling float64 `json:"tensor_active_ceiling,omitempty"`
	DRAMActiveCeiling   float64 `json:"dram_active_ceiling,omitempty"`
	OutputTokensPerSec  float64 `json:"output_tokens_per_sec_ceiling,omitempty"`

	// Conditions during measurement. A ceiling measured while the GPU was
	// thermally throttled is not a ceiling, and this is how that gets caught.
	ThrottledDuringMeasure bool `json:"throttled_during_measure"`
}

// HasSMOccupancy reports whether an SM occupancy ceiling was established.
func (c Ceiling) HasSMOccupancy() bool { return c.SMOccupancyCeiling > 0 }

// HasTokenThroughput reports whether a serving throughput ceiling exists.
func (c Ceiling) HasTokenThroughput() bool { return c.OutputTokensPerSec > 0 }

// Trustworthy reports whether this ceiling should be used as a denominator, and
// why not when it should not.
//
// A ceiling taken while the GPU was throttled understates capability, which
// inflates every ceiling ratio computed against it and makes a starved node
// look healthy. That failure is silent, so it is checked here rather than left
// to the caller to remember.
func (c Ceiling) Trustworthy() (bool, string) {
	if c.Source == SourceDeclared {
		return false, "declared by an operator, not measured on this node"
	}
	if c.ThrottledDuringMeasure {
		return false, "the GPU was throttled while the ceiling was measured, so the ceiling is too low"
	}
	if c.Taken.IsZero() {
		return false, "no timestamp, so the ceiling cannot be aged"
	}
	return true, ""
}

// Stale reports whether the ceiling is older than the given age. A driver
// upgrade, a power-limit change, or a different MIG profile all invalidate it,
// and none of them announce themselves.
func (c Ceiling) Stale(maxAge time.Duration, now time.Time) bool {
	return now.Sub(c.Taken) > maxAge
}

func (c Ceiling) String() string {
	return fmt.Sprintf("%s gpu%d (%s) ceiling from %s at %s",
		c.Node, c.GPU, c.Model, c.Source, c.Taken.Format(time.RFC3339))
}
