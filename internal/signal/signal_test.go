package signal

import (
	"strings"
	"testing"
	"time"

	"github.com/ehtishammubarik/truthscale/internal/ceiling"
	"github.com/ehtishammubarik/truthscale/internal/collect"
)

func sample(util, occ float64) collect.Sample {
	return collect.Sample{
		Taken: time.Now(), Node: "gpu-01", GPU: 0, Model: "H100-80GB",
		GPUUtil: collect.Known(util), SMOccupancy: collect.Known(occ),
	}
}

// The whole reason this project exists: a GPU reporting 100% utilization while
// almost nothing is happening must not be called busy.
func TestAResidentKernelIsNotCalledBusy(t *testing.T) {
	got := Evaluate(sample(1.0, 0.15), nil, DefaultThresholds())
	if got.Verdict != VerdictStarved {
		t.Fatalf("util 100%% with 15%% occupancy: want %q, got %q", VerdictStarved, got.Verdict)
	}
	gap, ok := got.UtilGap.Get()
	if !ok || gap < 0.84 || gap > 0.86 {
		t.Fatalf("util gap: want ~0.85, got %v (present=%v)", gap, ok)
	}
}

func TestRealWorkIsCalledBusy(t *testing.T) {
	if got := Evaluate(sample(0.95, 0.82), nil, DefaultThresholds()); got.Verdict != VerdictBusy {
		t.Fatalf("want %q, got %q", VerdictBusy, got.Verdict)
	}
}

func TestAnIdleGpuIsNotCalledStarved(t *testing.T) {
	if got := Evaluate(sample(0.01, 0.0), nil, DefaultThresholds()); got.Verdict != VerdictIdle {
		t.Fatalf("want %q, got %q", VerdictIdle, got.Verdict)
	}
}

// Rule 2. A throttled sample supports no conclusion about software, so it must
// short-circuit even when the numbers otherwise look like starvation.
func TestThrottlingPreemptsEveryOtherVerdict(t *testing.T) {
	s := sample(1.0, 0.10) // would be starved
	s.Throttles = []collect.ThrottleReason{collect.ThrottleThermal}
	got := Evaluate(s, nil, DefaultThresholds())
	if got.Verdict != VerdictThrottled {
		t.Fatalf("want %q, got %q", VerdictThrottled, got.Verdict)
	}
	if !strings.Contains(strings.Join(got.Reasons, " "), "hardware before it is software") {
		t.Fatalf("the throttle reason must say why no software conclusion follows: %v", got.Reasons)
	}
}

// A full KV cache with queued work is not an underused GPU. Calling it starved
// would send someone to add capacity when the fix may be configuration.
func TestAFullKvCacheWithQueuedWorkIsBlockedNotStarved(t *testing.T) {
	s := sample(1.0, 0.12)
	s.Serving = &collect.ServingState{
		Engine:             "vllm",
		KVCacheUtilization: collect.Known(0.97),
		QueuedRequests:     collect.Known(14),
	}
	got := Evaluate(s, nil, DefaultThresholds())
	if got.Verdict != VerdictBlocked {
		t.Fatalf("want %q, got %q", VerdictBlocked, got.Verdict)
	}
	if !strings.Contains(strings.Join(got.Reasons, " "), "cheaper") {
		t.Fatalf("blocked must offer the non-capacity fix: %v", got.Reasons)
	}
}

// Absent is not zero. An unavailable occupancy counter must produce "unknown",
// never "starved", because the latter is an accusation the data cannot support.
func TestMissingOccupancyIsUnknownNotStarved(t *testing.T) {
	s := sample(1.0, 0)
	s.SMOccupancy = collect.Absent()
	s.Unavailable = map[string]string{"sm_occupancy": "DCGM profiling unavailable on this driver"}
	got := Evaluate(s, nil, DefaultThresholds())
	if got.Verdict != VerdictUnknown {
		t.Fatalf("want %q, got %q", VerdictUnknown, got.Verdict)
	}
	if _, ok := got.UtilGap.Get(); ok {
		t.Fatal("util gap must be absent when occupancy is absent, not computed from zero")
	}
}

// A ceiling measured while throttled understates capability, which inflates
// every ratio computed against it and makes a starved node look healthy.
func TestAThrottledCeilingIsNotUsedAsADenominator(t *testing.T) {
	c := &ceiling.Ceiling{
		Source: ceiling.SourceMeasured, Taken: time.Now(),
		SMOccupancyCeiling: 0.30, ThrottledDuringMeasure: true,
	}
	got := Evaluate(sample(0.9, 0.25), c, DefaultThresholds())
	if _, ok := got.CeilingRatio.Get(); ok {
		t.Fatal("a ceiling measured under throttling must not produce a ratio")
	}
}

func TestCeilingRatioPrefersTokenThroughputOverOccupancy(t *testing.T) {
	c := &ceiling.Ceiling{
		Source: ceiling.SourceMeasured, Taken: time.Now(),
		SMOccupancyCeiling: 0.90, OutputTokensPerSec: 2000,
	}
	s := sample(0.95, 0.45)
	s.Serving = &collect.ServingState{Engine: "vllm", OutputTokensPerSec: collect.Known(500)}
	got := Evaluate(s, c, DefaultThresholds())
	if got.CeilingBasis != "output_tokens_per_sec" {
		t.Fatalf("want token basis, got %q", got.CeilingBasis)
	}
	r, _ := got.CeilingRatio.Get()
	if r < 0.24 || r > 0.26 {
		t.Fatalf("500/2000 should be ~0.25, got %v", r)
	}
}

func TestDeclaredCeilingsAreNotTrusted(t *testing.T) {
	c := &ceiling.Ceiling{Source: ceiling.SourceDeclared, Taken: time.Now(), SMOccupancyCeiling: 0.9}
	if _, ok := Evaluate(sample(0.9, 0.5), c, DefaultThresholds()).CeilingRatio.Get(); ok {
		t.Fatal("a declared ceiling must not be used as a measured denominator")
	}
}

// Explain is the command people screenshot. It must show the pairing, and it
// must say n/a rather than a number when a counter was unavailable.
func TestExplainShowsBothNumbersAndAdmitsWhatIsMissing(t *testing.T) {
	s := sample(1.0, 0.15)
	s.Serving = &collect.ServingState{Engine: "vllm", BatchSize: collect.Known(1), KVCacheUtilization: collect.Known(0.08)}
	out := Evaluate(s, nil, DefaultThresholds()).Explain()

	for _, want := range []string{"what the dashboard says", "what the SMs say", "the gap", "STARVED"} {
		if !strings.Contains(out, want) {
			t.Errorf("explain must contain %q:\n%s", want, out)
		}
	}
	if !strings.Contains(out, "n/a (no trustworthy ceiling") {
		t.Errorf("explain must admit a missing ceiling rather than printing a number:\n%s", out)
	}
	if !strings.Contains(out, "one small kernel per token") {
		t.Errorf("a batch of 1 should draw the decode explanation:\n%s", out)
	}
}

func TestExplainRendersAbsentMetricsAsNotAvailable(t *testing.T) {
	s := sample(0.9, 0)
	s.SMOccupancy = collect.Absent()
	s.Unavailable = map[string]string{"sm_occupancy": "driver too old for DCGM profiling"}
	out := Evaluate(s, nil, DefaultThresholds()).Explain()
	if !strings.Contains(out, "occupancy   n/a") {
		t.Errorf("absent occupancy must render n/a, not 0%%:\n%s", out)
	}
	if !strings.Contains(out, "driver too old") {
		t.Errorf("explain must surface why a counter was unavailable:\n%s", out)
	}
}
