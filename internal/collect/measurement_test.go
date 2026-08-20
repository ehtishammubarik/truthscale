package collect

import (
	"encoding/json"
	"strings"
	"testing"
)

// The invariant the whole package exists to protect: a measurement that was
// never taken must not become a zero on the way through JSON. A trace is
// written and read back, so if absence does not survive the round trip, every
// downstream consumer silently gains a fabricated data point.
func TestAbsenceSurvivesAJsonRoundTrip(t *testing.T) {
	in := Sample{Node: "gpu-01", GPUUtil: Known(0.97), SMOccupancy: Absent()}

	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"sm_occupancy":null`) {
		t.Fatalf("absent metric must marshal to null, got %s", b)
	}

	var out Sample
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.SMOccupancy.IsPresent() {
		t.Fatal("absent metric came back present after a round trip")
	}
	if v, ok := out.GPUUtil.Get(); !ok || v != 0.97 {
		t.Fatalf("present metric corrupted: %v %v", v, ok)
	}
}

// A measured zero is real data and must be distinguishable from a missing one.
// An idle GPU and an unsupported driver look identical if this breaks.
func TestAMeasuredZeroIsNotTheSameAsAbsent(t *testing.T) {
	zero, absent := Known(0), Absent()

	if !zero.IsPresent() {
		t.Fatal("a measured zero must be present")
	}
	if absent.IsPresent() {
		t.Fatal("an absent metric must not be present")
	}
	if zero.String() == absent.String() {
		t.Fatalf("zero and absent must not render alike: both %q", zero.String())
	}
	if absent.String() != "n/a" {
		t.Fatalf("absent must render as n/a, got %q", absent.String())
	}
}

// A key missing from the JSON entirely is the same as null: also absent.
func TestAMissingKeyIsAbsentNotZero(t *testing.T) {
	var s Sample
	if err := json.Unmarshal([]byte(`{"node":"gpu-01","gpu_util":0.5}`), &s); err != nil {
		t.Fatal(err)
	}
	if s.SMOccupancy.IsPresent() {
		t.Fatal("a key absent from the document must not arrive as a measured zero")
	}
}

func TestOrIsForRenderingAndDoesNotFabricatePresence(t *testing.T) {
	a := Absent()
	if got := a.Or(-1); got != -1 {
		t.Fatalf("Or should return the fallback, got %v", got)
	}
	if a.IsPresent() {
		t.Fatal("Or must not mutate presence")
	}
}

func TestThrottledReportsTheReasons(t *testing.T) {
	s := Sample{Throttles: []ThrottleReason{ThrottlePower, ThrottleThermal}}
	ok, reasons := s.Throttled()
	if !ok || len(reasons) != 2 {
		t.Fatalf("want 2 reasons, got %v %v", ok, reasons)
	}
	if none, _ := (Sample{}).Throttled(); none {
		t.Fatal("a sample with no throttles must not report throttled")
	}
}
