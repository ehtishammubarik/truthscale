package ceiling

import (
	"testing"
	"time"
)

// The failure this guards is silent and expensive: a ceiling measured while the
// GPU was throttled is too low, so every ratio against it is inflated and a
// starved node reads as healthy. Exactly backwards.
func TestAThrottledCeilingIsNotTrustworthy(t *testing.T) {
	c := Ceiling{Source: SourceMeasured, Taken: time.Now(), ThrottledDuringMeasure: true}
	ok, why := c.Trustworthy()
	if ok {
		t.Fatal("a ceiling measured under throttling must not be trustworthy")
	}
	if why == "" {
		t.Fatal("an untrustworthy ceiling must say why")
	}
}

func TestADeclaredCeilingIsNotTrustworthy(t *testing.T) {
	c := Ceiling{Source: SourceDeclared, Taken: time.Now(), SMOccupancyCeiling: 0.9}
	if ok, _ := c.Trustworthy(); ok {
		t.Fatal("a declared ceiling is not a measured one")
	}
}

func TestACeilingWithNoTimestampCannotBeAged(t *testing.T) {
	if ok, why := (Ceiling{Source: SourceMeasured}).Trustworthy(); ok {
		t.Fatalf("want untrustworthy, got trustworthy (%q)", why)
	}
}

func TestAMeasuredCeilingTakenCleanIsTrustworthy(t *testing.T) {
	c := Ceiling{Source: SourceMeasured, Taken: time.Now(), SMOccupancyCeiling: 0.88}
	if ok, why := c.Trustworthy(); !ok {
		t.Fatalf("want trustworthy, got %q", why)
	}
}

// An observed ceiling is a lower bound on the truth rather than the truth, but
// it is still measured on this node and is usable.
func TestAnObservedCeilingIsUsable(t *testing.T) {
	c := Ceiling{Source: SourceObserved, Taken: time.Now(), OutputTokensPerSec: 1800}
	if ok, why := c.Trustworthy(); !ok {
		t.Fatalf("observed ceilings should be usable: %q", why)
	}
	if !c.HasTokenThroughput() {
		t.Fatal("HasTokenThroughput should be true when a token ceiling is set")
	}
}

// A driver upgrade, a power-limit change, or a different MIG profile all
// invalidate a ceiling and none of them announce themselves, so age matters.
func TestStalenessIsMeasuredAgainstAClock(t *testing.T) {
	now := time.Now()
	fresh := Ceiling{Taken: now.Add(-1 * time.Hour)}
	old := Ceiling{Taken: now.Add(-30 * 24 * time.Hour)}

	if fresh.Stale(7*24*time.Hour, now) {
		t.Error("a one-hour-old ceiling is not stale against a seven-day limit")
	}
	if !old.Stale(7*24*time.Hour, now) {
		t.Error("a thirty-day-old ceiling is stale against a seven-day limit")
	}
}

func TestUnsetDimensionsReportAsAbsentRatherThanZero(t *testing.T) {
	c := Ceiling{Source: SourceMeasured, Taken: time.Now()}
	if c.HasSMOccupancy() {
		t.Error("an unset SM occupancy ceiling must not read as established")
	}
	if c.HasTokenThroughput() {
		t.Error("an unset token ceiling must not read as established")
	}
}

func TestStringNamesTheNodeAndTheSource(t *testing.T) {
	c := Ceiling{Node: "gpu-01", GPU: 3, Model: "H100", Source: SourceMeasured, Taken: time.Now()}
	got := c.String()
	for _, want := range []string{"gpu-01", "gpu3", "H100", "measured"} {
		if !contains(got, want) {
			t.Errorf("String() missing %q: %s", want, got)
		}
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (len(sub) == 0 || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
