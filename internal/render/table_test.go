package render

import (
	"bytes"
	"strings"
	"testing"

	"github.com/ehtishammubarik/truthscale/internal/collect"
	"github.com/ehtishammubarik/truthscale/internal/signal"
	"github.com/ehtishammubarik/truthscale/internal/synth"
)

func signalsFor(sc synth.Scenario, n int) []signal.Signal {
	var out []signal.Signal
	for _, s := range synth.Generate(sc, n, 11) {
		out = append(out, signal.Evaluate(s, nil, signal.DefaultThresholds()))
	}
	return out
}

// The column order is the argument: UTIL next to OCCUPANCY next to GAP, so the
// discrepancy is visible to a reader who did not know to look for it.
func TestTheLieAndTheTruthAreAdjacentColumns(t *testing.T) {
	var b bytes.Buffer
	Table(&b, signalsFor(synth.DecodeStarved, 1))
	header := strings.SplitN(b.String(), "\n", 2)[0]

	iUtil := strings.Index(header, "UTIL")
	iOcc := strings.Index(header, "OCCUPANCY")
	iGap := strings.Index(header, "GAP")
	if iUtil < 0 || iOcc < 0 || iGap < 0 {
		t.Fatalf("header must carry UTIL, OCCUPANCY and GAP: %q", header)
	}
	if !(iUtil < iOcc && iOcc < iGap) {
		t.Fatalf("UTIL, OCCUPANCY, GAP must appear in that order: %q", header)
	}
}

// Rule 1, at the last place it can be broken. Everything upstream can keep
// absence intact and a renderer printing 0%% still tells the lie.
func TestAnAbsentCounterRendersAsNotAvailableNeverZero(t *testing.T) {
	var b bytes.Buffer
	Table(&b, signalsFor(synth.NoProfiling, 1))
	out := b.String()

	if !strings.Contains(out, "n/a") {
		t.Fatalf("an absent counter must render n/a:\n%s", out)
	}
	// The row must not contain a 0% that came from an unmeasured field.
	row := strings.Split(strings.TrimSpace(out), "\n")[1]
	if strings.Contains(row, "0%") {
		t.Fatalf("absent metrics must not render as 0%%: %q", row)
	}
}

// A node with no inference server is normal, not broken, and must not render as
// though its serving metrics were measured and found to be zero.
func TestANodeWithNoServingStackRendersDashesNotZeroes(t *testing.T) {
	var b bytes.Buffer
	Table(&b, signalsFor(synth.ThermalThrottled, 1))
	row := strings.Split(strings.TrimSpace(b.String()), "\n")[1]
	if !strings.Contains(row, "-") {
		t.Fatalf("a node with no serving stack should show dashes: %q", row)
	}
}

// The number worth acting on is how many GPUs are lying, not any single row.
func TestSummaryCountsVerdictsAndPointsAtExplain(t *testing.T) {
	var b bytes.Buffer
	sigs := signalsFor(synth.MixedFleet, 4)
	Summary(&b, sigs)
	out := b.String()

	if !strings.Contains(out, "starved") {
		t.Errorf("summary must count starved GPUs:\n%s", out)
	}
	if !strings.Contains(out, "truthscale explain") {
		t.Errorf("summary must tell the reader what to run next:\n%s", out)
	}
}

// An unjudgeable GPU must be called out, because silence would read as health.
func TestSummarySaysUnknownGpusWereNotAssumedHealthy(t *testing.T) {
	var b bytes.Buffer
	Summary(&b, signalsFor(synth.NoProfiling, 2))
	out := b.String()
	if !strings.Contains(out, "rather than assumed healthy") {
		t.Errorf("unknown GPUs must be flagged as unjudged, not silently omitted:\n%s", out)
	}
}

// A negative gap should not happen. Hiding it behind an absolute value would
// turn a bug into a plausible-looking number.
func TestTheGapIsSignedSoAnImpossibleValueIsVisible(t *testing.T) {
	s := collect.Sample{
		Node: "gpu-01", GPUUtil: collect.Known(0.20), SMOccupancy: collect.Known(0.60),
	}
	var b bytes.Buffer
	Table(&b, []signal.Signal{signal.Evaluate(s, nil, signal.DefaultThresholds())})
	if !strings.Contains(b.String(), "-40") {
		t.Errorf("a negative gap must render with its sign:\n%s", b.String())
	}
}
