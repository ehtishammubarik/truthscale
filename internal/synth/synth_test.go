package synth

import (
	"testing"

	"github.com/ehtishammubarik/truthscale/internal/collect"
	"github.com/ehtishammubarik/truthscale/internal/signal"
)

// A committed trace has to be byte-reproducible, or `git diff traces/` is noise
// and CI cannot check that a trace still matches its generator.
func TestGenerationIsDeterministicForAGivenSeed(t *testing.T) {
	a := Generate(DecodeStarved, 20, 42)
	b := Generate(DecodeStarved, 20, 42)
	if len(a) != len(b) {
		t.Fatalf("length differs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		av, _ := a[i].GPUUtil.Get()
		bv, _ := b[i].GPUUtil.Get()
		if av != bv || !a[i].Taken.Equal(b[i].Taken) {
			t.Fatalf("sample %d differs between runs with the same seed", i)
		}
	}
}

// Each scenario exists to exercise one verdict. If a scenario stops producing
// the verdict it was built for, either the generator drifted or the thresholds
// changed, and both are things a reviewer needs told about.
func TestEachScenarioProducesTheVerdictItWasBuiltFor(t *testing.T) {
	cases := []struct {
		sc   Scenario
		want signal.Verdict
	}{
		{DecodeStarved, signal.VerdictStarved},
		{PrefillBusy, signal.VerdictBusy},
		{CacheBlocked, signal.VerdictBlocked},
		{ThermalThrottled, signal.VerdictThrottled},
		{NoProfiling, signal.VerdictUnknown},
	}
	for _, c := range cases {
		for i, s := range Generate(c.sc, 30, 7) {
			got := signal.Evaluate(s, nil, signal.DefaultThresholds())
			if got.Verdict != c.want {
				t.Errorf("%s sample %d: want %q, got %q (%v)", c.sc, i, c.want, got.Verdict, got.Reasons)
				break
			}
		}
	}
}

// The no-profiling scenario is the one that guards rule 1. Its absent counters
// must stay absent, and it must carry a reason a human can read.
func TestTheNoProfilingScenarioLeavesCountersAbsentWithAReason(t *testing.T) {
	for _, s := range Generate(NoProfiling, 5, 1) {
		for name, m := range map[string]collect.Metric{
			"sm_occupancy": s.SMOccupancy, "tensor_active": s.TensorActive, "dram_active": s.DRAMActive,
		} {
			if m.IsPresent() {
				t.Errorf("%s must be absent on a driver without profiling, not zero", name)
			}
			if s.Unavailable[name] == "" {
				t.Errorf("%s is absent with no reason recorded", name)
			}
		}
	}
}

func TestTheMixedFleetCoversEveryVerdictSoTheTopDemoIsHonest(t *testing.T) {
	seen := map[signal.Verdict]bool{}
	for _, s := range Generate(MixedFleet, 12, 6) {
		seen[signal.Evaluate(s, nil, signal.DefaultThresholds()).Verdict] = true
	}
	for _, want := range []signal.Verdict{
		signal.VerdictStarved, signal.VerdictBusy, signal.VerdictBlocked,
		signal.VerdictThrottled, signal.VerdictUnknown,
	} {
		if !seen[want] {
			t.Errorf("the mixed-fleet trace must contain a %q GPU", want)
		}
	}
}
