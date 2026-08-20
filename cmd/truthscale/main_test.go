package main

import (
	"bytes"
	"strings"
	"testing"
)

func exec(t *testing.T, args ...string) (string, string, error) {
	t.Helper()
	var out, errb bytes.Buffer
	err := run(args, &out, &errb)
	return out.String(), errb.String(), err
}

// The most dangerous possible behaviour: returning an empty table when no
// collector exists would let someone conclude their fleet was fine. It must
// fail, and the failure must say what to do instead.
func TestTopWithoutReplayFailsAndSaysWhatToRun(t *testing.T) {
	_, _, err := exec(t, "top")
	if err == nil {
		t.Fatal("top with no collector and no trace must fail, not report an empty fleet")
	}
	msg := err.Error()
	if !strings.Contains(msg, "--replay") || !strings.Contains(msg, "traces/") {
		t.Fatalf("the error must name the flag and where traces live: %q", msg)
	}
}

func TestTopAgainstATraceReportsEveryVerdict(t *testing.T) {
	out, _, err := exec(t, "top", "--replay", "../../traces/mixed-fleet.jsonl", "--last")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"STARVED", "BUSY", "BLOCKED", "THROTTLED", "UNKNOWN"} {
		if !strings.Contains(out, want) {
			t.Errorf("the demo trace must produce a %s row:\n%s", want, out)
		}
	}
	// The README leads on this trace, so the claim has to hold.
	if !strings.Contains(out, "n/a") {
		t.Errorf("the demo must show n/a somewhere, or rule 1 is not visible:\n%s", out)
	}
}

func TestExplainNarrowsToOneGpu(t *testing.T) {
	out, _, err := exec(t, "explain", "--replay", "../../traces/mixed-fleet.jsonl", "--gpu", "0")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "what the dashboard says") {
		t.Fatalf("explain must show the pairing:\n%s", out)
	}
}

func TestExplainOnADecodeTraceNamesTheCause(t *testing.T) {
	out, _, err := exec(t, "explain", "--replay", "../../traces/h100-vllm-decode.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "STARVED") {
		t.Errorf("a decode trace should read starved:\n%s", out)
	}
	if !strings.Contains(out, "one small kernel per token") {
		t.Errorf("explain should name the decode cause:\n%s", out)
	}
}

// A trace whose counters are unavailable must produce unknown, and must say the
// driver is why. Reporting "healthy" here would be the original sin.
func TestATraceWithNoProfilingReportsUnknownAndTheReason(t *testing.T) {
	out, _, err := exec(t, "explain", "--replay", "../../traces/l4-no-profiling.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "UNKNOWN") {
		t.Errorf("absent counters must produce UNKNOWN:\n%s", out)
	}
	if !strings.Contains(out, "not measured:") || !strings.Contains(out, "driver") {
		t.Errorf("explain must say which counters were missing and why:\n%s", out)
	}
}

func TestAMissingTraceFailsLoudly(t *testing.T) {
	if _, _, err := exec(t, "top", "--replay", "does-not-exist.jsonl"); err == nil {
		t.Fatal("a missing trace must fail rather than report nothing")
	}
}

func TestUnknownCommandFailsAndPrintsUsage(t *testing.T) {
	_, errOut, err := exec(t, "frobnicate")
	if err == nil {
		t.Fatal("an unknown command must fail")
	}
	if !strings.Contains(errOut, "truthscale top") {
		t.Fatalf("usage should be printed on an unknown command:\n%s", errOut)
	}
}

func TestNoArgsPrintsUsageAndSucceeds(t *testing.T) {
	out, _, err := exec(t)
	if err != nil {
		t.Fatalf("bare invocation should succeed: %v", err)
	}
	// The thesis has to be in the help text: many people read only this.
	if !strings.Contains(out, "a kernel was resident") {
		t.Fatalf("usage must state the thesis:\n%s", out)
	}
}

func TestVersionPrints(t *testing.T) {
	out, _, err := exec(t, "version")
	if err != nil || !strings.Contains(out, "truthscale") {
		t.Fatalf("version: %q %v", out, err)
	}
}
