// Package signal turns a raw sample into the numbers worth acting on, and can
// explain in words why they differ from the numbers people usually watch.
//
// The headline is CeilingRatio: delivered divided by measured ceiling. It is the
// only figure here that answers "should this node exist?", which is the question
// a GPU fleet is actually asking.
package signal

import (
	"fmt"
	"sort"
	"strings"

	"github.com/ehtishammubarik/truthscale/internal/ceiling"
	"github.com/ehtishammubarik/truthscale/internal/collect"
)

// Verdict is the shape of the gap, not a recommendation. Recommendations need
// history and policy and arrive in v0.3; this is what one instant supports.
type Verdict string

const (
	// VerdictBusy means the GPU is genuinely doing work near its ceiling.
	VerdictBusy Verdict = "busy"
	// VerdictIdle means it is doing nothing, and says so honestly.
	VerdictIdle Verdict = "idle"
	// VerdictStarved is the interesting one: the GPU looks busy and is not.
	// A resident kernel holding GPUUtil high while the machine does little.
	VerdictStarved Verdict = "starved"
	// VerdictThrottled means capability is being withheld by power or heat, so
	// no software conclusion may be drawn from this sample.
	VerdictThrottled Verdict = "throttled"
	// VerdictBlocked means work is queued and cannot be admitted, usually
	// because the KV cache is full. Adding GPUs may not help.
	VerdictBlocked Verdict = "blocked"
	// VerdictUnknown means the counters needed were unavailable. Reported
	// rather than guessed.
	VerdictUnknown Verdict = "unknown"
)

// Thresholds are the boundaries between verdicts. Exposed and documented rather
// than buried, because every one of them is a judgement someone may disagree
// with, and disagreement should be a config change and not a fork.
type Thresholds struct {
	// Above this, GPUUtil claims the GPU is busy.
	UtilBusy float64
	// Below this, SM occupancy says it is not really.
	OccupancyLow float64
	// Below this ceiling ratio, delivered throughput is far under capability.
	CeilingLow float64
	// At or above this, the KV cache is full enough to block admission.
	KVCacheFull float64
	// Above this many queued requests, work is waiting.
	QueueWaiting float64
}

// DefaultThresholds are starting points, not truths.
//
// UtilBusy at 0.80 and OccupancyLow at 0.30 come from the documented shape of
// autoregressive decode, where GPUUtil pins high while occupancy sits near 15%.
// A workload of large fused kernels legitimately runs at low occupancy and high
// efficiency, so on such a fleet these will misfire and should be changed.
func DefaultThresholds() Thresholds {
	return Thresholds{
		UtilBusy:     0.80,
		OccupancyLow: 0.30,
		CeilingLow:   0.50,
		KVCacheFull:  0.90,
		QueueWaiting: 1,
	}
}

// Signal is the derived view of one sample.
type Signal struct {
	Sample  collect.Sample `json:"sample"`
	Verdict Verdict        `json:"verdict"`

	// CeilingRatio is delivered over measured ceiling, absent when no
	// trustworthy ceiling exists. Absent is common and is not a failure.
	CeilingRatio collect.Metric `json:"ceiling_ratio"`
	// Which dimension the ratio was computed on, since it depends what the
	// ceiling covers.
	CeilingBasis string `json:"ceiling_basis,omitempty"`

	// UtilGap is GPUUtil minus SM occupancy: how much of the apparent busyness
	// is a resident kernel rather than work. The number this project exists for.
	UtilGap collect.Metric `json:"util_gap"`

	// Reasons the verdict is what it is, most important first.
	Reasons []string `json:"reasons,omitempty"`
}

// Evaluate derives the signal for one sample against a ceiling.
//
// A nil ceiling is fine. Everything except CeilingRatio still works, which
// matters because a node whose ceiling has not been measured yet should still
// report the util gap rather than nothing at all.
func Evaluate(s collect.Sample, c *ceiling.Ceiling, t Thresholds) Signal {
	out := Signal{Sample: s, Verdict: VerdictUnknown, CeilingRatio: collect.Absent(), UtilGap: collect.Absent()}

	util, hasUtil := s.GPUUtil.Get()
	occ, hasOcc := s.SMOccupancy.Get()

	if hasUtil && hasOcc {
		out.UtilGap = collect.Known(util - occ)
	}

	out.CeilingRatio, out.CeilingBasis = ratio(s, c)

	// Throttling first. Rule 2: nothing about software may be concluded from a
	// sample where the hardware was being held back, so this short-circuits.
	if throttled, reasons := s.Throttled(); throttled {
		out.Verdict = VerdictThrottled
		names := make([]string, 0, len(reasons))
		for _, r := range reasons {
			names = append(names, string(r))
		}
		sort.Strings(names)
		out.Reasons = append(out.Reasons,
			fmt.Sprintf("the GPU is throttled (%s), so any slowdown is hardware before it is software", strings.Join(names, ", ")))
		return out
	}

	// Blocked before starved. If the cache is full and requests are queued, the
	// GPU is not underused, it is fenced off, and another GPU may not fix it.
	if s.Serving != nil {
		kv, hasKV := s.Serving.KVCacheUtilization.Get()
		q, hasQ := s.Serving.QueuedRequests.Get()
		if hasKV && kv >= t.KVCacheFull && hasQ && q >= t.QueueWaiting {
			out.Verdict = VerdictBlocked
			out.Reasons = append(out.Reasons,
				fmt.Sprintf("the KV cache is %.0f%% full with %.0f request(s) queued, so no new sequence can be admitted", kv*100, q))
			out.Reasons = append(out.Reasons,
				"adding a GPU raises capacity; raising the cache or shortening max sequence length may be cheaper")
			return out
		}
	}

	if !hasUtil {
		out.Reasons = append(out.Reasons, "GPU utilization was not reported, so no verdict is possible")
		return out
	}

	// The case this tool exists for: looks busy, is not.
	if util >= t.UtilBusy {
		if !hasOcc {
			out.Verdict = VerdictUnknown
			out.Reasons = append(out.Reasons,
				fmt.Sprintf("utilization is %.0f%%, but SM occupancy was unavailable, so whether that is real work is unknown", util*100))
			return out
		}
		if occ < t.OccupancyLow {
			out.Verdict = VerdictStarved
			out.Reasons = append(out.Reasons,
				fmt.Sprintf("utilization reads %.0f%% while only %.0f%% of the SMs are occupied: a kernel is resident, the machine is mostly idle", util*100, occ*100))
			out.Reasons = append(out.Reasons, decodeHint(s)...)
			return out
		}
		out.Verdict = VerdictBusy
		out.Reasons = append(out.Reasons,
			fmt.Sprintf("utilization %.0f%% is backed by %.0f%% SM occupancy: the GPU is genuinely working", util*100, occ*100))
		return out
	}

	if util < 0.05 {
		out.Verdict = VerdictIdle
		out.Reasons = append(out.Reasons, fmt.Sprintf("utilization is %.0f%%: nothing is running", util*100))
		return out
	}

	if r, ok := out.CeilingRatio.Get(); ok && r < t.CeilingLow {
		out.Verdict = VerdictStarved
		out.Reasons = append(out.Reasons,
			fmt.Sprintf("delivering %.0f%% of this node's measured ceiling", r*100))
		return out
	}

	out.Verdict = VerdictBusy
	out.Reasons = append(out.Reasons, fmt.Sprintf("utilization is %.0f%% with no signal of waste", util*100))
	return out
}

// ratio computes delivered over ceiling, preferring the dimension that most
// directly reflects delivered work.
//
// Token throughput first: it is what a serving node is for, and it cannot be
// faked by a resident kernel. SM occupancy second, as a hardware-level proxy.
func ratio(s collect.Sample, c *ceiling.Ceiling) (collect.Metric, string) {
	if c == nil {
		return collect.Absent(), ""
	}
	if ok, _ := c.Trustworthy(); !ok {
		return collect.Absent(), ""
	}
	if c.HasTokenThroughput() && s.Serving != nil {
		if tps, ok := s.Serving.OutputTokensPerSec.Get(); ok {
			return collect.Known(tps / c.OutputTokensPerSec), "output_tokens_per_sec"
		}
	}
	if c.HasSMOccupancy() {
		if occ, ok := s.SMOccupancy.Get(); ok {
			return collect.Known(occ / c.SMOccupancyCeiling), "sm_occupancy"
		}
	}
	return collect.Absent(), ""
}

// decodeHint adds the serving-side explanation when the numbers match the
// signature of autoregressive decode. Only offered when the evidence is there.
func decodeHint(s collect.Sample) []string {
	if s.Serving == nil {
		return nil
	}
	var out []string
	if b, ok := s.Serving.BatchSize.Get(); ok && b <= 2 {
		out = append(out, fmt.Sprintf("the batch is %.0f: one small kernel per token keeps utilization pinned while the tensor cores idle", b))
	}
	if kv, ok := s.Serving.KVCacheUtilization.Get(); ok && kv < 0.25 {
		out = append(out, fmt.Sprintf("the KV cache is only %.0f%% used, so there is room to batch more work onto this GPU", kv*100))
	}
	return out
}

// Explain renders the reasoning as prose. This is the command people screenshot,
// so it says what was observed and what follows from it, and nothing else.
func (s Signal) Explain() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s gpu%d", s.Sample.Node, s.Sample.GPU)
	if s.Sample.Model != "" {
		fmt.Fprintf(&b, " (%s)", s.Sample.Model)
	}
	fmt.Fprintf(&b, ": %s\n\n", strings.ToUpper(string(s.Verdict)))

	fmt.Fprintf(&b, "  what the dashboard says   utilization %s\n", pct(s.Sample.GPUUtil))
	fmt.Fprintf(&b, "  what the SMs say          occupancy   %s\n", pct(s.Sample.SMOccupancy))
	if g, ok := s.UtilGap.Get(); ok {
		fmt.Fprintf(&b, "  the gap                   %.0f points of apparent busyness that is not work\n", g*100)
	}
	if r, ok := s.CeilingRatio.Get(); ok {
		fmt.Fprintf(&b, "  against measured ceiling  %.0f%% (%s)\n", r*100, s.CeilingBasis)
	} else {
		fmt.Fprintf(&b, "  against measured ceiling  n/a (no trustworthy ceiling for this node yet)\n")
	}

	if len(s.Reasons) > 0 {
		b.WriteString("\n")
		for _, r := range s.Reasons {
			fmt.Fprintf(&b, "  - %s\n", r)
		}
	}

	if len(s.Sample.Unavailable) > 0 {
		b.WriteString("\n  not measured:\n")
		keys := make([]string, 0, len(s.Sample.Unavailable))
		for k := range s.Sample.Unavailable {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(&b, "    %-22s %s\n", k, s.Sample.Unavailable[k])
		}
	}
	return b.String()
}

func pct(m collect.Metric) string {
	v, ok := m.Get()
	if !ok {
		return "n/a"
	}
	return fmt.Sprintf("%.0f%%", v*100)
}
