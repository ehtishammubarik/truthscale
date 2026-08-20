// Package render turns signals into the two views a human reads: a table for
// scanning many GPUs, and prose for understanding one.
package render

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/ehtishammubarik/truthscale/internal/collect"
	"github.com/ehtishammubarik/truthscale/internal/signal"
)

// Table writes one row per signal, with the misleading number and the honest
// one adjacent.
//
// The column order is the argument: UTIL sits next to OCCUPANCY sits next to
// GAP, so the discrepancy is visible without the reader having to know it
// exists. A table that put them in separate screens would be technically
// complete and would teach nobody anything.
func Table(w io.Writer, signals []signal.Signal) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NODE\tGPU\tVERDICT\tUTIL\tOCCUPANCY\tGAP\tCEILING\tKV CACHE\tQUEUE\tBATCH")

	for _, s := range signals {
		fmt.Fprintf(tw, "%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			s.Sample.Node,
			s.Sample.GPU,
			verdictLabel(s.Verdict),
			pct(s.Sample.GPUUtil),
			pct(s.Sample.SMOccupancy),
			gap(s.UtilGap),
			pct(s.CeilingRatio),
			servingPct(s.Sample, func(v collect.ServingState) collect.Metric { return v.KVCacheUtilization }),
			servingNum(s.Sample, func(v collect.ServingState) collect.Metric { return v.QueuedRequests }),
			servingNum(s.Sample, func(v collect.ServingState) collect.Metric { return v.BatchSize }),
		)
	}
	tw.Flush()
}

// Summary is the line printed under the table. It counts verdicts, because the
// number worth acting on is "how many of my GPUs are lying to me", not any
// single row.
func Summary(w io.Writer, signals []signal.Signal) {
	counts := map[signal.Verdict]int{}
	for _, s := range signals {
		counts[s.Verdict]++
	}
	order := []signal.Verdict{
		signal.VerdictStarved, signal.VerdictBlocked, signal.VerdictThrottled,
		signal.VerdictBusy, signal.VerdictIdle, signal.VerdictUnknown,
	}
	var parts []string
	for _, v := range order {
		if counts[v] > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", counts[v], v))
		}
	}
	fmt.Fprintf(w, "\n%d GPU(s): %s\n", len(signals), strings.Join(parts, ", "))

	if n := counts[signal.VerdictStarved]; n > 0 {
		fmt.Fprintf(w, "\n%d GPU(s) report high utilization while the SMs are mostly idle.\n", n)
		fmt.Fprintf(w, "Run `truthscale explain` on one of them to see why.\n")
	}
	if n := counts[signal.VerdictUnknown]; n > 0 {
		fmt.Fprintf(w, "\n%d GPU(s) could not be judged because a counter was unavailable.\n", n)
		fmt.Fprintf(w, "These are reported as unknown rather than assumed healthy. See docs/metrics.md.\n")
	}
}

func verdictLabel(v signal.Verdict) string { return strings.ToUpper(string(v)) }

func pct(m collect.Metric) string {
	v, ok := m.Get()
	if !ok {
		return "n/a"
	}
	return fmt.Sprintf("%.0f%%", v*100)
}

// gap is signed on purpose. A negative gap means occupancy exceeded the
// utilization flag, which should not happen and is worth seeing rather than
// hiding behind an absolute value.
func gap(m collect.Metric) string {
	v, ok := m.Get()
	if !ok {
		return "n/a"
	}
	return fmt.Sprintf("%+.0f", v*100)
}

func servingPct(s collect.Sample, pick func(collect.ServingState) collect.Metric) string {
	if s.Serving == nil {
		return "-"
	}
	return pct(pick(*s.Serving))
}

func servingNum(s collect.Sample, pick func(collect.ServingState) collect.Metric) string {
	if s.Serving == nil {
		return "-"
	}
	v, ok := pick(*s.Serving).Get()
	if !ok {
		return "n/a"
	}
	return fmt.Sprintf("%.0f", v)
}
