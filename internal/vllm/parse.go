// Package vllm reads serving state from a vLLM instance's Prometheus endpoint.
//
// The GPU counters say a node is busy. Only these say why, and the two verdicts
// that pay for this project depend on them: `blocked` needs the KV cache and the
// queue, and the decode explanation needs the batch size. Without them a
// fenced-off GPU and an underused one look identical, and someone buys capacity
// when the fix was configuration.
//
// Two properties govern everything here. Nothing absent is rendered as zero: a
// node with no inference server is the normal case for training and reports
// absent, not idle. And no rate is emitted from a single scrape, because a
// counter carries no rate until a second observation exists.
package vllm

import (
	"bufio"
	"io"
	"math"
	"strconv"
	"strings"
)

// series is one metric name with one label set, as exposed once per scrape.
type series struct {
	name   string
	labels map[string]string
	value  float64
}

// parse reads the Prometheus text exposition format.
//
// Deliberately not a full implementation: no histogram bucket reassembly, no
// exemplars, no timestamps. It reads `name{labels} value` lines and nothing
// else, because that is all vLLM emits that this package reads, and a partial
// parser that is honest about its scope beats a dependency here. The core of
// this project imports stdlib only and that constraint is worth more than
// completeness we would not use.
//
// Malformed lines are skipped rather than failing the scrape. A single
// unparseable line in an exposition page should not cost the caller every other
// metric on it; a metric that matters and did not parse surfaces as absent,
// with a reason, which is the behaviour the rest of this project guarantees.
func parse(r io.Reader) []series {
	var out []series
	sc := bufio.NewScanner(r)
	// Exposition pages are long but individual lines are short. The default
	// 64 KB token cap is generous here; raising it costs nothing and removes a
	// failure mode that would silently truncate a scrape.
	sc.Buffer(make([]byte, 0, 64*1024), 512*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		// `# HELP` and `# TYPE` carry no value. Skipping comments also skips
		// the type declarations, which means this parser does not know a
		// counter from a gauge. That knowledge lives in the collector, where
		// the decision about rate conversion is made explicitly per metric
		// rather than inferred from a line the server might omit.
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		s, ok := parseLine(line)
		if !ok {
			continue
		}
		out = append(out, s)
	}
	// A read error mid-page is treated the same as a short page: the caller
	// finds the metrics it needs absent, with a reason, rather than receiving a
	// partial set that looks complete.
	return out
}

func parseLine(line string) (series, bool) {
	name := line
	labels := map[string]string{}

	if i := strings.IndexByte(line, '{'); i >= 0 {
		j := strings.LastIndexByte(line, '}')
		if j < i {
			return series{}, false
		}
		name = line[:i]
		labels = parseLabels(line[i+1 : j])
		line = line[j+1:]
	} else {
		i := strings.IndexAny(line, " \t")
		if i < 0 {
			return series{}, false
		}
		name = line[:i]
		line = line[i:]
	}

	v, ok := parseValue(strings.TrimSpace(line))
	if !ok {
		return series{}, false
	}
	return series{name: strings.TrimSpace(name), labels: labels, value: v}, true
}

// parseLabels handles the quoted-value form vLLM emits. Escapes are limited to
// the three the exposition format defines.
func parseLabels(s string) map[string]string {
	out := map[string]string{}
	for _, part := range splitLabels(s) {
		eq := strings.IndexByte(part, '=')
		if eq < 0 {
			continue
		}
		k := strings.TrimSpace(part[:eq])
		v := strings.TrimSpace(part[eq+1:])
		v = strings.TrimPrefix(v, `"`)
		v = strings.TrimSuffix(v, `"`)
		v = strings.NewReplacer(`\"`, `"`, `\\`, `\`, `\n`, "\n").Replace(v)
		if k != "" {
			out[k] = v
		}
	}
	return out
}

// splitLabels splits on commas that are not inside a quoted value. A model name
// containing a comma is unusual but splitting naively would corrupt every label
// after it, and silently.
func splitLabels(s string) []string {
	var parts []string
	var b strings.Builder
	inQuote, escaped := false, false
	for _, r := range s {
		switch {
		case escaped:
			b.WriteRune(r)
			escaped = false
		case r == '\\' && inQuote:
			b.WriteRune(r)
			escaped = true
		case r == '"':
			inQuote = !inQuote
			b.WriteRune(r)
		case r == ',' && !inQuote:
			parts = append(parts, b.String())
			b.Reset()
		default:
			b.WriteRune(r)
		}
	}
	if b.Len() > 0 {
		parts = append(parts, b.String())
	}
	return parts
}

// parseValue rejects NaN and the infinities rather than letting them through.
//
// Prometheus uses NaN for "no observation yet", which is exactly the case this
// project refuses to render as a number. Returning false here makes it absent
// with a reason, instead of a NaN that propagates through every arithmetic
// operation downstream and prints as "NaN" on a dashboard.
func parseValue(s string) (float64, bool) {
	// An exposition line may carry a trailing millisecond timestamp. Value
	// first, timestamp second; the timestamp is not read.
	if i := strings.IndexAny(s, " \t"); i >= 0 {
		s = s[:i]
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return v, true
}
