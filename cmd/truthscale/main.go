// Command truthscale reports what a GPU is actually delivering, rather than
// what the utilization flag claims.
//
// Every subcommand accepts --replay, because a tool that cannot be tried
// without a GPU will not be tried.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/ehtishammubarik/truthscale/internal/ceiling"
	"github.com/ehtishammubarik/truthscale/internal/collect"
	"github.com/ehtishammubarik/truthscale/internal/render"
	"github.com/ehtishammubarik/truthscale/internal/replay"
	"github.com/ehtishammubarik/truthscale/internal/signal"
)

// Set by the release build via -ldflags. Reported by `truthscale version`, and
// recorded into any trace this binary writes, so a trace can be traced back to
// what produced it.
var version = "dev"

const usage = `truthscale - what your GPU is actually delivering

  nvidia-smi reports "utilization" and means "a kernel was resident". One small
  kernel on 1 of an H100's 132 SMs reads 100%. truthscale reports the number
  that means something instead.

Usage:
  truthscale top      [--replay FILE] [--last]     one row per GPU, the lie beside the truth
  truthscale explain  [--replay FILE] [--gpu N]    why the two numbers differ, in words
  truthscale version

Flags:
  --replay FILE   read samples from a recorded trace instead of live DCGM.
                  Use "-" for stdin. Traces ship in traces/.
  --last          evaluate only the final sample per GPU rather than all of them
  --gpu N         restrict to one GPU index

No GPU here? Every command works against a trace:

  truthscale top     --replay traces/h100-vllm-decode.jsonl
  truthscale explain --replay traces/h100-vllm-decode.jsonl
`

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "truthscale: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		fmt.Fprint(stdout, usage)
		return nil
	}

	cmd, rest := args[0], args[1:]
	switch cmd {
	case "top":
		return cmdTop(rest, stdout, stderr)
	case "explain":
		return cmdExplain(rest, stdout, stderr)
	case "version":
		fmt.Fprintf(stdout, "truthscale %s\n", version)
		return nil
	case "-h", "--help", "help":
		fmt.Fprint(stdout, usage)
		return nil
	default:
		fmt.Fprint(stderr, usage)
		return fmt.Errorf("unknown command %q", cmd)
	}
}

// flags shared by the reading commands.
type commonFlags struct {
	replayPath string
	last       bool
	gpu        int
}

func bind(fs *flag.FlagSet, c *commonFlags) {
	fs.StringVar(&c.replayPath, "replay", "", `read from a recorded trace; "-" for stdin`)
	fs.BoolVar(&c.last, "last", false, "evaluate only the final sample per GPU")
	fs.IntVar(&c.gpu, "gpu", -1, "restrict to one GPU index")
}

func cmdTop(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("top", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var c commonFlags
	bind(fs, &c)
	if err := fs.Parse(args); err != nil {
		return err
	}

	signals, malformed, err := load(c)
	if err != nil {
		return err
	}
	if len(signals) == 0 {
		return errors.New("no samples to report")
	}

	render.Table(stdout, signals)
	render.Summary(stdout, signals)
	reportMalformed(stderr, malformed)
	return nil
}

func cmdExplain(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("explain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var c commonFlags
	bind(fs, &c)
	if err := fs.Parse(args); err != nil {
		return err
	}
	// Explain is about one moment, so the newest sample per GPU is the useful
	// default rather than a wall of repeated prose.
	c.last = true

	signals, malformed, err := load(c)
	if err != nil {
		return err
	}
	if len(signals) == 0 {
		return errors.New("no samples to explain")
	}
	for i, s := range signals {
		if i > 0 {
			fmt.Fprintln(stdout)
		}
		fmt.Fprint(stdout, s.Explain())
	}
	reportMalformed(stderr, malformed)
	return nil
}

// load reads samples and evaluates them.
//
// Live DCGM collection is not in this release. Saying so plainly beats a stub
// that returns an empty table and lets someone conclude their GPUs are fine.
func load(c commonFlags) ([]signal.Signal, int, error) {
	if c.replayPath == "" {
		return nil, 0, errors.New(
			"live DCGM collection is not in v0.1. Pass --replay with a trace from traces/, " +
				"for example: truthscale top --replay traces/h100-vllm-decode.jsonl")
	}

	r, err := replay.Open(c.replayPath)
	if err != nil {
		return nil, 0, err
	}
	defer r.Close()

	samples, err := r.All()
	if err != nil {
		return nil, r.Malformed, err
	}

	if c.gpu >= 0 {
		samples = filter(samples, func(s collect.Sample) bool { return s.GPU == c.gpu })
	}
	if c.last {
		samples = lastPerGPU(samples)
	}

	thresholds := signal.DefaultThresholds()
	out := make([]signal.Signal, 0, len(samples))
	for _, s := range samples {
		// v0.1 has no ceiling store, so no ratio is reported rather than one
		// computed against a datasheet figure nobody measured.
		var c *ceiling.Ceiling
		out = append(out, signal.Evaluate(s, c, thresholds))
	}
	return out, r.Malformed, nil
}

func filter(in []collect.Sample, keep func(collect.Sample) bool) []collect.Sample {
	out := in[:0:0]
	for _, s := range in {
		if keep(s) {
			out = append(out, s)
		}
	}
	return out
}

// lastPerGPU keeps the newest sample for each node and GPU, ordered stably so
// output does not shuffle between runs.
func lastPerGPU(in []collect.Sample) []collect.Sample {
	type key struct {
		node string
		gpu  int
	}
	newest := map[key]collect.Sample{}
	for _, s := range in {
		k := key{s.Node, s.GPU}
		if prev, ok := newest[k]; !ok || s.Taken.After(prev.Taken) {
			newest[k] = s
		}
	}
	out := make([]collect.Sample, 0, len(newest))
	for _, s := range newest {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Node != out[j].Node {
			return out[i].Node < out[j].Node
		}
		return out[i].GPU < out[j].GPU
	})
	return out
}

// reportMalformed surfaces dropped lines. A trace that lost records is a trace
// whose conclusions are wrong, so this is never silent.
func reportMalformed(stderr io.Writer, n int) {
	if n > 0 {
		fmt.Fprintf(stderr, "\nwarning: %d line(s) in the trace did not parse and were skipped\n", n)
	}
}
