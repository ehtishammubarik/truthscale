// Command gen-traces regenerates the committed traces under traces/.
//
// Committed so a reviewer can see exactly how the shapes were chosen, and
// deterministic so regenerating produces no diff unless the generator changed.
// Run with `go run ./cmd/gen-traces`, verified in CI.
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/ehtishammubarik/truthscale/internal/replay"
	"github.com/ehtishammubarik/truthscale/internal/synth"
)

func main() {
	dir := "traces"
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// Seeds are fixed and listed here so the traces are reproducible.
	specs := []struct {
		sc      synth.Scenario
		samples int
		seed    int64
	}{
		{synth.DecodeStarved, 60, 1},
		{synth.PrefillBusy, 60, 2},
		{synth.CacheBlocked, 60, 3},
		{synth.ThermalThrottled, 60, 4},
		{synth.NoProfiling, 60, 5},
		{synth.MixedFleet, 12, 6},
	}

	for _, s := range specs {
		path := filepath.Join(dir, string(s.sc)+".jsonl")
		f, err := os.Create(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := replay.Write(f, synth.Generate(s.sc, s.samples, s.seed)); err != nil {
			f.Close()
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		f.Close()
		fmt.Printf("wrote %s\n", path)
	}
}
