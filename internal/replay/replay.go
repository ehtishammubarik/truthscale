// Package replay reads a recorded trace of samples from disk.
//
// This is not a test fixture. Most people evaluating a GPU tool do not have a
// spare GPU and will not attach one to try a stranger's binary, so replay is
// how the tool is demonstrated, reviewed, and tested. Every command supports
// it, and the traces in traces/ are committed.
//
// Traces are JSON Lines: one sample per line, appended as taken. That format is
// deliberate. It survives a truncated write, it streams without loading the
// file, and it is greppable, which matters when someone is trying to work out
// why a number looked wrong an hour ago.
package replay

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ehtishammubarik/truthscale/internal/collect"
)

// Reader streams samples from a trace file.
type Reader struct {
	scanner *bufio.Scanner
	closer  io.Closer
	line    int
	// Malformed lines are counted, never skipped silently. A trace that lost
	// records is a trace whose conclusions are wrong, and the user has to be
	// able to see that happened.
	Malformed int
}

// Open opens a trace for reading. "-" reads stdin, so a trace can be piped.
func Open(path string) (*Reader, error) {
	if path == "-" {
		return &Reader{scanner: newScanner(os.Stdin)}, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open trace: %w", err)
	}
	return &Reader{scanner: newScanner(f), closer: f}, nil
}

func newScanner(r io.Reader) *bufio.Scanner {
	s := bufio.NewScanner(r)
	// A sample with serving state and an unavailable map is comfortably under
	// 8 KB, but the default 64 KB token limit is not worth relying on.
	s.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return s
}

// Next returns the next sample. It reports io.EOF when the trace is exhausted.
//
// A line that does not parse is counted in Malformed and skipped, and reading
// continues: one corrupt line in a long trace should not lose the rest. The
// count is reported by the caller at the end, so nothing vanishes quietly.
func (r *Reader) Next() (collect.Sample, error) {
	for r.scanner.Scan() {
		r.line++
		text := strings.TrimSpace(r.scanner.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		var s collect.Sample
		if err := json.Unmarshal([]byte(text), &s); err != nil {
			r.Malformed++
			continue
		}
		return s, nil
	}
	if err := r.scanner.Err(); err != nil {
		return collect.Sample{}, fmt.Errorf("read trace at line %d: %w", r.line, err)
	}
	return collect.Sample{}, io.EOF
}

// All reads the whole trace. Convenient for short traces and for tests; use
// Next for anything long enough that holding it in memory matters.
func (r *Reader) All() ([]collect.Sample, error) {
	var out []collect.Sample
	for {
		s, err := r.Next()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return out, err
		}
		out = append(out, s)
	}
}

// Close releases the underlying file, if any.
func (r *Reader) Close() error {
	if r.closer != nil {
		return r.closer.Close()
	}
	return nil
}

// Write appends samples to a writer as JSON Lines. Used by the trace generator
// and by `truthscale top --record`.
func Write(w io.Writer, samples []collect.Sample) error {
	enc := json.NewEncoder(w)
	for i, s := range samples {
		if err := enc.Encode(s); err != nil {
			return fmt.Errorf("write sample %d: %w", i, err)
		}
	}
	return nil
}
