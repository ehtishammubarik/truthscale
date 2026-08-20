package replay

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/ehtishammubarik/truthscale/internal/collect"
)

func write(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "t.jsonl")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// A corrupt line must not lose the rest of the trace, and must not vanish
// either. A trace that silently dropped records produces conclusions that are
// wrong with no sign anything happened.
func TestAMalformedLineIsCountedAndTheRestIsRead(t *testing.T) {
	p := write(t, `{"node":"a","gpu":0,"gpu_util":0.5}
not json at all
{"node":"a","gpu":1,"gpu_util":0.6}
`)
	r, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	got, err := r.All()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 good samples, got %d", len(got))
	}
	if r.Malformed != 1 {
		t.Fatalf("want 1 malformed counted, got %d", r.Malformed)
	}
}

func TestBlankLinesAndCommentsAreStructuralNotRecords(t *testing.T) {
	p := write(t, "\n# a note about this trace\n{\"node\":\"a\",\"gpu\":0}\n\n")
	r, _ := Open(p)
	defer r.Close()
	got, err := r.All()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || r.Malformed != 0 {
		t.Fatalf("want 1 sample and 0 malformed, got %d and %d", len(got), r.Malformed)
	}
}

// The round trip is what makes a trace trustworthy: absence has to survive it.
func TestWriteThenReadPreservesAbsence(t *testing.T) {
	in := []collect.Sample{{
		Node: "gpu-01", GPUUtil: collect.Known(0.99), SMOccupancy: collect.Absent(),
	}}
	var buf bytes.Buffer
	if err := Write(&buf, in); err != nil {
		t.Fatal(err)
	}
	p := write(t, buf.String())
	r, _ := Open(p)
	defer r.Close()
	got, err := r.All()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1, got %d", len(got))
	}
	if got[0].SMOccupancy.IsPresent() {
		t.Fatal("absence did not survive write then read")
	}
	if v, ok := got[0].GPUUtil.Get(); !ok || v != 0.99 {
		t.Fatalf("value corrupted: %v %v", v, ok)
	}
}

func TestOpenReportsAMissingFileRatherThanReturningNothing(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "nope.jsonl")); err == nil {
		t.Fatal("opening a missing trace must fail loudly")
	}
}
