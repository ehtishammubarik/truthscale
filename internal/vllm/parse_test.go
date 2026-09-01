package vllm

import (
	"strings"
	"testing"
)

func TestParseReadsNameLabelsAndValue(t *testing.T) {
	got := parse(strings.NewReader(`vllm:num_requests_running{model_name="a/b"} 4.0`))
	if len(got) != 1 {
		t.Fatalf("want 1 series, got %d", len(got))
	}
	if got[0].name != "vllm:num_requests_running" {
		t.Errorf("name = %q", got[0].name)
	}
	if got[0].labels["model_name"] != "a/b" {
		t.Errorf("model_name = %q", got[0].labels["model_name"])
	}
	if got[0].value != 4 {
		t.Errorf("value = %v", got[0].value)
	}
}

func TestParseHandlesUnlabelledSeries(t *testing.T) {
	got := parse(strings.NewReader("vllm:num_requests_running 3"))
	if len(got) != 1 || got[0].value != 3 {
		t.Fatalf("got %+v", got)
	}
}

// NaN is Prometheus for "no observation yet". Letting it through would put a
// NaN into arithmetic that feeds a verdict, and print "NaN" as though measured.
func TestParseRejectsNaNAndInfinities(t *testing.T) {
	for _, v := range []string{"NaN", "+Inf", "-Inf", "nonsense", ""} {
		got := parse(strings.NewReader("vllm:x{a=\"b\"} " + v))
		if len(got) != 0 {
			t.Errorf("value %q should not parse, got %+v", v, got)
		}
	}
}

func TestParseSkipsCommentsAndMalformedLines(t *testing.T) {
	in := `# HELP vllm:x help text
# TYPE vllm:x gauge
vllm:x{a="b"} 1.0
this line is garbage
vllm:y 2.0
`
	got := parse(strings.NewReader(in))
	if len(got) != 2 {
		t.Fatalf("want 2 series, got %d: %+v", len(got), got)
	}
}

// A comma inside a quoted label value must not split the label list, or every
// label after it is silently corrupted.
func TestParseHandlesCommaInsideQuotedLabel(t *testing.T) {
	got := parse(strings.NewReader(`vllm:x{model_name="org/a,b",engine="v1"} 5`))
	if len(got) != 1 {
		t.Fatalf("want 1 series, got %d", len(got))
	}
	if got[0].labels["model_name"] != "org/a,b" {
		t.Errorf("model_name = %q, want org/a,b", got[0].labels["model_name"])
	}
	if got[0].labels["engine"] != "v1" {
		t.Errorf("engine = %q, want v1", got[0].labels["engine"])
	}
}

func TestParseIgnoresTrailingTimestamp(t *testing.T) {
	got := parse(strings.NewReader(`vllm:x{a="b"} 7 1699999999000`))
	if len(got) != 1 || got[0].value != 7 {
		t.Fatalf("got %+v", got)
	}
}
