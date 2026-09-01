package vllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ehtishammubarik/truthscale/internal/collect"
)

// serve returns a server whose body the test can swap between scrapes.
func serve(t *testing.T, body *string, status *int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if status != nil && *status != 0 {
			w.WriteHeader(*status)
		}
		_, _ = w.Write([]byte(*body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func fixture(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("testdata/vllm-decode.txt")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(b)
}

func mustGet(t *testing.T, m interface{ Get() (float64, bool) }, field string) float64 {
	t.Helper()
	v, ok := m.Get()
	if !ok {
		t.Fatalf("%s absent, want present", field)
	}
	return v
}

func mustAbsent(t *testing.T, m interface{ IsPresent() bool }, field string, un map[string]string) {
	t.Helper()
	if m.IsPresent() {
		t.Errorf("%s present, want absent", field)
	}
	if un[field] == "" {
		t.Errorf("%s absent but no reason recorded in unavailable", field)
	}
}

// The gauges are readable from one scrape; the rates are not. This is the
// central guarantee of the issue.
func TestFirstScrapeReadsGaugesAndWithholdsRates(t *testing.T) {
	body := fixture(t)
	srv := serve(t, &body, nil)
	c := New(srv.URL)

	st, un, err := c.Scrape(context.Background(), time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	if st == nil {
		t.Fatal("serving state nil, want populated")
	}
	if st.Engine != "vllm" {
		t.Errorf("engine = %q", st.Engine)
	}
	if got := mustGet(t, st.KVCacheUtilization, "kv_cache_utilization"); got != 0.97 {
		t.Errorf("kv_cache_utilization = %v, want 0.97", got)
	}
	if got := mustGet(t, st.RunningRequests, "running_requests"); got != 1 {
		t.Errorf("running_requests = %v, want 1", got)
	}
	if got := mustGet(t, st.QueuedRequests, "queued_requests"); got != 7 {
		t.Errorf("queued_requests = %v, want 7", got)
	}
	// The sentence the whole thesis rests on: cache full, queue deep, batch 1.
	if got := mustGet(t, st.BatchSize, "batch_size"); got != 1 {
		t.Errorf("batch_size = %v, want 1", got)
	}

	mustAbsent(t, st.OutputTokensPerSec, "output_tokens_per_sec", un)
	mustAbsent(t, st.PromptTokensPerSec, "prompt_tokens_per_sec", un)
	mustAbsent(t, st.QueueSeconds, "queue_seconds", un)
	mustAbsent(t, st.MaxBatchSize, "max_batch_size", un)

	if models := c.Models(); len(models) != 1 || models[0] != "meta-llama/Llama-2-7b-hf" {
		t.Errorf("models = %v", models)
	}
}

func TestSecondScrapeProducesRates(t *testing.T) {
	body := fixture(t)
	srv := serve(t, &body, nil)
	c := New(srv.URL)
	ctx := context.Background()

	if _, _, err := c.Scrape(ctx, time.Unix(1000, 0)); err != nil {
		t.Fatalf("first scrape: %v", err)
	}

	// 10 seconds later: +2000 generation tokens, +1000 prompt tokens,
	// +100 queued requests completed accumulating +50s of wait.
	body = `vllm:gpu_cache_usage_perc{model_name="m"} 0.5
vllm:num_requests_running{model_name="m"} 8
vllm:num_requests_waiting{model_name="m"} 0
vllm:generation_tokens_total{model_name="m"} 52000
vllm:prompt_tokens_total{model_name="m"} 101000
vllm:request_queue_time_seconds_sum{model_name="m"} 850
vllm:request_queue_time_seconds_count{model_name="m"} 500
`
	st, un, err := c.Scrape(ctx, time.Unix(1010, 0))
	if err != nil {
		t.Fatalf("second scrape: %v", err)
	}
	if got := mustGet(t, st.OutputTokensPerSec, "output_tokens_per_sec"); got != 200 {
		t.Errorf("output_tokens_per_sec = %v, want 200", got)
	}
	if got := mustGet(t, st.PromptTokensPerSec, "prompt_tokens_per_sec"); got != 100 {
		t.Errorf("prompt_tokens_per_sec = %v, want 100", got)
	}
	// 50 seconds of accumulated wait over 100 completions.
	if got := mustGet(t, st.QueueSeconds, "queue_seconds"); got != 0.5 {
		t.Errorf("queue_seconds = %v, want 0.5", got)
	}
	if un["output_tokens_per_sec"] != "" {
		t.Errorf("rate present but a reason was recorded: %q", un["output_tokens_per_sec"])
	}
}

// A restart is the case the issue calls out by name. The interval spans it, so
// its true rate is unknowable and must be absent rather than negative.
func TestCounterResetReportsAbsentNotNegative(t *testing.T) {
	body := fixture(t)
	srv := serve(t, &body, nil)
	c := New(srv.URL)
	ctx := context.Background()
	if _, _, err := c.Scrape(ctx, time.Unix(1000, 0)); err != nil {
		t.Fatalf("first scrape: %v", err)
	}

	body = `vllm:gpu_cache_usage_perc{model_name="m"} 0.1
vllm:num_requests_running{model_name="m"} 0
vllm:num_requests_waiting{model_name="m"} 0
vllm:generation_tokens_total{model_name="m"} 12
vllm:prompt_tokens_total{model_name="m"} 5
vllm:request_queue_time_seconds_sum{model_name="m"} 1
vllm:request_queue_time_seconds_count{model_name="m"} 1
`
	st, un, err := c.Scrape(ctx, time.Unix(1010, 0))
	if err != nil {
		t.Fatalf("second scrape: %v", err)
	}
	for _, tc := range []struct {
		field string
		m     interface{ IsPresent() bool }
	}{
		{"output_tokens_per_sec", st.OutputTokensPerSec},
		{"prompt_tokens_per_sec", st.PromptTokensPerSec},
		{"queue_seconds", st.QueueSeconds},
	} {
		mustAbsent(t, tc.m, tc.field, un)
	}
	if want := "counter decreased, vLLM restarted during the interval"; un["output_tokens_per_sec"] != want {
		t.Errorf("reason = %q, want %q", un["output_tokens_per_sec"], want)
	}
}

// A zero or reversed interval must never divide: +Inf and negative rates both
// render as measurements.
func TestNonPositiveIntervalNeverDivides(t *testing.T) {
	for _, name := range []string{"equal", "reversed"} {
		t.Run(name, func(t *testing.T) {
			body := fixture(t)
			srv := serve(t, &body, nil)
			c := New(srv.URL)
			ctx := context.Background()
			first := time.Unix(1000, 0)
			if _, _, err := c.Scrape(ctx, first); err != nil {
				t.Fatalf("first scrape: %v", err)
			}
			second := first
			if name == "reversed" {
				second = first.Add(-5 * time.Second)
			}
			body = `vllm:gpu_cache_usage_perc{model_name="m"} 0.5
vllm:generation_tokens_total{model_name="m"} 99999
vllm:prompt_tokens_total{model_name="m"} 99999
vllm:num_requests_running{model_name="m"} 1
vllm:num_requests_waiting{model_name="m"} 0
vllm:request_queue_time_seconds_sum{model_name="m"} 9999
vllm:request_queue_time_seconds_count{model_name="m"} 9999
`
			st, un, err := c.Scrape(ctx, second)
			if err != nil {
				t.Fatalf("second scrape: %v", err)
			}
			mustAbsent(t, st.OutputTokensPerSec, "output_tokens_per_sec", un)
			mustAbsent(t, st.QueueSeconds, "queue_seconds", un)
		})
	}
}

// No request finishing queueing is not a queue of zero seconds. Zero is a much
// more reassuring statement than "nothing to report" and would be a lie.
func TestNoQueueCompletionsInIntervalIsAbsentNotZero(t *testing.T) {
	body := `vllm:gpu_cache_usage_perc{model_name="m"} 0.5
vllm:num_requests_running{model_name="m"} 2
vllm:num_requests_waiting{model_name="m"} 3
vllm:generation_tokens_total{model_name="m"} 10
vllm:prompt_tokens_total{model_name="m"} 10
vllm:request_queue_time_seconds_sum{model_name="m"} 42
vllm:request_queue_time_seconds_count{model_name="m"} 9
`
	srv := serve(t, &body, nil)
	c := New(srv.URL)
	ctx := context.Background()
	if _, _, err := c.Scrape(ctx, time.Unix(1000, 0)); err != nil {
		t.Fatalf("first scrape: %v", err)
	}
	st, un, err := c.Scrape(ctx, time.Unix(1010, 0)) // identical body: no deltas
	if err != nil {
		t.Fatalf("second scrape: %v", err)
	}
	mustAbsent(t, st.QueueSeconds, "queue_seconds", un)
	if want := "no requests completed queueing during the interval"; un["queue_seconds"] != want {
		t.Errorf("reason = %q, want %q", un["queue_seconds"], want)
	}
	// A genuinely unchanged counter is a real rate of zero, not absence.
	if got := mustGet(t, st.OutputTokensPerSec, "output_tokens_per_sec"); got != 0 {
		t.Errorf("output_tokens_per_sec = %v, want 0", got)
	}
}

// No serving stack is the normal case for a training node. It must be absent
// with a reason, and it must not fail the caller, who still has a GPU to report.
func TestNoServerIsAbsentAndNotAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close() // nothing is listening now

	c := New(url)
	st, un, err := c.Scrape(context.Background(), time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("a node with no serving stack must not be an error, got %v", err)
	}
	if st != nil {
		t.Errorf("serving state = %+v, want nil", st)
	}
	if un["serving"] == "" {
		t.Error("no reason recorded for absent serving state")
	}
}

func TestNonVLLMEndpointIsDistinguishedFromNoServer(t *testing.T) {
	body := "go_goroutines 12\nprocess_cpu_seconds_total 3.4\n"
	srv := serve(t, &body, nil)
	c := New(srv.URL)
	st, un, err := c.Scrape(context.Background(), time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	if st != nil {
		t.Errorf("serving state = %+v, want nil", st)
	}
	if got := un["serving"]; got == "" || !strings.Contains(got, "no vllm: metrics") {
		t.Errorf("reason = %q, want it to name the missing vllm: metrics", got)
	}
}

func TestHTTPErrorStatusIsAbsentWithReason(t *testing.T) {
	body := ""
	status := http.StatusServiceUnavailable
	srv := serve(t, &body, &status)
	c := New(srv.URL)
	st, un, err := c.Scrape(context.Background(), time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	if st != nil {
		t.Errorf("serving state = %+v, want nil", st)
	}
	if !strings.Contains(un["serving"], "503") {
		t.Errorf("reason = %q, want it to name the status", un["serving"])
	}
}

// Two models on one GPU contend for it, so counts sum. Cache utilisation is a
// fraction of one physical cache, so it takes the maximum: summing would exceed
// 1.0 and a mean would hide a full cache behind an empty one.
func TestMultiModelAggregation(t *testing.T) {
	body := `vllm:gpu_cache_usage_perc{model_name="a"} 0.95
vllm:gpu_cache_usage_perc{model_name="b"} 0.10
vllm:num_requests_running{model_name="a"} 2
vllm:num_requests_running{model_name="b"} 3
vllm:num_requests_waiting{model_name="a"} 1
vllm:num_requests_waiting{model_name="b"} 4
`
	srv := serve(t, &body, nil)
	c := New(srv.URL)
	st, _, err := c.Scrape(context.Background(), time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	if got := mustGet(t, st.KVCacheUtilization, "kv_cache_utilization"); got != 0.95 {
		t.Errorf("kv_cache_utilization = %v, want 0.95 (max, not sum or mean)", got)
	}
	if got := mustGet(t, st.RunningRequests, "running_requests"); got != 5 {
		t.Errorf("running_requests = %v, want 5 (sum)", got)
	}
	if models := c.Models(); len(models) != 2 || models[0] != "a" || models[1] != "b" {
		t.Errorf("models = %v, want [a b] sorted", models)
	}
}

// A build that omits a metric reports that field absent, not zero, and the rest
// of the scrape still lands.
func TestMissingMetricIsAbsentWithReason(t *testing.T) {
	body := `vllm:num_requests_running{model_name="m"} 4
vllm:gpu_cache_usage_perc{model_name="m"} 0.2
`
	srv := serve(t, &body, nil)
	c := New(srv.URL)
	st, un, err := c.Scrape(context.Background(), time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	mustAbsent(t, st.QueuedRequests, "queued_requests", un)
	if got := mustGet(t, st.RunningRequests, "running_requests"); got != 4 {
		t.Errorf("running_requests = %v, want 4", got)
	}
}

// The scrape must round trip through a trace without absence becoming zero,
// which is the property collect.Metric exists to guarantee.
func TestServingStateSurvivesJSONRoundTrip(t *testing.T) {
	body := fixture(t)
	srv := serve(t, &body, nil)
	c := New(srv.URL)
	st, _, err := c.Scrape(context.Background(), time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	if st.OutputTokensPerSec.IsPresent() {
		t.Fatal("precondition: rate should be absent on first scrape")
	}
	roundTripped := roundTrip(t, st)
	if roundTripped.OutputTokensPerSec.IsPresent() {
		t.Error("absent rate became present through JSON")
	}
	if got := mustGet(t, roundTripped.KVCacheUtilization, "kv_cache_utilization"); got != 0.97 {
		t.Errorf("kv_cache_utilization = %v after round trip", got)
	}
}

func roundTrip(t *testing.T, in *collect.ServingState) *collect.ServingState {
	t.Helper()
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out collect.ServingState
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return &out
}
