package vllm

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/ehtishammubarik/truthscale/internal/collect"
)

// The metric names read from vLLM. Every one is documented in docs/metrics.md
// with its units and what it does not tell you.
const (
	metricCacheUsage   = "vllm:gpu_cache_usage_perc"
	metricRunning      = "vllm:num_requests_running"
	metricWaiting      = "vllm:num_requests_waiting"
	metricQueueSum     = "vllm:request_queue_time_seconds_sum"
	metricQueueCount   = "vllm:request_queue_time_seconds_count"
	metricGenTokens    = "vllm:generation_tokens_total"
	metricPromptTokens = "vllm:prompt_tokens_total"
)

// DefaultTimeout bounds a scrape. A vLLM instance under load can be slow to
// answer its own metrics endpoint, and a collector that blocks forever on it
// stops reporting the GPU too.
const DefaultTimeout = 3 * time.Second

// Collector scrapes one vLLM endpoint and converts its counters into rates.
//
// It is stateful, and has to be: a counter carries no rate until it has been
// observed twice. One Collector belongs to one endpoint, and reusing it across
// endpoints would compare a counter against an unrelated one and emit a rate
// that describes neither. Not safe for concurrent use.
type Collector struct {
	// URL is the full metrics endpoint, for example
	// http://127.0.0.1:8000/metrics.
	URL string
	// HTTP is optional. The zero value uses a client with DefaultTimeout.
	HTTP *http.Client

	prev   map[string]float64
	prevA  time.Time
	models []string
	// True once a scrape has succeeded. Distinguishes "no previous sample"
	// from "previous sample happened to be all zeroes", which matters because
	// the first reports absent and the second is a real rate of zero.
	seeded bool
}

// New returns a Collector for an endpoint.
func New(url string) *Collector { return &Collector{URL: url} }

// Scrape reads the endpoint and returns the serving state at instant `at`.
//
// `at` is supplied by the caller rather than read here on purpose. The GPU
// sample and this one have to sit on one clock for a throughput dip to be
// attributable to the GPU behaviour that caused it, and two pollers each
// calling time.Now() cannot promise that. The caller takes the instant once and
// passes it to both.
//
// A nil ServingState is not an error. No inference server on this node is the
// normal case for a training node or a bare one, and the reason is returned so
// a reader can tell it apart from a scrape that failed.
func (c *Collector) Scrape(ctx context.Context, at time.Time) (*collect.ServingState, map[string]string, error) {
	unavailable := map[string]string{}

	client := c.HTTP
	if client == nil {
		client = &http.Client{Timeout: DefaultTimeout}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.URL, nil)
	if err != nil {
		return nil, map[string]string{"serving": fmt.Sprintf("bad endpoint %q: %v", c.URL, err)}, err
	}
	req.Header.Set("Accept", "text/plain")

	resp, err := client.Do(req)
	if err != nil {
		// Refused, unresolvable, or timed out. Absent with a reason, and not an
		// error to the caller: a GPU node with no serving stack must still
		// report its GPU.
		return nil, map[string]string{"serving": fmt.Sprintf("no vLLM at %s: %v", c.URL, err)}, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, map[string]string{"serving": fmt.Sprintf("vLLM metrics returned HTTP %d", resp.StatusCode)}, nil
	}

	all := parse(resp.Body)
	cur, models := aggregate(all)
	if len(cur) == 0 {
		// Something answered, but it was not vLLM. Worth distinguishing from a
		// refused connection: the port is probably wrong rather than the node
		// having no serving stack.
		return nil, map[string]string{
			"serving": fmt.Sprintf("endpoint %s answered but exposed no vllm: metrics", c.URL),
		}, nil
	}

	st := &collect.ServingState{Engine: "vllm"}

	// Gauges. Read directly: an instantaneous value needs no history.
	st.KVCacheUtilization = gauge(cur, metricCacheUsage, unavailable, "kv_cache_utilization")
	st.RunningRequests = gauge(cur, metricRunning, unavailable, "running_requests")
	st.QueuedRequests = gauge(cur, metricWaiting, unavailable, "queued_requests")

	// batch_size is running_requests at this instant.
	//
	// vLLM exposes no mean-batch metric, and deriving one from tokens is not
	// possible without knowing sequence lengths. The instantaneous value is the
	// number that makes the decode case legible ("utilization is 1.0 and the
	// batch is 1"), so it is reported as measured rather than withheld pending
	// a window. A windowed mean arrives with the ceiling store, which is where
	// the history to compute one will already live.
	st.BatchSize = st.RunningRequests
	if !st.BatchSize.IsPresent() {
		unavailable["batch_size"] = "derived from running_requests, which was absent"
	}

	// MaxBatchSize has no metric. vLLM carries max_num_seqs in its engine
	// configuration and does not expose it on /metrics, so it is absent rather
	// than guessed from the highest running count seen, which would be a
	// high-water mark presented as a limit.
	st.MaxBatchSize = collect.Absent()
	unavailable["max_batch_size"] = "vLLM does not expose max_num_seqs on /metrics"

	// Rates, which need two observations.
	st.OutputTokensPerSec = c.rate(cur, metricGenTokens, at, unavailable, "output_tokens_per_sec")
	st.PromptTokensPerSec = c.rate(cur, metricPromptTokens, at, unavailable, "prompt_tokens_per_sec")

	// Mean queue wait over the interval, from the histogram's sum and count.
	//
	// Not the cumulative sum/count, which is the mean since the server started
	// and moves so slowly under load that it hides exactly the spike someone is
	// looking for.
	st.QueueSeconds = c.queueMean(cur, at, unavailable)

	c.prev, c.prevA, c.seeded, c.models = cur, at, true, models

	if len(unavailable) == 0 {
		unavailable = nil
	}
	return st, unavailable, nil
}

// Models reports the model names seen on the last successful scrape, sorted.
// Empty if no series carried a model_name label, which some vLLM builds omit.
//
// Exposed because a scrape aggregating two models is a different measurement
// from one aggregating a single model, and a reader comparing numbers across
// nodes needs to know which they have.
func (c *Collector) Models() []string { return c.models }

func gauge(cur map[string]float64, name string, unavailable map[string]string, field string) collect.Metric {
	v, ok := cur[name]
	if !ok {
		unavailable[field] = fmt.Sprintf("%s not exposed by this vLLM build", name)
		return collect.Absent()
	}
	return collect.Known(v)
}

// rate converts a monotonic counter into a per-second rate across the interval
// since the previous scrape.
func (c *Collector) rate(cur map[string]float64, name string, at time.Time, unavailable map[string]string, field string) collect.Metric {
	v, ok := cur[name]
	if !ok {
		unavailable[field] = fmt.Sprintf("%s not exposed by this vLLM build", name)
		return collect.Absent()
	}
	if !c.seeded {
		unavailable[field] = "first scrape: a rate needs two observations of a counter"
		return collect.Absent()
	}
	p, ok := c.prev[name]
	if !ok {
		unavailable[field] = fmt.Sprintf("%s absent from the previous scrape", name)
		return collect.Absent()
	}
	dt := at.Sub(c.prevA).Seconds()
	if dt <= 0 {
		// Equal or reversed timestamps. Dividing here produces +Inf or a
		// negative rate from a positive delta, both of which look like
		// measurements.
		unavailable[field] = "non-positive interval between scrapes"
		return collect.Absent()
	}
	d := v - p
	if d < 0 {
		// A counter only decreases when the process restarted. The interval
		// spans the restart, so its true rate is unknowable: the tokens served
		// before the restart are lost. Absent, not zero, and not the raw
		// post-restart value divided by the interval, which would understate it
		// arbitrarily.
		unavailable[field] = "counter decreased, vLLM restarted during the interval"
		return collect.Absent()
	}
	return collect.Known(d / dt)
}

// queueMean is delta(sum) / delta(count) for the queue-time histogram.
func (c *Collector) queueMean(cur map[string]float64, at time.Time, unavailable map[string]string) collect.Metric {
	const field = "queue_seconds"
	sum, okS := cur[metricQueueSum]
	count, okC := cur[metricQueueCount]
	if !okS || !okC {
		unavailable[field] = "request_queue_time_seconds sum or count not exposed"
		return collect.Absent()
	}
	if !c.seeded {
		unavailable[field] = "first scrape: a windowed mean needs two observations"
		return collect.Absent()
	}
	pSum, okPS := c.prev[metricQueueSum]
	pCount, okPC := c.prev[metricQueueCount]
	if !okPS || !okPC {
		unavailable[field] = "request_queue_time_seconds absent from the previous scrape"
		return collect.Absent()
	}
	if at.Sub(c.prevA) <= 0 {
		unavailable[field] = "non-positive interval between scrapes"
		return collect.Absent()
	}
	dSum, dCount := sum-pSum, count-pCount
	if dSum < 0 || dCount < 0 {
		unavailable[field] = "counter decreased, vLLM restarted during the interval"
		return collect.Absent()
	}
	if dCount == 0 {
		// No request finished queueing in this interval, so the window holds no
		// observation. Zero would claim an instant queue, which is a different
		// and much more reassuring statement than "nothing to report".
		unavailable[field] = "no requests completed queueing during the interval"
		return collect.Absent()
	}
	return collect.Known(dSum / dCount)
}

// aggregate folds a scrape's series into one value per metric name.
//
// vLLM normally serves one model per process, but the metrics carry a
// model_name label and a multi-model process would expose several series per
// name. Counts and counters sum, because two models queueing on one GPU do
// contend for it. Cache utilisation takes the maximum, because it is a fraction
// of one physical cache and the binding constraint is whichever is fullest;
// summing fractions would produce values above 1.0 and a mean would hide a full
// cache behind an empty one.
func aggregate(all []series) (map[string]float64, []string) {
	out := map[string]float64{}
	models := map[string]bool{}
	for _, s := range all {
		if m := s.labels["model_name"]; m != "" {
			models[m] = true
		}
		switch s.name {
		case metricCacheUsage:
			if v, ok := out[s.name]; !ok || s.value > v {
				out[s.name] = s.value
			}
		case metricRunning, metricWaiting, metricQueueSum, metricQueueCount,
			metricGenTokens, metricPromptTokens:
			out[s.name] += s.value
		}
	}
	names := make([]string, 0, len(models))
	for m := range models {
		names = append(names, m)
	}
	sort.Strings(names)
	return out, names
}
