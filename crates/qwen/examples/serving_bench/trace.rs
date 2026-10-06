//! Finite open-loop traffic over the real Engine; no executor or scheduling policy.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ribn::{Engine, Event, FinishReason, GenerationOptions, TokenRequest};
use serde::{Deserialize, Serialize};

use super::options::read_fixture_prompt;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    arrival_ms: u64,
    output_tokens: u32,
    prompt_fixture: PathBuf,
    prompt_tokens: Option<usize>,
}

pub(super) struct ScheduledRequest {
    pub arrival: Duration,
    pub prompt: Arc<[u32]>,
    pub output_tokens: u32,
}

pub(super) struct Workload {
    pub inputs: Vec<Input>,
    pub requests: Vec<ScheduledRequest>,
    pub context_tokens: u32,
    pub max_prompt_tokens: u32,
}

impl Workload {
    pub fn read(path: &Path) -> Result<Self, String> {
        let file =
            std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let inputs: Vec<Input> = serde_json::from_reader(file)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if inputs.is_empty() || u32::try_from(inputs.len()).is_err() {
            return Err("workload must contain 1..=u32::MAX requests".to_owned());
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut fixtures = HashMap::<PathBuf, Vec<u32>>::new();
        let mut prefixes = HashMap::<(PathBuf, usize), Arc<[u32]>>::new();
        let mut requests = Vec::with_capacity(inputs.len());
        let mut context_tokens = 0;
        let mut max_prompt_tokens = 0;
        let mut output_total = 0_u32;
        let mut previous_arrival = 0;
        for input in &inputs {
            if input.arrival_ms < previous_arrival || input.output_tokens == 0 {
                return Err(
                    "arrivals must be nondecreasing and output_tokens must be positive".to_owned(),
                );
            }
            previous_arrival = input.arrival_ms;
            let fixture = parent.join(&input.prompt_fixture);
            if !fixtures.contains_key(&fixture) {
                let tokens = read_fixture_prompt(&fixture)?;
                fixtures.insert(fixture.clone(), tokens);
            }
            let tokens = &fixtures[&fixture];
            let length = input.prompt_tokens.unwrap_or(tokens.len());
            if length == 0 || length > tokens.len() {
                return Err(format!(
                    "{}: requested prefix {length}, fixture has {} tokens",
                    fixture.display(),
                    tokens.len()
                ));
            }
            let prompt_tokens = u32::try_from(length).map_err(|_| "prompt length exceeds u32")?;
            let reach = prompt_tokens
                .checked_add(input.output_tokens)
                .ok_or("request reach overflowed")?;
            context_tokens = context_tokens.max(reach);
            max_prompt_tokens = max_prompt_tokens.max(prompt_tokens);
            output_total = output_total
                .checked_add(input.output_tokens)
                .ok_or("workload output total exceeds u32")?;
            let prompt = prefixes
                .entry((fixture, length))
                .or_insert_with(|| Arc::from(&tokens[..length]));
            requests.push(ScheduledRequest {
                arrival: Duration::from_millis(input.arrival_ms),
                prompt: Arc::clone(prompt),
                output_tokens: input.output_tokens,
            });
        }
        if Instant::now()
            .checked_add(requests.last().expect("nonempty workload").arrival)
            .is_none()
        {
            return Err("workload arrival exceeds the monotonic clock range".to_owned());
        }
        Ok(Self {
            inputs,
            requests,
            context_tokens,
            max_prompt_tokens,
        })
    }
}

#[derive(Default, Serialize)]
#[expect(
    clippy::struct_field_names,
    reason = "exported objective values carry millisecond units"
)]
pub(super) struct Objectives {
    pub ttft_ms: Option<u64>,
    pub itl_ms: Option<u64>,
    pub e2e_ms: Option<u64>,
}

impl Objectives {
    pub fn configured(&self) -> bool {
        self.ttft_ms.is_some() || self.itl_ms.is_some() || self.e2e_ms.is_some()
    }

    fn meets(&self, first: Duration, max_itl: Duration, end: Duration) -> bool {
        [
            (self.ttft_ms, first),
            (self.itl_ms, max_itl),
            (self.e2e_ms, end),
        ]
        .into_iter()
        .all(|(limit, value)| limit.is_none_or(|ms| value <= Duration::from_millis(ms)))
    }
}

#[derive(Serialize)]
#[serde(tag = "status", content = "error", rename_all = "snake_case")]
pub(super) enum Outcome {
    Completed,
    Rejected(String),
    Failed(String),
}

#[derive(Default)]
struct Observation {
    attempted: Duration,
    first: Option<Duration>,
    last: Option<Duration>,
    end: Duration,
    inter_token: Vec<Duration>,
    tokens: u32,
    token_log: Vec<u32>,
    outcome: Option<Outcome>,
}

#[derive(Serialize)]
pub(super) struct RequestReport {
    pub arrival_ms: f64,
    pub submit_lag_ms: f64,
    pub prompt_tokens: usize,
    pub requested_output_tokens: u32,
    pub emitted_tokens: u32,
    pub ttft_ms: Option<f64>,
    pub e2e_ms: f64,
    pub max_itl_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub outcome: Outcome,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<u32>,
}

#[derive(Serialize)]
#[expect(
    clippy::struct_field_names,
    reason = "exported latency values carry millisecond units"
)]
pub(super) struct Distribution {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

impl Distribution {
    fn of(mut values: Vec<Duration>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        let count = u32::try_from(values.len()).expect("workload sample counts bounded by u32");
        // Nearest rank, without multiplying the entire sample count by percent.
        let percentile = |percent: usize| {
            let rank = values.len() / 100 * percent + (values.len() % 100 * percent).div_ceil(100);
            ms(values[rank.max(1) - 1])
        };
        Some(Self {
            mean_ms: values.iter().map(|value| ms(*value)).sum::<f64>() / f64::from(count),
            p50_ms: percentile(50),
            p95_ms: percentile(95),
            p99_ms: percentile(99),
            max_ms: ms(*values.last().expect("nonempty samples")),
        })
    }
}

#[derive(Serialize)]
pub(super) struct Measurements {
    pub elapsed_s: f64,
    pub offered: u32,
    pub rejected: u32,
    pub failed: u32,
    pub completed: u32,
    pub emitted_tokens: u32,
    pub completed_tokens_per_s: f64,
    pub completed_requests_per_s: f64,
    pub slo_completed: Option<u32>,
    pub goodput_requests_per_s: Option<f64>,
    pub ttft: Option<Distribution>,
    pub itl: Option<Distribution>,
    pub e2e: Option<Distribution>,
    pub submit_lag: Option<Distribution>,
    pub requests: Vec<RequestReport>,
}

impl Measurements {
    pub fn print(&self) {
        println!("Qwen ribn::Engine open-loop trace (not an HTTP benchmark)");
        println!(
            "  offered: {}, completed: {}, rejected: {}, failed: {}",
            self.offered, self.completed, self.rejected, self.failed
        );
        println!(
            "  elapsed: {:.3} s; completed throughput: {:.2} tok/s, {:.2} requests/s",
            self.elapsed_s, self.completed_tokens_per_s, self.completed_requests_per_s
        );
        if let Some(goodput) = self.goodput_requests_per_s {
            println!(
                "  SLO goodput: {goodput:.2} requests/s ({} completions)",
                self.slo_completed.expect("configured objectives")
            );
        }
        for (label, distribution) in [
            ("TTFT", &self.ttft),
            ("ITL", &self.itl),
            ("E2E", &self.e2e),
            ("submit lag", &self.submit_lag),
        ] {
            if let Some(value) = distribution {
                println!(
                    "  {label} ms: mean {:.3}, p50 {:.3}, p95 {:.3}, p99 {:.3}, max {:.3}",
                    value.mean_ms, value.p50_ms, value.p95_ms, value.p99_ms, value.max_ms
                );
            }
        }
        println!(
            "  latency starts at planned arrival; submit lag is included, not erased. Latency distributions include successful requests only (submit lag includes all attempts)."
        );
    }
}

pub(super) fn measure(
    engine: &mut Engine,
    workload: &Workload,
    objectives: &Objectives,
    print_tokens: bool,
) -> Result<Measurements, String> {
    let mut observations = (0..workload.requests.len())
        .map(|_| Observation::default())
        .collect::<Vec<_>>();
    let mut pending = HashMap::new();
    let mut next = 0;
    // Preloaded input and client bookkeeping are outside the arrival timeline.
    let started = Instant::now();
    while next < workload.requests.len() || !pending.is_empty() {
        // Open-loop: never delay the planned clock or retry a rejected arrival.
        // If Engine::step held the host, overdue arrivals retain their original time.
        while next < workload.requests.len() && workload.requests[next].arrival <= started.elapsed()
        {
            let request = &workload.requests[next];
            let observation = &mut observations[next];
            observation.attempted = started.elapsed();
            match engine.enqueue(TokenRequest::new(
                Arc::clone(&request.prompt),
                GenerationOptions {
                    max_output_tokens: request.output_tokens,
                    ..GenerationOptions::default()
                },
            )) {
                Ok(id) => {
                    pending.insert(id, next);
                }
                Err(error) => {
                    observation.end = started.elapsed();
                    observation.outcome = Some(Outcome::Rejected(error.to_string()));
                }
            }
            next += 1;
        }
        if pending.is_empty() {
            if let Some(request) = workload.requests.get(next) {
                std::thread::sleep(request.arrival.saturating_sub(started.elapsed()));
            }
            continue;
        }
        let progress = engine.step().map_err(|error| error.to_string())?;
        while let Some(event) = engine.pop_event() {
            let index = *pending
                .get(&event.request())
                .ok_or("output belongs to an unknown/finished request")?;
            let observation = &mut observations[index];
            let now = started.elapsed();
            match event {
                Event::Token { token, .. } => {
                    observation.first.get_or_insert(now);
                    if let Some(previous) = observation.last.replace(now) {
                        observation.inter_token.push(
                            now.checked_sub(previous)
                                .expect("one monotonic measurement clock"),
                        );
                    }
                    observation.tokens += 1; // Sum of all offered budgets was prevalidated.
                    if print_tokens {
                        observation.token_log.push(token);
                    }
                }
                Event::Finished {
                    request,
                    reason,
                    usage,
                } => {
                    if usage.completion_tokens != observation.tokens
                        || (reason == FinishReason::Length
                            && observation.tokens != workload.requests[index].output_tokens)
                    {
                        return Err(format!(
                            "request {index}: streamed tokens and terminal usage/budget disagree"
                        ));
                    }
                    observation.end = now;
                    observation.outcome = Some(match reason {
                        FinishReason::Length => Outcome::Completed,
                        other => Outcome::Failed(format!("{other:?}")),
                    });
                    pending.remove(&request);
                }
            }
        }
        if !pending.is_empty() && !progress.completed && !progress.submitted {
            if !engine.status().in_flight {
                return Err(
                    "workload stalled with requests but no runnable/in-flight work".to_owned(),
                );
            }
            std::thread::yield_now();
        }
    }
    summarize(workload, observations, objectives, started.elapsed())
}

fn summarize(
    workload: &Workload,
    observations: Vec<Observation>,
    objectives: &Objectives,
    elapsed: Duration,
) -> Result<Measurements, String> {
    let mut ttft = Vec::new();
    let mut itl = Vec::new();
    let mut e2e = Vec::new();
    let mut submit_lag = Vec::new();
    let mut rejected = 0;
    let mut failed = 0;
    let mut completed = 0;
    let mut completed_tokens = 0;
    let mut emitted_tokens = 0;
    let mut slo_completed = 0;
    let mut requests = Vec::with_capacity(observations.len());
    for (request, observation) in workload.requests.iter().zip(observations) {
        let first = observation.first.map(|first| {
            first
                .checked_sub(request.arrival)
                .expect("only due requests are submitted")
        });
        let end = observation
            .end
            .checked_sub(request.arrival)
            .expect("only due requests are submitted");
        let lag = observation
            .attempted
            .checked_sub(request.arrival)
            .expect("only due requests are submitted");
        let max_itl = observation.inter_token.iter().copied().max();
        let outcome = observation
            .outcome
            .ok_or("request has no terminal/rejection outcome")?;
        match &outcome {
            Outcome::Completed => {
                let first = first.ok_or("completed request has no first token")?;
                completed += 1;
                completed_tokens += observation.tokens;
                ttft.push(first);
                e2e.push(end);
                itl.extend_from_slice(&observation.inter_token);
                if objectives.meets(first, max_itl.unwrap_or(Duration::ZERO), end) {
                    slo_completed += 1;
                }
            }
            Outcome::Rejected(_) => rejected += 1,
            Outcome::Failed(_) => failed += 1,
        }
        emitted_tokens += observation.tokens;
        submit_lag.push(lag);
        requests.push(RequestReport {
            arrival_ms: ms(request.arrival),
            submit_lag_ms: ms(lag),
            prompt_tokens: request.prompt.len(),
            requested_output_tokens: request.output_tokens,
            emitted_tokens: observation.tokens,
            ttft_ms: first.map(ms),
            e2e_ms: ms(end),
            max_itl_ms: max_itl.map(ms),
            tpot_ms: observation
                .last
                .zip(observation.first)
                .filter(|_| observation.tokens > 1)
                .map(|(last, first)| {
                    ms(last
                        .checked_sub(first)
                        .expect("one monotonic measurement clock"))
                        / f64::from(observation.tokens - 1)
                }),
            outcome,
            tokens: observation.token_log,
        });
    }
    let elapsed_s = elapsed.as_secs_f64();
    Ok(Measurements {
        elapsed_s,
        offered: u32::try_from(workload.requests.len()).expect("bounded workload"),
        rejected,
        failed,
        completed,
        emitted_tokens,
        completed_tokens_per_s: f64::from(completed_tokens) / elapsed_s,
        completed_requests_per_s: f64::from(completed) / elapsed_s,
        slo_completed: objectives.configured().then_some(slo_completed),
        goodput_requests_per_s: objectives
            .configured()
            .then_some(f64::from(slo_completed) / elapsed_s),
        ttft: Distribution::of(ttft),
        itl: Distribution::of(itl),
        e2e: Distribution::of(e2e),
        submit_lag: Distribution::of(submit_lag),
        requests,
    })
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_reader_validates_rows_before_execution_and_resolves_relative_fixtures() {
        let dir = std::env::temp_dir().join(format!(
            "ribn-trace-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("trace.json");
        std::fs::write(dir.join("prompt.tokens"), "artifact\n1 2 3\n4 5\n").unwrap();
        let row = serde_json::json!({"arrival_ms": 0, "output_tokens": 2, "prompt_fixture": "prompt.tokens", "prompt_tokens": 2});
        let valid = serde_json::json!([row.clone(), row.clone()]);
        std::fs::write(&path, valid.to_string()).unwrap();
        let workload = Workload::read(&path).unwrap();
        assert_eq!(
            (workload.context_tokens, workload.max_prompt_tokens),
            (4, 2)
        );
        assert_eq!(&*workload.requests[0].prompt, &[1, 2]);
        assert!(Arc::ptr_eq(
            &workload.requests[0].prompt,
            &workload.requests[1].prompt
        ));
        for invalid in [
            serde_json::json!([]),
            serde_json::json!([{"arrival_ms": 0, "output_tokens": 0, "prompt_fixture": "prompt.tokens"}]),
            serde_json::json!([{"arrival_ms": 0, "output_tokens": 2, "prompt_fixture": "prompt.tokens", "prompt_tokens": 4}]),
            serde_json::json!([{"arrival_ms": 0, "output_tokens": 4294967295_u32, "prompt_fixture": "prompt.tokens"}]),
            serde_json::json!([{"arrival_ms": 0, "output_tokens": 2, "prompt_fixture": "prompt.tokens", "typo": 1}]),
            serde_json::json!([{"arrival_ms": 1, "output_tokens": 2, "prompt_fixture": "prompt.tokens"}, row]),
        ] {
            std::fs::write(&path, invalid.to_string()).unwrap();
            assert!(Workload::read(&path).is_err(), "{invalid}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn planned_arrivals_include_dispatch_delay_and_failures_do_not_earn_goodput() {
        let request = || ScheduledRequest {
            arrival: Duration::from_millis(10),
            prompt: Arc::from([1]),
            output_tokens: 2,
        };
        let workload = Workload {
            inputs: vec![],
            requests: vec![request(), request(), request()],
            context_tokens: 3,
            max_prompt_tokens: 1,
        };
        let measured = summarize(
            &workload,
            vec![
                Observation {
                    attempted: Duration::from_millis(40),
                    first: Some(Duration::from_millis(50)),
                    last: Some(Duration::from_millis(70)),
                    end: Duration::from_millis(80),
                    tokens: 2,
                    inter_token: vec![Duration::from_millis(20)],
                    outcome: Some(Outcome::Completed),
                    ..Observation::default()
                },
                Observation {
                    attempted: Duration::from_millis(40),
                    end: Duration::from_millis(40),
                    outcome: Some(Outcome::Rejected("full".to_owned())),
                    ..Observation::default()
                },
                Observation {
                    attempted: Duration::from_millis(40),
                    end: Duration::from_millis(40),
                    outcome: Some(Outcome::Failed("device refusal".to_owned())),
                    ..Observation::default()
                },
            ],
            &Objectives {
                ttft_ms: Some(25),
                ..Objectives::default()
            },
            Duration::from_millis(100),
        )
        .unwrap();
        assert_eq!(
            (measured.completed, measured.rejected, measured.failed),
            (1, 1, 1)
        );
        assert_eq!(measured.requests[0].ttft_ms, Some(40.0));
        assert_eq!(measured.requests[0].submit_lag_ms, 30.0);
        assert_eq!(measured.requests[0].e2e_ms, 70.0);
        assert_eq!(measured.slo_completed, Some(0));
        assert_eq!(measured.completed_tokens_per_s, 20.0);
        assert_eq!(measured.itl.unwrap().p99_ms, 20.0);
    }

    #[test]
    fn percentiles_use_nearest_rank_and_one_token_has_no_itl_requirement() {
        let distribution =
            Distribution::of((1..=100).map(Duration::from_millis).collect()).unwrap();
        assert_eq!(
            (
                distribution.p50_ms,
                distribution.p95_ms,
                distribution.p99_ms
            ),
            (50.0, 95.0, 99.0)
        );
        assert!(
            Objectives {
                itl_ms: Some(1),
                ..Objectives::default()
            }
            .meets(
                Duration::from_secs(2),
                Duration::ZERO,
                Duration::from_secs(2)
            )
        );
        assert!(
            !Objectives {
                e2e_ms: Some(1),
                ..Objectives::default()
            }
            .meets(Duration::ZERO, Duration::ZERO, Duration::from_secs(2))
        );
    }
}
