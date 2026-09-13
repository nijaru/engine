//! Test-only pressure model for VLM-style encoder dependencies during AR prefill.
//!
//! This deliberately does not define Ribn's production multimodal API. It asks a
//! narrower question: what information must a generation scheduler/resource layer
//! know in order to advance a prompt without either eagerly encoding every media
//! item or entering a placeholder span whose encoder output is unavailable?
//!
//! Everything here is a pure function over a proposed decision, so passing these
//! tests shows the decision is *expressible*, not that the engine can carry it.
//! `multimodal_admission.rs` asks that second question through the real admission,
//! submission, and completion loop, and records where the two answers differ: a
//! prefill step must advance exactly the chunk the engine chose, so the range that
//! stops before an unavailable placeholder can only be produced by choosing a
//! policy chunk whose granularity matches the prompt's encoder items.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ItemId(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EncoderItem {
    id: ItemId,
    token_start: usize,
    token_end: usize,
    compute_units: usize,
    cache_units: usize,
}

impl EncoderItem {
    fn overlaps(self, start: usize, end: usize) -> bool {
        self.token_start < end && start < self.token_end
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Budgets {
    tokens: usize,
    encoder_compute: usize,
    encoder_cache: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct PrefillDecision {
    tokens: usize,
    encode: Vec<ItemId>,
    encoder_compute: usize,
    encoder_cache: usize,
}

/// Provisional scheduling oracle for one request's next prefill range.
///
/// The function knows nothing about images, audio, processors, tensor layouts, or
/// device handles. It sees only model-prepared prompt dependencies and two encoder
/// resource budgets. An uncached item can execute in the same iteration as the
/// prompt range that consumes it when both budgets permit it. Otherwise token
/// progress stops immediately before that item's placeholder span.
fn schedule_prefill(
    prefix: usize,
    prompt_tokens: usize,
    items: &[EncoderItem],
    cached: &HashSet<ItemId>,
    budgets: Budgets,
) -> PrefillDecision {
    let mut end = prefix.saturating_add(budgets.tokens).min(prompt_tokens);
    let mut compute_left = budgets.encoder_compute;
    let mut cache_left = budgets.encoder_cache;
    let mut encode = Vec::new();
    let mut used_compute = 0;
    let mut used_cache = 0;

    for item in items {
        if !item.overlaps(prefix, end) || cached.contains(&item.id) {
            continue;
        }

        if item.compute_units <= compute_left && item.cache_units <= cache_left {
            compute_left -= item.compute_units;
            cache_left -= item.cache_units;
            used_compute += item.compute_units;
            used_cache += item.cache_units;
            encode.push(item.id);
            continue;
        }

        end = end.min(item.token_start.max(prefix));
        break;
    }

    PrefillDecision {
        tokens: end.saturating_sub(prefix),
        encode,
        encoder_compute: used_compute,
        encoder_cache: used_cache,
    }
}

fn image(id: u32, token_start: usize, token_end: usize, cost: usize) -> EncoderItem {
    EncoderItem {
        id: ItemId(id),
        token_start,
        token_end,
        compute_units: cost,
        cache_units: cost,
    }
}

#[test]
fn future_encoder_item_is_not_eagerly_scheduled_before_its_prompt_span() {
    let items = [image(1, 6, 10, 4)];
    let decision = schedule_prefill(
        0,
        12,
        &items,
        &HashSet::new(),
        Budgets {
            tokens: 4,
            encoder_compute: 8,
            encoder_cache: 8,
        },
    );

    assert_eq!(
        decision,
        PrefillDecision {
            tokens: 4,
            encode: Vec::new(),
            encoder_compute: 0,
            encoder_cache: 0,
        }
    );
}

#[test]
fn prompt_progress_stops_before_uncached_item_when_encoder_compute_is_insufficient() {
    let items = [image(1, 4, 7, 6)];
    let decision = schedule_prefill(
        0,
        10,
        &items,
        &HashSet::new(),
        Budgets {
            tokens: 8,
            encoder_compute: 4,
            encoder_cache: 16,
        },
    );

    assert_eq!(decision.tokens, 4);
    assert!(decision.encode.is_empty());
    assert_eq!(decision.encoder_compute, 0);
}

#[test]
fn encoder_item_can_share_iteration_with_the_prompt_range_that_consumes_it() {
    let items = [image(1, 4, 7, 6)];
    let decision = schedule_prefill(
        0,
        10,
        &items,
        &HashSet::new(),
        Budgets {
            tokens: 8,
            encoder_compute: 6,
            encoder_cache: 6,
        },
    );

    assert_eq!(decision.tokens, 8);
    assert_eq!(decision.encode, vec![ItemId(1)]);
    assert_eq!(decision.encoder_compute, 6);
    assert_eq!(decision.encoder_cache, 6);
}

#[test]
fn cached_encoder_output_crosses_placeholder_without_compute_budget() {
    let items = [image(1, 4, 7, 6)];
    let cached = HashSet::from([ItemId(1)]);
    let decision = schedule_prefill(
        0,
        10,
        &items,
        &cached,
        Budgets {
            tokens: 8,
            encoder_compute: 0,
            encoder_cache: 0,
        },
    );

    assert_eq!(decision.tokens, 8);
    assert!(decision.encode.is_empty());
    assert_eq!(decision.encoder_compute, 0);
    assert_eq!(decision.encoder_cache, 0);
}

#[test]
fn encoder_cache_pressure_is_distinct_from_encoder_compute_pressure() {
    let items = [image(1, 4, 7, 6)];
    let decision = schedule_prefill(
        0,
        10,
        &items,
        &HashSet::new(),
        Budgets {
            tokens: 8,
            encoder_compute: 12,
            encoder_cache: 4,
        },
    );

    assert_eq!(decision.tokens, 4);
    assert!(decision.encode.is_empty());
    assert_eq!(decision.encoder_compute, 0);
    assert_eq!(decision.encoder_cache, 0);
}

#[test]
fn later_item_can_truncate_chunk_after_an_earlier_item_is_scheduled() {
    let items = [image(1, 2, 4, 3), image(2, 7, 9, 4)];
    let decision = schedule_prefill(
        0,
        12,
        &items,
        &HashSet::new(),
        Budgets {
            tokens: 10,
            encoder_compute: 5,
            encoder_cache: 8,
        },
    );

    assert_eq!(decision.tokens, 7);
    assert_eq!(decision.encode, vec![ItemId(1)]);
    assert_eq!(decision.encoder_compute, 3);
    assert_eq!(decision.encoder_cache, 3);
}

#[test]
fn starting_at_unready_placeholder_stalls_tokens_until_encoder_resource_exists() {
    let items = [image(1, 4, 7, 6)];
    let decision = schedule_prefill(
        4,
        10,
        &items,
        &HashSet::new(),
        Budgets {
            tokens: 3,
            encoder_compute: 0,
            encoder_cache: 8,
        },
    );

    assert_eq!(decision.tokens, 0);
    assert!(decision.encode.is_empty());
}
