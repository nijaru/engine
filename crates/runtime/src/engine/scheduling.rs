//! Selection over engine-owned queues; no duplicate request or resource owner.
use super::{Engine, WorkState};
use crate::{BatchItem, StepKind};

impl Engine {
    pub(super) fn build_batch(&mut self) {
        self.batch.clear();
        self.batch_slots.clear();
        let mut budget = self.policy.max_batch_tokens;
        let force_prefill =
            !self.prefill.is_empty() && self.decode_only_steps >= self.policy.max_decode_only_steps;
        if force_prefill {
            self.schedule_kind(StepKind::Prefill, &mut budget, 1);
        }
        self.schedule_kind(StepKind::Decode, &mut budget, usize::MAX);
        self.schedule_kind(StepKind::Prefill, &mut budget, usize::MAX);
        if self.batch.iter().any(|item| item.kind == StepKind::Prefill) {
            self.decode_only_steps = 0;
        } else if !self.batch.is_empty() && !self.prefill.is_empty() {
            self.decode_only_steps = self.decode_only_steps.saturating_add(1);
        }
    }

    fn schedule_kind(&mut self, kind: StepKind, budget: &mut u32, limit: usize) {
        let queue = match kind {
            StepKind::Prefill => &mut self.prefill,
            StepKind::Decode => &mut self.decode,
        };
        let mut selected = 0;
        // Visit each ready row at most once. A full client mailbox does not
        // prevent later rows from using their own output credits.
        for _ in 0..queue.len() {
            if selected >= limit
                || self.batch.len() >= self.config.max_active_requests
                || *budget == 0
            {
                break;
            }
            let index = queue.pop_front().expect("nonempty ready queue");
            let sequence = self.slots[index].as_mut().expect("runnable slot exists");
            let credits = self.output.credits(sequence.output);
            let (tokens, outputs) = match kind {
                StepKind::Prefill => {
                    let tokens = (sequence.prompt_tokens - sequence.prefix)
                        .min(self.policy.prefill_chunk_tokens)
                        .min(*budget);
                    (
                        tokens,
                        u32::from(sequence.prefix + tokens == sequence.prompt_tokens),
                    )
                }
                StepKind::Decode => {
                    let output_credits =
                        u32::try_from(credits.saturating_sub(1)).unwrap_or(u32::MAX);
                    let tokens = self
                        .policy
                        .decode_tokens
                        .min(sequence.options.max_output_tokens - sequence.generated)
                        .min(*budget)
                        .min(output_credits);
                    (tokens, tokens)
                }
            };
            if tokens == 0 || u64::from(outputs) + 1 > credits as u64 {
                queue.push_back(index);
                continue;
            }
            self.output.reserve(sequence.output, outputs as usize + 1);
            sequence.work = WorkState::InFlight;
            self.batch.push(BatchItem {
                sequence: sequence.id,
                kind,
                prefix: sequence.prefix,
                token_budget: tokens,
                output_budget: outputs,
            });
            self.batch_slots.push(index);
            *budget -= tokens;
            selected += 1;
        }
    }

    pub(super) fn release_output_reservations(&mut self) {
        for &index in &self.batch_slots {
            let sequence = self.slots[index].as_ref().expect("submitted slot");
            self.output.unreserve(sequence.output);
        }
    }
}
