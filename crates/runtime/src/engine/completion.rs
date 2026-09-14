//! Validate the entire submitted batch before committing any output or prefix.
use super::{Engine, WorkState};
use crate::{
    EngineError, Event, ExecutionError, FinishReason, StepCompletion, StepKind, StepOutcome,
};

/// What one validated row is allowed to do when the batch is committed.
enum RowPlan {
    Commit(StepCompletion),
    /// The backend reported that it could do nothing for this sequence. The row
    /// committed no progress and is resubmitted later.
    Blocked,
}

/// Result of observing one submission: whether it completed, and how many rows
/// reported that they could not proceed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CompletionReport {
    pub completed: bool,
    pub blocked: usize,
}

impl Engine {
    pub(super) fn poll_completion(&mut self) -> Result<CompletionReport, EngineError> {
        let Some(submission) = self.pending else {
            return Ok(CompletionReport::default());
        };
        let completion = self
            .model
            .as_mut()
            .expect("engine owns model")
            .poll(submission);
        let rows = match completion {
            Ok(None) => return Ok(CompletionReport::default()),
            Ok(Some(rows)) => rows,
            Err(error) => {
                self.fault_all(&error);
                self.flush_terminals()?;
                return Err(EngineError::Faulted(error));
            }
        };
        let Some(plans) = self.plan_completion(&rows) else {
            let error = ExecutionError::new(
                "completion does not match its submitted sequence, prefix, or output budget",
            );
            self.fault_all(&error);
            self.flush_terminals()?;
            return Err(EngineError::Faulted(error));
        };
        self.pending = None;
        self.release_output_reservations();
        // Every row was validated before any logical prefix or output is changed.
        let kinds = self.batch.iter().map(|item| item.kind).collect::<Vec<_>>();
        let slots = std::mem::take(&mut self.batch_slots);
        let mut blocked = 0_usize;
        for ((plan, index), kind) in plans.into_iter().zip(slots).zip(kinds) {
            match plan {
                RowPlan::Commit(row) => self.commit_row(index, row),
                RowPlan::Blocked => {
                    blocked += 1;
                    let sequence = self.slots[index]
                        .as_mut()
                        .expect("validated completion slot");
                    if sequence.work == WorkState::Cancelling {
                        self.terminate(index, FinishReason::Cancelled);
                        continue;
                    }
                    // The settled submission no longer accesses the sequence.
                    sequence.work = WorkState::Idle;
                    match kind {
                        StepKind::Prefill => self.prefill.push_back(index),
                        StepKind::Decode => self.decode.push_back(index),
                    }
                }
            }
        }
        self.batch.clear();
        Ok(CompletionReport {
            completed: true,
            blocked,
        })
    }

    /// Validate every row and decide what each may commit. `None` means the
    /// batch is malformed and nothing may be committed.
    fn plan_completion(&self, rows: &[StepOutcome]) -> Option<Vec<RowPlan>> {
        if rows.len() != self.batch.len() {
            return None;
        }
        rows.iter()
            .zip(&self.batch)
            .zip(&self.batch_slots)
            .map(|((row, item), &index)| {
                let sequence = self.slots[index].as_ref()?;
                if sequence.work == WorkState::Idle || sequence.id != item.sequence {
                    return None;
                }
                match row {
                    StepOutcome::Blocked(blocked) => {
                        if *blocked != item.sequence || sequence.prefix != item.prefix {
                            return None;
                        }
                        Some(RowPlan::Blocked)
                    }
                    StepOutcome::Progress(row) => {
                        if row.sequence != item.sequence || sequence.prefix != item.prefix {
                            return None;
                        }
                        let advance = row.prefix.checked_sub(item.prefix)?;
                        let outputs = u32::try_from(row.tokens.len()).ok()?;
                        let accepted = match item.kind {
                            // A prefill row may accept fewer inputs than its budget,
                            // which is how a backend stops before work it cannot do.
                            // Sampled output belongs only to the row that completed
                            // the prompt; an intermediate one produces none.
                            StepKind::Prefill => {
                                advance <= item.token_budget
                                    && if row.prefix == sequence.prompt_tokens {
                                        outputs == item.output_budget
                                    } else {
                                        outputs == 0
                                    }
                            }
                            // Decode is not allowed to make an empty successful step:
                            // a backend with nothing to commit reports `Blocked`.
                            StepKind::Decode => {
                                advance > 0
                                    && advance <= item.token_budget
                                    && outputs == advance
                                    && outputs <= item.output_budget
                            }
                        };
                        if accepted {
                            Some(RowPlan::Commit(row.clone()))
                        } else {
                            None
                        }
                    }
                }
            })
            .collect()
    }

    fn commit_row(&mut self, index: usize, row: StepCompletion) {
        let sequence = self.slots[index]
            .as_mut()
            .expect("validated completion slot");
        let cancelled = sequence.work == WorkState::Cancelling;
        sequence.work = WorkState::Idle;
        sequence.prefix = row.prefix;
        if cancelled {
            self.terminate(index, FinishReason::Cancelled);
            return;
        }
        let mut reason = None;
        for token in row.tokens {
            sequence.generated += 1;
            if sequence.options.stop_tokens.contains(&token) {
                reason = Some(FinishReason::Stop);
                break;
            }
            self.output.push_to(
                sequence.output,
                Event::Token {
                    request: sequence.request,
                    token,
                },
            );
            if sequence.generated == sequence.options.max_output_tokens {
                reason = Some(FinishReason::Length);
                break;
            }
        }
        if let Some(reason) = reason {
            self.terminate(index, reason);
        } else if sequence.prefix < sequence.prompt_tokens {
            self.prefill.push_back(index);
        } else {
            self.decode.push_back(index);
        }
    }
}
