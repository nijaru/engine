//! Validate the entire submitted batch before committing any output or prefix.
use super::{Engine, WorkState};
use crate::{EngineError, Event, ExecutionError, FinishReason, StepCompletion, StepKind};

impl Engine {
    pub(super) fn poll_completion(&mut self) -> Result<bool, EngineError> {
        let Some(submission) = self.pending else {
            return Ok(false);
        };
        let completion = self
            .model
            .as_mut()
            .expect("engine owns model")
            .poll(submission);
        let rows = match completion {
            Ok(None) => return Ok(false),
            Ok(Some(rows)) => rows,
            Err(error) => {
                self.fault_all(&error);
                self.flush_terminals()?;
                return Err(EngineError::Faulted(error));
            }
        };
        // Every row is validated before any logical prefix or output changes.
        if !self.completion_is_valid(&rows) {
            let error = ExecutionError::new(
                "completion does not match its submitted sequence, prefix, or output budget",
            );
            self.fault_all(&error);
            self.flush_terminals()?;
            return Err(EngineError::Faulted(error));
        }
        self.pending = None;
        self.release_output_reservations();
        for (position, row) in rows.into_iter().enumerate() {
            self.commit_row(self.batch_slots[position], row);
        }
        self.batch.clear();
        self.batch_slots.clear();
        Ok(true)
    }

    /// Validate every row against the settled batch. `false` means the batch is
    /// malformed and nothing may be committed.
    fn completion_is_valid(&self, rows: &[StepCompletion]) -> bool {
        if rows.len() != self.batch.len() {
            return false;
        }
        rows.iter()
            .zip(&self.batch)
            .zip(&self.batch_slots)
            .all(|((row, item), &index)| {
                let Some(sequence) = self.slots[index].as_ref() else {
                    return false;
                };
                if sequence.work == WorkState::Idle || sequence.id != item.sequence {
                    return false;
                }
                if row.sequence != item.sequence || sequence.prefix != item.prefix {
                    return false;
                }
                let Some(advance) = row.prefix.checked_sub(item.prefix) else {
                    return false;
                };
                let Ok(outputs) = u32::try_from(row.tokens.len()) else {
                    return false;
                };
                match item.kind {
                    // A prefill row may accept fewer inputs than its budget, but
                    // only positive progress is a completion. Sampled output
                    // belongs only to the row that completed the prompt; an
                    // intermediate one produces none.
                    StepKind::Prefill => {
                        advance > 0
                            && advance <= item.token_budget
                            && if row.prefix == sequence.prompt_tokens {
                                outputs == 1 && outputs <= item.output_budget
                            } else {
                                outputs == 0
                            }
                    }
                    // Decode is not allowed to make an empty successful step: a
                    // backend with nothing to commit must not report a completion.
                    StepKind::Decode => {
                        advance > 0
                            && advance <= item.token_budget
                            && outputs == advance
                            && outputs <= item.output_budget
                    }
                }
            })
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
