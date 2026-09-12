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
        if !self.valid_completion(&rows) {
            let error = ExecutionError::new(
                "completion does not match its submitted sequence, prefix, or output budget",
            );
            self.fault_all(&error);
            self.flush_terminals()?;
            return Err(EngineError::Faulted(error));
        }
        self.pending = None;
        self.release_output_reservations();
        // Every row was checked before any logical prefix or output is changed.
        for (row_index, row) in rows.into_iter().enumerate() {
            let index = self.batch_slots[row_index];
            self.commit_row(index, row);
        }
        self.batch.clear();
        self.batch_slots.clear();
        Ok(true)
    }

    fn valid_completion(&self, rows: &[StepCompletion]) -> bool {
        rows.len() == self.batch.len()
            && rows
                .iter()
                .zip(&self.batch)
                .zip(&self.batch_slots)
                .all(|((row, item), &index)| {
                    let Some(sequence) = self.slots[index].as_ref() else {
                        return false;
                    };
                    if sequence.work == WorkState::Idle
                        || sequence.id != row.sequence
                        || row.sequence != item.sequence
                        || sequence.prefix != item.prefix
                    {
                        return false;
                    }
                    let Some(advance) = row.prefix.checked_sub(item.prefix) else {
                        return false;
                    };
                    let Ok(outputs) = u32::try_from(row.tokens.len()) else {
                        return false;
                    };
                    match item.kind {
                        StepKind::Prefill => {
                            advance == item.token_budget && outputs == item.output_budget
                        }
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
