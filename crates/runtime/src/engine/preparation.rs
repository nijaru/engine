//! Pre-submit negotiation over the offered rows; the executor owns reservations.
use super::{Engine, WorkState};
use crate::{BatchPreparation, EngineError, ExecutionError, FinishReason, PreparedRow, StepKind};

impl Engine {
    pub(super) fn prepare_batch(&mut self) -> Result<bool, EngineError> {
        let prepared = match self
            .model
            .as_mut()
            .expect("engine owns model")
            .prepare(&self.batch)
        {
            Ok(prepared) => prepared,
            Err(error) => return self.preparation_fault(error),
        };
        match prepared {
            BatchPreparation::Ready => {
                for &index in &self.batch_slots {
                    self.slots[index].as_mut().expect("selected slot").work = WorkState::InFlight;
                }
            }
            BatchPreparation::Selected(rows) => {
                if !self.preparation_is_valid(&rows) {
                    let mut error = ExecutionError::new(
                        "preparation does not match its offered rows or budgets",
                    );
                    if let Err(cleanup) = self
                        .model
                        .as_mut()
                        .expect("engine owns model")
                        .abandon_preparation()
                    {
                        error = ExecutionError::new(format!(
                            "{error}; preparation abandonment failed: {cleanup}"
                        ));
                    }
                    return self.preparation_fault(error);
                }
                let mut accepted = 0;
                for (position, row) in rows.into_iter().enumerate() {
                    let index = self.batch_slots[position];
                    let sequence = self.slots[index].as_mut().expect("selected slot");
                    match row {
                        PreparedRow::Ready(item) => {
                            if item.output_budget != self.batch[position].output_budget {
                                self.output.unreserve(sequence.output);
                                self.output
                                    .reserve(sequence.output, item.output_budget as usize + 1);
                            }
                            sequence.work = WorkState::InFlight;
                            self.batch[accepted] = item;
                            self.batch_slots[accepted] = index;
                            accepted += 1;
                        }
                        PreparedRow::Deferred(wait) => {
                            self.output.unreserve(sequence.output);
                            sequence.resource_wait = Some(wait);
                            self.parked.push_back(index);
                        }
                        PreparedRow::Rejected(error) => {
                            self.output.unreserve(sequence.output);
                            self.terminate(index, FinishReason::Failed(error));
                        }
                    }
                }
                self.batch.truncate(accepted);
                self.batch_slots.truncate(accepted);
            }
        }
        if self.batch.is_empty() {
            if let Err(error) = self
                .model
                .as_mut()
                .expect("engine owns model")
                .abandon_preparation()
            {
                return self.preparation_fault(error);
            }
            return Ok(false);
        }
        Ok(true)
    }

    fn preparation_is_valid(&self, rows: &[PreparedRow]) -> bool {
        rows.len() == self.batch.len()
            && rows.iter().zip(&self.batch).zip(&self.batch_slots).all(
                |((row, offered), &index)| {
                    let PreparedRow::Ready(item) = row else {
                        return true;
                    };
                    let sequence = self.slots[index].as_ref().expect("selected slot");
                    if item.sequence != offered.sequence
                        || item.kind != offered.kind
                        || item.prefix != offered.prefix
                        || item.token_budget == 0
                        || item.token_budget > offered.token_budget
                        || item.output_budget > offered.output_budget
                    {
                        return false;
                    }
                    let Some(end) = item.prefix.checked_add(item.token_budget) else {
                        return false;
                    };
                    match item.kind {
                        StepKind::Prefill => {
                            end <= sequence.prompt_tokens
                                && item.output_budget == u32::from(end == sequence.prompt_tokens)
                        }
                        StepKind::Decode => item.output_budget == item.token_budget,
                    }
                },
            )
    }

    fn preparation_fault(&mut self, error: ExecutionError) -> Result<bool, EngineError> {
        self.fault_all(&error);
        self.flush_terminals()?;
        Err(EngineError::Faulted(error))
    }

    pub(super) fn reactivate_parked(&mut self) {
        for _ in 0..self.parked.len() {
            let index = self.parked.pop_front().expect("nonempty parked queue");
            let sequence = self.slots[index].as_mut().expect("parked slot");
            if !sequence
                .resource_wait
                .as_ref()
                .expect("parked readiness")
                .changed()
            {
                self.parked.push_back(index);
                continue;
            }
            sequence.resource_wait = None;
            if sequence.prefix < sequence.prompt_tokens {
                self.prefill.push_back(index);
            } else {
                self.decode.push_back(index);
            }
        }
    }
}
