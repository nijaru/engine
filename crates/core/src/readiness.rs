//! Explicit readiness state for serving/runtime startup.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReadinessState {
    Created,
    Loading,
    Preparing,
    Warming,
    Ready,
    Failed,
}

impl ReadinessState {
    #[must_use]
    pub const fn can_serve(self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RuntimeReadiness {
    state: ReadinessState,
}

impl Default for RuntimeReadiness {
    fn default() -> Self {
        Self {
            state: ReadinessState::Created,
        }
    }
}

impl RuntimeReadiness {
    #[must_use]
    pub const fn state(self) -> ReadinessState {
        self.state
    }

    #[must_use]
    pub const fn can_serve(self) -> bool {
        self.state.can_serve()
    }

    /// # Errors
    ///
    /// Returns [`ReadinessError::InvalidTransition`] when startup state would
    /// move backward or leave a terminal state.
    pub fn transition(&mut self, next: ReadinessState) -> Result<(), ReadinessError> {
        let valid = matches!(
            (self.state, next),
            (ReadinessState::Created, ReadinessState::Loading)
                | (ReadinessState::Created, ReadinessState::Failed)
                | (ReadinessState::Loading, ReadinessState::Preparing)
                | (ReadinessState::Loading, ReadinessState::Failed)
                | (ReadinessState::Preparing, ReadinessState::Warming)
                | (ReadinessState::Preparing, ReadinessState::Ready)
                | (ReadinessState::Preparing, ReadinessState::Failed)
                | (ReadinessState::Warming, ReadinessState::Ready)
                | (ReadinessState::Warming, ReadinessState::Failed)
        );
        if !valid {
            return Err(ReadinessError::InvalidTransition);
        }
        self.state = next;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReadinessError {
    InvalidTransition,
}

impl fmt::Display for ReadinessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition => f.write_str("runtime readiness transition is invalid"),
        }
    }
}

impl std::error::Error for ReadinessError {}
