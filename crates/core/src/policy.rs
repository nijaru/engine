//! Immutable runtime-policy snapshots.

use std::fmt;

use crate::model::ModelCapabilities;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct PolicyVersion(u64);

impl PolicyVersion {
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateTierPreference {
    Device,
    Host,
    Automatic,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpeculationPolicy {
    Disabled,
    NativeMtp { max_draft_tokens: u8 },
}

impl SpeculationPolicy {
    #[must_use]
    pub const fn native_mtp(max_draft_tokens: u8) -> Option<Self> {
        if max_draft_tokens == 0 {
            None
        } else {
            Some(Self::NativeMtp { max_draft_tokens })
        }
    }
}

/// A snapshot is constructed once and passed to a scheduler/plan builder. It
/// has no mutation or global publication mechanism in this initial contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PolicySnapshot {
    version: PolicyVersion,
    max_batch_size: u32,
    max_batch_tokens: u32,
    state_tier: StateTierPreference,
    speculation: SpeculationPolicy,
}

impl PolicySnapshot {
    /// # Errors
    ///
    /// Returns a [`PolicyError`] when a batch or enabled speculation budget is
    /// zero.
    pub fn new(
        version: PolicyVersion,
        max_batch_size: u32,
        max_batch_tokens: u32,
        state_tier: StateTierPreference,
        speculation: SpeculationPolicy,
    ) -> Result<Self, PolicyError> {
        if max_batch_size == 0 || max_batch_tokens == 0 {
            return Err(PolicyError::ZeroBatchBudget);
        }
        if let SpeculationPolicy::NativeMtp {
            max_draft_tokens: 0,
        } = speculation
        {
            return Err(PolicyError::ZeroDraftBudget);
        }
        Ok(Self {
            version,
            max_batch_size,
            max_batch_tokens,
            state_tier,
            speculation,
        })
    }

    #[must_use]
    pub const fn version(self) -> PolicyVersion {
        self.version
    }

    #[must_use]
    pub const fn max_batch_size(self) -> u32 {
        self.max_batch_size
    }

    #[must_use]
    pub const fn max_batch_tokens(self) -> u32 {
        self.max_batch_tokens
    }

    #[must_use]
    pub const fn state_tier(self) -> StateTierPreference {
        self.state_tier
    }

    #[must_use]
    pub const fn speculation(self) -> SpeculationPolicy {
        self.speculation
    }

    /// Validate model-specific policy choices before preparing a plan.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError::SpeculationUnavailable`] when native MTP is not
    /// advertised, or [`PolicyError::DraftBudgetExceedsCapability`] when its
    /// budget is larger than the model capability.
    pub fn validate_for(&self, capabilities: ModelCapabilities) -> Result<(), PolicyError> {
        if let SpeculationPolicy::NativeMtp { max_draft_tokens } = self.speculation {
            let Some(mtp) = capabilities.mtp() else {
                return Err(PolicyError::SpeculationUnavailable);
            };
            if max_draft_tokens > mtp.max_draft_tokens() {
                return Err(PolicyError::DraftBudgetExceedsCapability);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PolicyError {
    ZeroBatchBudget,
    ZeroDraftBudget,
    SpeculationUnavailable,
    DraftBudgetExceedsCapability,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBatchBudget => f.write_str("batch budgets must be greater than zero"),
            Self::ZeroDraftBudget => {
                f.write_str("speculation draft budget must be greater than zero")
            }
            Self::SpeculationUnavailable => {
                f.write_str("native MTP is not advertised by the model")
            }
            Self::DraftBudgetExceedsCapability => {
                f.write_str("speculation draft budget exceeds model capability")
            }
        }
    }
}

impl std::error::Error for PolicyError {}
