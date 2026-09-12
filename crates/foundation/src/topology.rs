use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeId(u32);

impl NodeId {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BackendId(String);

impl BackendId {
    /// # Errors
    /// Returns an error for an empty backend identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, TopologyError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(TopologyError::EmptyBackend);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DeviceId {
    node: NodeId,
    backend: BackendId,
    ordinal: u32,
}

impl DeviceId {
    #[must_use]
    pub fn new(node: NodeId, backend: BackendId, ordinal: u32) -> Self {
        Self {
            node,
            backend,
            ordinal,
        }
    }

    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    #[must_use]
    pub fn backend(&self) -> &BackendId {
        &self.backend
    }

    #[must_use]
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeDevice {
    id: DeviceId,
    memory_bytes: Option<u64>,
}

impl ComputeDevice {
    #[must_use]
    pub fn new(id: DeviceId, memory_bytes: Option<u64>) -> Self {
        Self { id, memory_bytes }
    }

    #[must_use]
    pub fn id(&self) -> &DeviceId {
        &self.id
    }

    #[must_use]
    pub const fn memory_bytes(&self) -> Option<u64> {
        self.memory_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceLink {
    from: DeviceId,
    to: DeviceId,
    bandwidth_bytes_per_second: Option<u64>,
    latency_nanos: Option<u64>,
    direct: bool,
}

impl DeviceLink {
    #[must_use]
    pub fn new(
        from: DeviceId,
        to: DeviceId,
        bandwidth_bytes_per_second: Option<u64>,
        latency_nanos: Option<u64>,
        direct: bool,
    ) -> Self {
        Self {
            from,
            to,
            bandwidth_bytes_per_second,
            latency_nanos,
            direct,
        }
    }

    #[must_use]
    pub fn from(&self) -> &DeviceId {
        &self.from
    }

    #[must_use]
    pub fn to(&self) -> &DeviceId {
        &self.to
    }

    #[must_use]
    pub const fn bandwidth_bytes_per_second(&self) -> Option<u64> {
        self.bandwidth_bytes_per_second
    }

    #[must_use]
    pub const fn latency_nanos(&self) -> Option<u64> {
        self.latency_nanos
    }

    #[must_use]
    pub const fn is_direct(&self) -> bool {
        self.direct
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceTopology {
    devices: Vec<ComputeDevice>,
    links: Vec<DeviceLink>,
}

impl ResourceTopology {
    /// # Errors
    /// Returns an error for duplicate devices or links to unknown devices.
    pub fn new(devices: Vec<ComputeDevice>, links: Vec<DeviceLink>) -> Result<Self, TopologyError> {
        let mut known = HashSet::with_capacity(devices.len());
        for device in &devices {
            if !known.insert(device.id().clone()) {
                return Err(TopologyError::DuplicateDevice);
            }
        }
        if links
            .iter()
            .any(|link| !known.contains(link.from()) || !known.contains(link.to()))
        {
            return Err(TopologyError::UnknownLinkEndpoint);
        }
        Ok(Self { devices, links })
    }

    #[must_use]
    pub fn devices(&self) -> &[ComputeDevice] {
        &self.devices
    }

    #[must_use]
    pub fn links(&self) -> &[DeviceLink] {
        &self.links
    }

    #[must_use]
    pub fn contains(&self, device: &DeviceId) -> bool {
        self.devices.iter().any(|known| known.id() == device)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopologyError {
    EmptyBackend,
    DuplicateDevice,
    UnknownLinkEndpoint,
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBackend => f.write_str("backend identity must not be empty"),
            Self::DuplicateDevice => f.write_str("resource topology contains a duplicate device"),
            Self::UnknownLinkEndpoint => {
                f.write_str("resource topology link references an unknown device")
            }
        }
    }
}

impl std::error::Error for TopologyError {}
