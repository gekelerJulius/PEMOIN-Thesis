"""
Capability flags describing latency, quality, and resource characteristics.
"""


class CapabilityFlags:
    """Describes module performance characteristics for profile validation."""

    def __init__(self, latency_tier, quality_tier, memory_tier):
        self.latency_tier = latency_tier
        self.quality_tier = quality_tier
        self.memory_tier = memory_tier

