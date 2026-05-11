"""
Diagnostics bus for collecting timing, confidence, and debug visualization data.
"""


class DiagnosticsBus:
    """Shares diagnostic metrics between modules and optional visualization sinks."""

    def publish(self, record):
        """Publish a diagnostic record for downstream tooling."""
        raise NotImplementedError("Diagnostics publishing will be implemented later.")

    def subscribe(self, consumer):
        """Register a consumer interested in diagnostic updates."""
        raise NotImplementedError("Diagnostics subscription will be implemented later.")

