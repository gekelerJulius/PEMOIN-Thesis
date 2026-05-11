"""
Graceful degradation policies for live execution scenarios.
"""


class DegradationPolicy:
    """Defines fallback strategies when modules miss latency budgets."""

    def apply(self, runtime_state):
        """Mutate runtime state or reuse prior outputs when needed."""
        raise NotImplementedError("Degradation policy logic will be implemented later.")

